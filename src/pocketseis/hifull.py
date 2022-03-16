import numpy as np
from scipy.signal import fftconvolve

from pyrocko import model as pmodel, orthodrome as pod
from pyrocko.guts import Object, Float
from torch.nn.functional import conv2d
import torch
from xarray import Dataset

from pocketseis.minimum_1d import HIFullMaterial
from pocketseis.source import SmoothRampSTF, GaussianSTF, ZeroCrossingSTF
from pocketseis.util import column_stack_3d, time_to_index, time_range


guts_prefix = 'pf'


def _get_relative_data(lat0, lon0, depth0, lats, lons, depths):
    # We use a (x=North, y=East, z=Down) coordinate system.
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)

    deltazs = depths - depth0
    deltaxs, deltays = pod.latlon_to_ne_numpy(lat0, lon0, lats, lons)

    # Position vectors relative to reference loc., array of shape (3, n_points)
    pos_vecs = np.vstack((deltaxs, deltays, deltazs))
    assert pos_vecs.shape == (3, lats.size)

    # 3-D distances from reference loc., array of shape (n_points)
    dists_3d = np.sqrt(np.sum(pos_vecs**2, axis=0, keepdims=False))

    # Unit vectors of direction cosines
    dircos_vecs = pos_vecs / dists_3d

    return (dists_3d, dircos_vecs)


class SurfaceDASCable(Object):
    nominal_zerooffset_loc = pmodel.Location.T(
        help='Geographycal location of first channel')
    nominal_length = Float.T(
        help='Cable length. Unit: [m]')
    azimuth = Float.T(
        help='Azimuth measured clockwise from the North(=x) and is '
             'defined as the angle between vector pointing from the '
             'interrogator to the North and the vector pointing from '
             'the interrogator to the other end of the cable. Unit: [deg]')
    depth = Float.T(
        help='Cable depth below surface (positive downward). Unit: [m]')
    channel_spacing = Float.T(
        help='Channel spacing. Unit: [m]')
    gauge_length = Float.T(
        help='Gauge length. Unit: [m]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = self._get_n_channels()
        (self.channel_locs,
            self._channel_lats,
            self._channel_lons) = self._get_channel_locs_lats_lons()
        self._real_length = None
        self._real_zerooffset_loc = None
        self._grid_spacing = None
        self._n_grids = None
        self._grid_locs = None
        self._grid_lats = None
        self._grid_lons = None

    def _get_n_channels(self):
        """
        Returns
        -------
        n : int
            Number of channels.
        """
        return int(round(self.nominal_length / self.channel_spacing)) + 1

    def _get_channel_locs_lats_lons(self):
        """
        Geographical locations of the channels, as well as their
        latitudes and longitudes.

        Returns
        -------
        out : tuple of 3
            0 -> List of :py:class:`pyrocko.model.Location` objects
            1 -> ndarray of channel latitudes
            2 -> ndarray of channel longitudes
        """
        cha_offsets = np.linspace(
            0.0, self.nominal_length, self.n_channels)

        # Channel lats & lons, 2-tuple with arrays of shape (n_channels,)
        cha_lats, cha_lons = pod.azidist_to_latlon(
            self.nominal_zerooffset_loc.effective_lat,
            self.nominal_zerooffset_loc.effective_lon,
            np.ones(self.n_channels) * self.azimuth,
            cha_offsets * pod.m2d)

        cha_locs = []
        for clat, clon in zip(cha_lats, cha_lons):
            cha_locs.append(pmodel.Location(lat=clat, lon=clon))

        return (cha_locs, cha_lats, cha_lons)

    @property
    def real_length(self):
        """
        Cable total length in m, from x=(first_channel - GL/2) to
        x=(last_channel + GL/2), where GL is the gauge length.

        Returns
        -------
        float
        """
        if self._real_length is None:
            self._real_length = self.nominal_length + self.gauge_length
        return self._real_length

    @property
    def real_zerooffset_loc(self):
        """
        Geographical location of point x=(first_channel - GL/2), where
        GL is the gauge length.

        Returns
        -------
        :py:class:`pyrocko.model.Location` object
        """
        if self._real_zerooffset_loc is None:
            lat0, lon0 = pod.azidist_to_latlon(
                self.nominal_zerooffset_loc.effective_lat,
                self.nominal_zerooffset_loc.effective_lon,
                self.azimuth + 180.0,
                self.gauge_length / 2.0 * pod.m2d)

            self._real_zerooffset_loc = pmodel.Location(lat=lat0, lon=lon0)

        return self._real_zerooffset_loc

    def _get_grid_locs_lats_lons(self, grid_spacing):
        n_grids = int(round(self.real_length / grid_spacing)) + 1
        grid_offsets = np.linspace(0.0, self.real_length, n_grids)

        # Grid-point lats & lons, 2-tuple with arrays of shape (n_grids,)
        grid_lats, grid_lons = pod.azidist_to_latlon(
            self.real_zerooffset_loc.effective_lat,
            self.real_zerooffset_loc.effective_lon,
            np.ones(n_grids) * self.azimuth,
            grid_offsets * pod.m2d)

        grid_locs = []
        for glat, glon in zip(grid_lats, grid_lons):
            grid_locs.append(pmodel.Location(lat=glat, lon=glon))

        return (grid_locs, grid_lats, grid_lons)

    @property
    def grid_spacing(self):
        """
        Grid spacing that is used to discretise the gauge length in
        order to calculate point strains.
        """
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, new_grid_spacing):
        grid_locs, grid_lats, grid_lons = (
            self._get_grid_locs_lats_lons(new_grid_spacing))
        self._grid_spacing = new_grid_spacing
        self._n_grids = len(grid_locs)
        self._grid_locs = grid_locs
        self._grid_lats = grid_lats
        self._grid_lons = grid_lons

    @property
    def grid_locs(self):
        """
        Geographical locations of the grid points to calculate point
        strains. They are set when `grid_spacing` is set.

        Returns
        -------
        list of :py:class:`pyrocko.model.Location` objects
        """
        assert self._grid_spacing is not None, (
            "cannot return grid locations when 'grid_spacing' is None. "
            "Set a value to 'grid_spacing' first.")
        return self._grid_locs

    def get_event_relative_data(self, event, level='channel'):
        """
        Parameters
        ----------
        event : :py:class:`pyrocko.model.Event` object
            Seismic event. Its location is considered as reference
            location (the origin of the Cartesian system).
        level : {'channel', 'grid'}
            Determines the observation points for which relative data
            are calculated.

        Returns
        -------
        dists_3d : ndarray of shape (n_receivers,)
            3-D distances of observation points from event. Unit: [m]
        dircos_vecs : ndarray of shape (3, n_receivers)
            Unit vectors of direction cosines. Relative event-receiver
            (either channels or grids) position vectors are in Cartesian
            coordinate system (i.e. vectors whose initial and terminal
            are event and observation-point positions, respectively).
            The indexes of the first dimension represent
            (0, 1, 2)->(North, East, Down) axes.
        """
        if level not in (valid_levels := ('channel', 'grid')):
            raise ValueError(f"Valid levels ar {valid_levels}")

        if level == 'grid':
            assert self._grid_spacing is not None, (
                "cannot set relative data for grids when 'gridspacing' "
                "is None. Set a value to 'grid_spacing' first.")

        if level == 'channel':
            lats = self._channel_lats
            lons = self._channel_lons
            depths = self.depth * np.ones(self.n_channels)
        else:
            lats = self._grid_lats
            lons = self._grid_lons
            depths = self.depth * np.ones(self._n_grids)

        return _get_relative_data(
            *event.effective_latlon, event.depth, lats, lons, depths)


def calc_radiation_pattern_factors(
        mt_symmat, dircos_vecs, quantity, far=True, intermediate=True,
        near=True, intermediate_far=True, intermediate_near=True):
    """
    Parameters
    ----------
    mt_symmat : ndarray of shape (3, 3)
        Seismic moment tensor as a plain **symmetric** matrix.
    dircos_vecs : ndarray of shape (3, n_receivers)
        Unit vectors of direction cosines.
    quantity : {'displacement', 'strain'}
        Measurement quantity type.

    Returns
    -------
    ds : :py:class:`xarray.Dataset` object
        Radiation pattern factores returned as an xarray Dataset.
        If `quantity='displacement'`, the Dataset keys are 'FP', 'FS',
        'IP', 'IS' and 'N'.
        If `quantity='strain'`, the Dataset keys are 'FP', 'FS', 'IFP',
        'IFS', 'INP', 'INS' and 'N'.
        Each key of the Dataser is mapped to a 2-D array of shape
        `(3, n_receivers)`. The dimesion names of each 2-D array are
        `'axis'` (with size equals to 3) and `'i_reciever'` (with size
        equals to `n_receivers`). The coordinate names of dimension
        `axis` are 'x', 'y' and 'z' (indices of 0, 1 and 2,
        respectively).
    """

    # Dimension and coordinate names of output dataset
    dims = ['axis', 'i_reciever']
    coords = {'axis': ['x', 'y', 'z']}

    # Cache common terms to save computation time
    # Array shapes are M::(3, 3), Γ::(n_rec, 3, 1), ΓT::(n_rec, 1, 3)
    M = np.asarray(mt_symmat, dtype=np.float64)
    Γ = column_stack_3d(np.asarray(dircos_vecs, dtype=np.float64))
    ΓT = Γ.transpose((0, 2, 1))

    n_rec = dircos_vecs.shape[1]
    assert M.shape == (3, 3)
    assert Γ.shape == (n_rec, 3, 1)
    assert ΓT.shape == (n_rec, 1, 3)

    q1 = M @ Γ
    k1 = ΓT @ q1   # shape of (n_rec, 1, 1)
    k2 = np.trace(M)
    q2 = k1 * Γ
    q3 = k2 * Γ

    if quantity == 'strain':
        J = np.ones((3, 1), dtype=np.float64)
        q4 = k1 * J
        q5 = k2 * J
        q6 = M.diagonal()[:, np.newaxis]

        if far is True:
            # Far-field P-wave strain (FP ∝ 1/r)
            A_fp = q2
            B_fp = A_fp * Γ

            # Far-field S-wave strain (FS ∝ 1/r)
            A_fs = q2 - q1
            B_fs = A_fs * Γ
        else:
            B_fp = B_fs = np.zeros_like(Γ)

        if intermediate_far is True:
            # Intermediate-far field P-wave strain (IFP ∝ 1/r²)
            A_ip = (6.0 * q2) - q3 - (2.0 * q1)
            B_ifp = q4 + ((-4.0 * q2) + (2.0 * q1) - A_ip) * Γ

            # Intermediate-far field S-wave strain (IFS ∝ 1/r²)
            A_is = (6.0 * q2) - q3 - (3.0 * q1)
            B_ifs = q4 + ((-4.0 * q2) + (4.0 * q1) - A_is) * Γ - q6
        else:
            B_ifp = B_ifs = np.zeros_like(Γ)

        if intermediate_near is True:
            # Intermediate-near field P-wave strain (INP ∝ 1/r³)
            A_n = 3.0 * ((5.0 * q2) - q3 - (2.0 * q1))
            B_inp = (
                (6.0 * q4)
                + ((-30.0 * q2) + (18.0 * q1) + (3.0 * q3) - A_n) * Γ
                - q5 - (2.0 * q6))

            # Intermediate-near field S-wave strain (INS ∝ 1/r³)
            B_ins = (
                (6.0 * q4)
                + ((-30.0 * q2) + (17.0 * q1) + (3.0 * q3) - A_n) * Γ
                - q5 - (3.0 * q6))
        else:
            B_inp = B_ins = np.zeros_like(Γ)

        if near is True:
            # Near-field strain (N ∝ 1/r⁵)
            B_n = (
                (15.0 * q4) + 15.0 * ((-7.0 * q2) + (4.0 * q1) + q3) * Γ
                - (3.0 * q5) - (6.0 * q6))
        else:
            B_n = np.zeros_like(Γ)

        # Reshape arrays from (n_rec, 3, 1) to (3, n_rec) and save
        # them into a `xr.DataSet` with dims & coords defined above.
        data_vars = {
            'FP': (dims, B_fp.squeeze(axis=2).T),
            'FS': (dims, B_fs.squeeze(axis=2).T),
            'IFP': (dims, B_ifp.squeeze(axis=2).T),
            'IFS': (dims, B_ifs.squeeze(axis=2).T),
            'INP': (dims, B_inp.squeeze(axis=2).T),
            'INS': (dims, B_ins.squeeze(axis=2).T),
            'N': (dims, B_n.squeeze(axis=2).T)}

        return Dataset(data_vars=data_vars, coords=coords)

    # ----------
    elif quantity == 'displacement':
        if far is True:
            # Far-field displacements (FP & FS ∝ 1/r)
            A_fp = q2
            A_fs = q2 - q1
        else:
            A_fp = A_fs = np.zeros_like(Γ)

        if intermediate is True:
            # Intermediate-field displacement (IP & IS ∝ 1/r²)
            A_ip = (6.0 * q2) - q3 - (2.0 * q1)
            A_is = (6.0 * q2) - q3 - (3.0 * q1)
        else:
            A_ip = A_is = np.zeros_like(Γ)

        if near is True:
            # Near-field displacement (N ∝ 1/r⁴)
            A_n = 3.0 * ((5.0 * q2) - q3 - (2.0 * q1))
        else:
            A_n = np.zeros_like(Γ)

        # Reshape arrays from (n_rec, 3, 1) to (3, n_rec) and
        # save them into a `xr.DataSet`.
        data_vars = {
            'FP': (dims, A_fp.squeeze(axis=2).T),
            'FS': (dims, A_fs.squeeze(axis=2).T),
            'IP': (dims, A_ip.squeeze(axis=2).T),
            'IS': (dims, A_is.squeeze(axis=2).T),
            'N': (dims, A_n.squeeze(axis=2).T)}

        return Dataset(data_vars=data_vars, coords=coords)


def pad_stf(tphase, stf_amps, deltat, total_len):
    """
    Pad source-time function to obtain intermediate- or far-field motion
    at a fixed receiver. It is assumed that reference time is zero. This
    means that `tphase` is phase travel-time from source to receiver.

    Parameters
    ----------
    tphase : float
        Phase (P or S) travel-time in s. It's either `r/α` or `r/β`,
        where `r` is source-reciver 3-D distance, and `α` and `β` are
        P- and S-wave velocities, respectively.
    stf_amps : array-like of shape (n_samples,)
        Source-time function amplitudes.
    deltat : float
        Time-sampling interval. Unit: [s]
    total_len : int
        Length of motion time-function in *number of samples* (rather
        than s). This is equal to a time-length of `r/β + T`, where `T`
        is the duration of source-time function in s.

    Returns
    -------
    out : ndarray of shape (total_len,)
        Padded source-time function.
    """
    before = time_to_index(tphase, deltat)
    after = total_len - (before + stf_amps.size)

    return np.pad(
        stf_amps, pad_width=(before, after), mode='constant',
        constant_values=(0.0, stf_amps[-1]))


def convolve_stf(tp, ts, stf_amps, deltat, total_len):
    """
    Convolve source-time function with time to obtain near-field motion
    at a fixed receiver. It is assumed that reference time is zero. This
    means that `tp` and `ts` are P- and S-wave travel-times from source
    to receiver.

    Parameters
    ----------
    tp : float
        P-wave travel-time in s.
    ts : float
        S-wave travel-time in s.
    stf_amps : array-like of shape (n_samples,)
        Source-time function amplitudes.
    deltat : float
        Time-sampling interval. Unit: [s]
    total_len : int
        Length of motion time-function in *number of samples* (rather
        than sec). This is equal to a time-length of `r/β + T`, where
        `T` is the duration of source-time function in sec.
    """
    tau = time_range(tp, ts, deltat)

    # Repeat end point to prevent boundary effects
    pady = np.pad(stf_amps, (0, tau.size), mode='edge')
    convy = fftconvolve(pady, tau)[:-tau.size] * deltat

    before = time_to_index(tp, deltat)
    after = total_len - (before + convy.size)

    return np.pad(
        convy, pad_width=(before, after), mode='constant',
        constant_values=(0.0, convy[-1]))


class HIFullScenario(Object):
    """
    Base class for forward modelling scenario for homogeneous,
    isotropic, unbounded (full-space) elastic medium.
    """
    deltat = Float.T(help='Time-sampling interval. Unit: [s]')
    material = HIFullMaterial.T(
        default=HIFullMaterial(),
        help='Isotropic elastic material')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Reciprocals of the material parameters
        self._recips = {
            '1/4πρ': 1.0 / (4.0 * np.pi * self.material.rho),
            '1/α': 1.0 / self.material.vp,
            '1/β': 1.0 / self.material.vs}

    def _cache_dist_recips(self, dists_3d, quantity):
        """
        Cache reciprocals of powers of 3-D distances.
        Arrays of shape (n_receivers, 1, 1).

        Parameters
        ----------
        dists_3d : ndarray of shape (n_receivers,)
            3-D distances of observation points from event. Unit: [m]
        quantity : {'displacement', 'strain'}
            Measurement quantity type.

        Returns
        -------
        None
        """
        self._recips['1/r'] = 1.0 / dists_3d.reshape(-1, 1, 1)
        self._recips['1/r2'] = pow(self._recips['1/r'], 2)

        if quantity == 'displacement':
            self._recips['1/r4'] = pow(self._recips['1/r'], 4)
        elif quantity == 'strain':
            self._recips['1/r3'] = pow(self._recips['1/r'], 3)
            self._recips['1/r5'] = pow(self._recips['1/r'], 5)
        else:
            raise ValueError(
                "'quantity' must be either 'displacement' or 'strain'")

    def _calc_ptimes(self, dists_3d):
        """
        P-wave travel times.

        Parameters
        ----------
        dists_3d : ndarray of shape (n_receivers,)
            3-D distances of observation points from event. Unit: [m]

        Returns
        -------
        tp_all : ndarrays of shape (n_receivers,)
            P-wave travel-times.
        """
        return dists_3d * self._recips['1/α']

    def _calc_stimes(self, dists_3d):
        """
        S-wave travel times.

        Parameters
        ----------
        dists_3d : ndarray of shape (n_receivers,)
            3-D distances of observation points from event. Unit: [m]

        Returns
        -------
        ts_all : ndarrays of shape (n_receivers,)
            S-wave travel-times.
        """
        return dists_3d * self._recips['1/β']


class StrainHIFullScenario(HIFullScenario):
    """
    Forward-modelling scenario to calculate strains in an homogeneous,
    isotropic, unbounded (full-space) elastic medium.

    Note
    ----
    2021-10-15: Only supports surface DAS-cable (zero dip angle), for
    which a 2-D rotation into cable axis is applied. To generalise to a
    DAS cable with an arbitrary dip angle, rotation in 3-D should be
    applied. For a borehole DAS cable, the z component is the desired
    modelled strain.
    """

    def process(
            self, source, cable, stf_duration, far=True, intermediate_far=True,
            intermediate_near=True, near=True):
        """
        Parameters
        ----------
        source :
            Seismic point-source object. It must provide a method
            `pyrocko_events()`, which returns
            :py:class:`pyrocko.MomentTensor` object.
        cable :
            DAS cable. Grid spacing must be set before passing it to
            this method.
        stf_duration : flot
            Source-time function duration in [s].

        Returns
        -------
        ds : :py:class:`xarray.Dataset` object
            Longitudinal strains (along cable axis) as an `xr.Dataset`.
            The Dataset keys are 'F', 'IF', 'IN', 'N' and 'total'. Each
            key is mapped to a 2-D array of shape `(n_channels, n_times)`.
            The dimesion names of each 2-D array are `'i_reciever'` and
            `'time'`.

        Notes
        -----
          * The system time of fisrt sample is zero (i.e. relative to
          source onset time), and the system time of last sample depends
          on the STF length. These values can be easily adapted later
          with respect to P-wave arrival time (``tp``) and S-wave arrival
          time plus STF duration (``ts+T``).
        """
        event = source.pyrocko_event()
        dists_3d, dircos_vecs = \
            cable.get_event_relative_data(event, level='grid')

        # Cache reciprocals of powers of 3-D distances.
        self._cache_dist_recips(dists_3d=dists_3d, quantity='strain')

        # Radiation-pattern factors
        mt_symmat = np.asarray(source.pyrocko_moment_tensor().m())
        xset = calc_radiation_pattern_factors(
            mt_symmat, dircos_vecs, quantity='strain',
            far=far, intermediate_far=intermediate_far,
            intermediate_near=intermediate_near, near=near)

        # Time-dependent seismic moment, M(t). It should be normalised
        # to one, otherwise the source magnitude becomes meaningless.
        stf_ramp = SmoothRampSTF(duration=stf_duration, anchor=-1.0)
        _, D = stf_ramp.discretize_t(self.deltat, 0.0, normalise=False)
        assert D.max() == 1.0, 'Seismic moment STF should be normalized to 1'

        # Seismic moment-rate, dM(t)/dt
        stf_gaus = GaussianSTF(duration=stf_duration, anchor=-1.0)
        _, Ddot = stf_gaus.discretize_t(self.deltat, 0.0, normalise=False)

        # Time-derivative of the moment rate, d²M(t)/dt²
        stf_zcros = ZeroCrossingSTF(duration=stf_duration, anchor=-1.0)
        _, Dddot = stf_zcros.discretize_t(self.deltat, 0.0, normalise=False)

        # P- and S-wave travel times (flattened arrays)
        tp_all = self._calc_ptimes(dists_3d=dists_3d)
        ts_all = self._calc_stimes(dists_3d=dists_3d)

        # Number of time samples (longest waveform)
        ts_max = ts_all.max()
        data_len = time_to_index(ts_max, self.deltat) + D.size

        c = self._recips
        deltat = self.deltat
        n_rec = dists_3d.size

        if far is True:
            # Far-field strain (ε_f ∝ 1/r)

            T_fp = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            T_fs = np.zeros_like(T_fp)
            for i_rec in range(n_rec):
                T_fp[i_rec] = pad_stf(tp_all[i_rec], Dddot, deltat, data_len)
                T_fs[i_rec] = pad_stf(ts_all[i_rec], Dddot, deltat, data_len)

            B_fp = column_stack_3d(xset['FP'].values)
            B_fs = column_stack_3d(xset['FS'].values)

            ε_fp = -c['1/4πρ'] * c['1/α']**4 * c['1/r'] * B_fp * T_fp
            ε_fs = +c['1/4πρ'] * c['1/β']**4 * c['1/r'] * B_fs * T_fs

            ε_f = ε_fp + ε_fs
        else:
            ε_f = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        if intermediate_far is True:
            # Intermediate-far field strain (ε_if ∝ 1/r²)

            T_ifp = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            T_ifs = np.zeros_like(T_ifp)
            for i_rec in range(n_rec):
                T_ifp[i_rec] = pad_stf(tp_all[i_rec], Ddot, deltat, data_len)
                T_ifs[i_rec] = pad_stf(ts_all[i_rec], Ddot, deltat, data_len)

            B_ifp = column_stack_3d(xset['IFP'].values)
            B_ifs = column_stack_3d(xset['IFS'].values)

            ε_ifp = +c['1/4πρ'] * c['1/α']**3 * c['1/r2'] * B_ifp * T_ifp
            ε_ifs = -c['1/4πρ'] * c['1/β']**3 * c['1/r2'] * B_ifs * T_ifs

            ε_if = ε_ifp + ε_ifs
        else:
            ε_if = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        if intermediate_near is True:
            # Intermediate-near field strain (ε_in ∝ 1/r³)

            T_inp = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            T_ins = np.zeros_like(T_inp)
            for i_rec in range(n_rec):
                T_inp[i_rec] = pad_stf(tp_all[i_rec], D, deltat, data_len)
                T_ins[i_rec] = pad_stf(ts_all[i_rec], D, deltat, data_len)

            B_inp = column_stack_3d(xset['INP'].values)
            B_ins = column_stack_3d(xset['INS'].values)

            ε_inp = +c['1/4πρ'] * c['1/α']**2 * c['1/r3'] * B_inp * T_inp
            ε_ins = -c['1/4πρ'] * c['1/β']**2 * c['1/r3'] * B_ins * T_ins

            ε_in = ε_inp + ε_ins
        else:
            ε_in = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        if near is True:
            # Near-field strain (ε_n ∝ 1/r⁵)

            T_n = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            for i_rec in range(n_rec):
                T_n[i_rec] = convolve_stf(
                    tp_all[i_rec], ts_all[i_rec], D, deltat, data_len)

            B_n = column_stack_3d(xset['N'].values)

            ε_n = +c['1/4πρ'] * c['1/r5'] * B_n * T_n
        else:
            ε_n = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        # Important: this is only for surface DAS cable.
        # Rotate horizontal components (N, E)=(x, y) into axial
        # direction along the fiber and get rid of z component. Shapes
        # will change from (n_rec, 3, data_len) to (n_rec, 1, data_len)
        ϕ = np.deg2rad(cable.azimuth)
        R = np.array([np.cos(ϕ), np.sin(ϕ), 0.0])[np.newaxis, :]
        ε_f = R @ ε_f
        ε_if = R @ ε_if
        ε_in = R @ ε_in
        ε_n = R @ ε_n

        # Apply moving average
        # Reshape arrays (n_rec, 1, data_len)->(1, 1, n_rec, data_len).
        # Concatenate along axis=0. shape(x_in)==(4, 1, n_rec, data_len)
        taxes = (1, 0, 2)
        x_in = np.concatenate((
            ε_f.transpose(taxes)[np.newaxis, :, :, :],
            ε_if.transpose(taxes)[np.newaxis, :, :, :],
            ε_in.transpose(taxes)[np.newaxis, :, :, :],
            ε_n.transpose(taxes)[np.newaxis, :, :, :]), axis=0)

        # Kernel height and width
        kH = int(round(cable.gauge_length / cable.grid_spacing)) + 1
        kW = 1
        kernel = np.ones((1, 1, kH, kW), dtype=np.float64)

        # Stride height and width
        sH = int(round(cable.channel_spacing / cable.grid_spacing))
        sW = 1

        x_out = conv2d(
            input=torch.from_numpy(x_in),
            weight=torch.from_numpy(kernel),
            stride=(sH, sW), padding=0)
        x_out = x_out.numpy()

        # Save results into a `xr.DataSet` with following dimensions and
        # coordinates. Strains are longitudinal only (along the cable).
        dims = ['i_reciever', 'time']
        coords = {'time': np.arange(data_len) * self.deltat}
        data_vars = {
            'F': (dims, x_out[0].squeeze(axis=0)),
            'IF': (dims, x_out[1].squeeze(axis=0)),
            'IN': (dims, x_out[2].squeeze(axis=0)),
            'N': (dims, x_out[3].squeeze(axis=0)),
            'total': (dims, x_out.sum(axis=0).squeeze(axis=0))}

        return Dataset(data_vars=data_vars, coords=coords)


class DisplacementHIFullScenario(HIFullScenario):
    """
    Forward-modelling scenario to calculate displacements in an
    homogeneous, isotropic, unbounded (full-space) elastic medium.
    """

    def process(
            self, source, receivers, stf_duration,
            far=True, intermediate=True, near=True):
        """
        Parameters
        ----------
        source :
            Seismic point-source object. It must provide a method
            `pyrocko_events()`, which returns
            :py:class:`pyrocko.MomentTensor` object.
        receivers : list of :py:class:`pyrocko.model.Station` objects
            Seismic sensors.
        stf_duration : float
            Source-time function duration in [s].

        Returns
        -------
        ds : :py:class:`xarray.Dataset` object
            Displacements, measured in m, in (x=North, y=East, z=Down)
            Cartesian coordinate system as an `xr.Dataset`. The Dataset
            keys are 'F', 'I', 'N' and 'total'. Each key is mapped to a
            3-D array of shape `(n_receivers, 3, n_times)`. The dimesion
            ames of each 3-D array are `'i_reciever'`, `'axis'` and
            `'time'`. The indices of the `'axis'` dimension represent
            (0, 1, 2)->(x=North, y=East, z=Down).

        Notes
        -----
          * The system time of fisrt sample is zero (i.e. relative to
          source onset time), and the system time of last sample depends
          on the STF length. These values can be easily adapted later
          with respect to P-wave arrival time (``tp``) and S-wave arrival
          time plus STF duration (``ts+T``).
        """
        event = source.pyrocko_event()
        a = [rec.effective_latlon + (rec.elevation,) for rec in receivers]
        rlats, rlons, rdepths = zip(*a)
        dists_3d, dircos_vecs = _get_relative_data(
            *event.effective_latlon, event.depth, rlats, rlons, rdepths)

        # Cache reciprocals of powers of 3-D distances.
        self._cache_dist_recips(dists_3d=dists_3d, quantity='displacement')

        # Radiation-pattern factors
        mt_symmat = np.asarray(source.pyrocko_moment_tensor().m())
        xset = calc_radiation_pattern_factors(
            mt_symmat, dircos_vecs, quantity='displacement',
            far=far, intermediate=intermediate, near=near)

        # Time-dependent seismic moment, M(t). It should be normalised
        # to one, otherwise the source magnitude becomes meaningless.
        stf_ramp = SmoothRampSTF(duration=stf_duration, anchor=-1.0)
        _, D = stf_ramp.discretize_t(self.deltat, 0.0, normalise=False)
        assert D.max() == 1.0, 'Seismic moment STF should be normalized to 1'

        # Seismic moment-rate, dM(t)/dt
        stf_gaus = GaussianSTF(duration=stf_duration, anchor=-1.0)
        _, Ddot = stf_gaus.discretize_t(self.deltat, 0.0, normalise=False)

        # P- and S-wave travel times (flattened arrays)
        tp_all = self._calc_ptimes(dists_3d=dists_3d)
        ts_all = self._calc_stimes(dists_3d=dists_3d)

        # Number of time samples (longest waveform)
        ts_max = ts_all.max()
        data_len = time_to_index(ts_max, self.deltat) + D.size

        c = self._recips
        deltat = self.deltat
        n_rec = dists_3d.size

        if far is True:
            # Far-field displacement (u_f ∝ 1/r)

            T_fp = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            T_fs = np.zeros_like(T_fp)
            for i_rec in range(n_rec):
                T_fp[i_rec] = pad_stf(tp_all[i_rec], Ddot, deltat, data_len)
                T_fs[i_rec] = pad_stf(ts_all[i_rec], Ddot, deltat, data_len)

            A_fp = column_stack_3d(xset['FP'].values)
            A_fs = column_stack_3d(xset['FS'].values)

            u_fp = +c['1/4πρ'] * c['1/α']**3 * c['1/r'] * A_fp * T_fp
            u_fs = -c['1/4πρ'] * c['1/β']**3 * c['1/r'] * A_fs * T_fs

            u_f = u_fp + u_fs
        else:
            u_f = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        if intermediate is True:
            # Intermediate-field displacement (u_i ∝ 1/r²)

            T_ip = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            T_is = np.zeros_like(T_ip)
            for i_rec in range(n_rec):
                T_ip[i_rec] = pad_stf(tp_all[i_rec], D, deltat, data_len)
                T_is[i_rec] = pad_stf(ts_all[i_rec], D, deltat, data_len)

            A_ip = column_stack_3d(xset['IP'].values)
            A_is = column_stack_3d(xset['IS'].values)

            u_ip = +c['1/4πρ'] * c['1/α']**2 * c['1/r2'] * A_ip * T_ip
            u_is = -c['1/4πρ'] * c['1/β']**2 * c['1/r2'] * A_is * T_is

            u_i = u_ip + u_is
        else:
            u_i = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        if near is True:
            # Near-field displacement (u_n ∝ 1/r⁵)

            T_n = np.zeros((n_rec, 1, data_len), dtype=np.float64)
            for i_rec in range(n_rec):
                T_n[i_rec] = convolve_stf(
                    tp_all[i_rec], ts_all[i_rec], D, deltat, data_len)

            A_n = column_stack_3d(xset['N'].values)

            u_n = +c['1/4πρ'] * c['1/r4'] * A_n * T_n
        else:
            u_n = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        # Total-field displacement
        u_total = u_f + u_i + u_n

        # Save results into a `xr.DataSet` with following dimensions and
        # coordinates. Each array is of shape (n_receivers, 3, data_len).
        dims = ['i_reciever', 'axis', 'time']
        coords = {
            'axis': ['x', 'y', 'z'],
            'time': np.arange(data_len) * self.deltat}
        data_vars = {
            'F': (dims, u_f),
            'I': (dims, u_i),
            'N': (dims, u_n),
            'total': (dims, u_total)}

        return Dataset(data_vars=data_vars, coords=coords)
