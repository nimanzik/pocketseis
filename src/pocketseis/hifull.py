import numpy as np
from scipy.signal import fftconvolve

from pyrocko import model as pmodel, orthodrome as pod
from pyrocko.guts import Float, Object
from torch.nn.functional import avg_pool2d
import torch
import xarray as xr

from pocketseis.minimum_1d import HIFullMaterial
from pocketseis.rotation import cartesian_rotmat
from pocketseis.source import SmoothRampSTF, GaussianSTF, ZeroCrossingSTF
from pocketseis.util import column_stack_3d, time_to_index, time_range


guts_prefix = 'pf'


def _get_relative_data(lat0, lon0, depth0, lats, lons, depths):
    """
    Calculates the 3-D distances and directional cosines of a set of
    observation points (potential receivers) from a reference location
    (potential seismic source) in a Cartesian coordinate system defined
    as (x, y, z)=(North, East, Down).

    Parameters
    ----------
    lat0, lon0, depth0: float
        Geographical coordinates of the reference point.
    lats, lons, depths : list of floats
        Geographical coordinates of the observation points.

    Returns
    -------
    dists_3d : ndarray of shape (n_receivers,)
        3-D distances of observation points from event. Unit: [m]
    dircos_vecs : ndarray of shape (3, n_receivers)
        Unit vectors of direction cosines. Relative event-receiver
        position vectors are in Cartesian coordinate system (i.e.
        vectors whose initial and terminal are event and observation
        points, respectively). The indexes of the first dimension
        represent (0, 1, 2)->(North, East, Down) axes.
    """
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


class BaseDASCable(Object):
    nominal_refloc = pmodel.Location.T(
        default=pmodel.Location(lat=0.0, lon=0.0, depth=0.0),
        help='Geographycal location of the first channel')
    nominal_len = Float.T(
        default=1000.0,
        help='Cable length between the first and last channels. Unit: [m]')
    channel_spacing = Float.T(
        default=1.0,
        help='Channel spacing. Unit: [m]')
    gauge_len = Float.T(
        default=10.0,
        help='Gauge length. Unit: [m]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_spacing = None
        self.grid_locs = []

    @property
    def n_channels(self):
        return int(round(self.nominal_len / self.channel_spacing)) + 1

    @property
    def effective_len(self):
        """
        Returns
        -------
        Effective length in m, from `xi=(first_channel - GL/2)` to
        `xf =(last_channel + GL/2)`, where `GL` is the gauge length.
        """
        return self.nominal_len + self.gauge_len


class SurfaceDASCable(BaseDASCable):
    azimuth = Float.T(
        default=0.0,
        help='Azimuth measured clockwise from the North(=x) and is '
             'defined as the angle between vector pointing from the '
             'interrogator to the North and the vector pointing from '
             'the interrogator to the other end of the cable. Unit: [deg]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cable depth below surface (positive downward). Unit: [m]
        self.depth = self.nominal_refloc.depth
        self.effective_refloc = self._get_effective_refloc()
        self.channel_locs = self._get_channel_locs()

    def _get_effective_refloc(self):
        """
        Geographical location of point `x=(first_channel - GL/2)`, where
        `GL` is the gauge length.

        Returns
        -------
        :py:class:`pyrocko.model.Location` object
        """
        lat0, lon0 = pod.azidist_to_latlon(
            self.nominal_refloc.effective_lat,
            self.nominal_refloc.effective_lon,
            self.azimuth + 180.0,
            self.gauge_len / 2.0 * pod.m2d)

        return pmodel.Location(lat=lat0, lon=lon0, depth=self.depth)

    def _get_channel_locs(self):
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
        cha_offsets = np.linspace(0.0, self.nominal_len, self.n_channels)

        # Channel lats & lons, 2-tuple with arrays of shape (n_channels,)
        cha_lats, cha_lons = pod.azidist_to_latlon(
            self.nominal_refloc.effective_lat,
            self.nominal_refloc.effective_lon,
            np.ones(self.n_channels) * self.azimuth,
            cha_offsets * pod.m2d)

        cha_locs = [
            pmodel.Location(lat=clat, lon=clon, depth=self.depth)
            for clat, clon in zip(cha_lats, cha_lons)]

        return cha_locs

    def set_grid_spacing(self, new_grid_spacing):
        """
        Set grid spacing in [m] that is used for discretising the gauge
        length to calculate point strains.
        """
        if self.channel_spacing % new_grid_spacing == 0.0:
            # Some grids overlap. Can use moving average
            n_grids = int(round(self.effective_len / new_grid_spacing)) + 1
            grid_offsets = np.linspace(0.0, self.effective_len, n_grids)
        else:
            # Grids do not overlap. Must take average over separate GLs
            # In this case, n_rec = n_channels * n_grids_per_gl
            n_grids_per_gl = int(round(self.gauge_len / new_grid_spacing)) + 1
            a = np.arange(self.n_channels)[:, None] * self.channel_spacing
            b = np.linspace(0.0, self.gauge_len, n_grids_per_gl)
            grid_offsets = (a + b).flatten()

        # Grid lats & lons, 2-tuple of arrays of shape (n_grids,)
        grid_lats, grid_lons = pod.azidist_to_latlon(
            self.effective_refloc.effective_lat,
            self.effective_refloc.effective_lon,
            np.ones_like(grid_offsets) * self.azimuth,
            grid_offsets * pod.m2d)

        grid_locs = [
            pmodel.Location(lat=glat, lon=glon, depth=self.depth)
            for glat, glon in zip(grid_lats, grid_lons)]

        self.grid_spacing = new_grid_spacing
        self.grid_locs = grid_locs

    def get_event_relative_data(self, event, level='channel'):
        """
        Parameters
        ----------
        event : :py:class:`pyrocko.model.Event` object
            Seismic event. Its location is considered as the origin of
            the Cartesian system to obtain relative data.
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
            are event and observation  points, respectively). The
            indexes of the first dimension represent
            (0, 1, 2)->(North, East, Down) axes.
        """
        if level not in (valid_levels := ('channel', 'grid')):
            raise ValueError(f"Valid levels are {valid_levels}")

        if level == 'grid':
            assert self.grid_spacing is not None, (
                "cannot set relative data for grids when 'grid_spacing' "
                "is None. Run 'set_grid_spacing()' first")

        if level == 'channel':
            rdepths = np.ones_like(self.channel_locs) * self.depth
            rlats, rlons = zip(
                *[c.effective_latlon for c in self.channel_locs])
        else:
            rdepths = np.ones_like(self.grid_locs) * self.depth
            rlats, rlons = zip(*[g.effective_latlon for g in self.grid_locs])

        return _get_relative_data(
            *event.effective_latlon, event.depth, rlats, rlons, rdepths)


def calc_radiation_patterns_mt(
        mt_symmat, dircos_vecs, quantity, far=True, intermediate=True,
        near=True, intermediate_far=True, intermediate_near=True):
    """
    Radiation patterns of displacement field or *normal* strain field
    (εᵢᵢ; i∈{x, y, z}) from a moment-tensor point source.

    Parameters
    ----------
    mt_symmat : ndarray of shape (3, 3)
        Seismic moment tensor as a plain *symmetric* matrix.
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

    if quantity not in (valid_quants := ('displacement', 'strain')):
        raise ValueError(f"Valid quantities are: {valid_quants}")

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
    k2 = M.trace()
    q2 = k1 * Γ
    q3 = k2 * Γ

    # Dimension and coordinate names when saving RPs into `xr.DataSet`
    dims = ['axis', 'i_reciever']
    coords = {'axis': ['x', 'y', 'z']}

    # ----------
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
            k: (dims, v.squeeze(axis=2).T)
            for k, v in zip(
                ['FP', 'FS', 'IFP', 'IFS', 'INP', 'INS', 'N'],
                [B_fp, B_fs, B_ifp, B_ifs, B_inp, B_ins, B_n])}

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
        # save them into a `xr.DataSet`
        data_vars = {
            k: (dims, v.squeeze(axis=2).T)
            for k, v in zip(
                ['FP', 'FS', 'IP', 'IS', 'N'],
                [A_fp, A_fs, A_ip, A_is, A_n])}

    return xr.Dataset(data_vars=data_vars, coords=coords)


def pad_stf(t_phase, stf_amps, deltat, total_len):
    """
    Pad source-time function to obtain intermediate- or far-field motion
    at a fixed receiver. It is assumed that reference time is zero. This
    means that `t_phase` is phase travel-time from source to receiver.

    Parameters
    ----------
    t_phase : float
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
    before = time_to_index(t_phase, deltat)
    after = total_len - (before + stf_amps.size)

    return np.pad(
        stf_amps,
        pad_width=(before, after),
        mode='constant',
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
        convy,
        pad_width=(before, after),
        mode='constant',
        constant_values=(0.0, convy[-1]))


class HIFullScenario(Object):
    """
    Base class for forward modelling scenario for homogeneous,
    isotropic, unbounded (full-space) elastic medium.
    """
    deltat = Float.T(help='Time-sampling interval. Unit: [s]')
    material = HIFullMaterial.T(help='Isotropic elastic material')

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
        if quantity not in (valid_quants := ('displacement', 'strain')):
            raise ValueError(f"Valid quantities are: {valid_quants}")

        self._recips['1/r'] = 1.0 / dists_3d.reshape(-1, 1, 1)
        self._recips['1/r2'] = pow(self._recips['1/r'], 2)

        if quantity == 'displacement':
            self._recips['1/r4'] = pow(self._recips['1/r'], 4)
        elif quantity == 'strain':
            self._recips['1/r3'] = pow(self._recips['1/r'], 3)
            self._recips['1/r5'] = pow(self._recips['1/r'], 5)

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
    2022-04-08: Only supports surface DAS cable (zero dip angle) and
    borehole DAS cable (vertical dip angle). To generalise to a DAS
    cable with an arbitrary azimuth and dip angle, two consecutive
    rotations in 3-D should be applied.
    """

    def process(
            self, source, cable, stf_duration, far=True,
            intermediate_far=True, intermediate_near=True, near=True):
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

        # Cache reciprocals of powers of 3-D distances
        self._cache_dist_recips(dists_3d=dists_3d, quantity='strain')

        # We assume that the DAS cable is oriented in the `x` direction.
        # The axial strain along the cable can be deduced as the normal
        # strain, `εₓₓ`. Therefore, we use a rotated coordinate system
        # in which the cable coincides with the `x` axis.
        if isinstance(cable, SurfaceDASCable):
            rotmat = cartesian_rotmat(np.deg2rad(cable.azimuth), 'z')
        else:
            # Borehole DAS cable
            rotmat = cartesian_rotmat(-np.pi / 2.0, 'y')

        mt_symmat = np.asarray(source.pyrocko_moment_tensor().m())
        mt_symmat_rotated = rotmat.T @ mt_symmat @ rotmat

        dircos_vecs_rotated = rotmat.T @ dircos_vecs

        # Radiation-pattern factors
        xset = calc_radiation_patterns_mt(
            mt_symmat_rotated, dircos_vecs_rotated, quantity='strain',
            far=far, intermediate_far=intermediate_far,
            intermediate_near=intermediate_near, near=near)

        # Time-dependent seismic moment, M(t). It should be normalised
        # to one, otherwise the source magnitude becomes meaningless.
        ramp_stf = SmoothRampSTF(duration=stf_duration, anchor=-1.0)
        _, D = ramp_stf.discretize_t(self.deltat, 0.0, scale=False)
        assert D.max() == 1.0, 'Seismic moment STF should be normalized to 1'

        # Seismic moment-rate, dM(t)/dt
        gaus_stf = GaussianSTF(duration=stf_duration, anchor=-1.0)
        _, Ddot = gaus_stf.discretize_t(self.deltat, 0.0, scale=False)

        # Time-derivative of the moment rate, d²M(t)/dt²
        zcros_stf = ZeroCrossingSTF(duration=stf_duration, anchor=-1.0)
        _, Dddot = zcros_stf.discretize_t(self.deltat, 0.0, scale=False)

        # P- and S-wave travel times (flattened arrays)
        tp_all = self._calc_ptimes(dists_3d=dists_3d)
        ts_all = self._calc_stimes(dists_3d=dists_3d)

        # Number of time samples (longest waveform)
        ts_max = ts_all.max()
        data_len = time_to_index(ts_max, self.deltat) + D.size

        c = self._recips
        deltat = self.deltat
        n_rec = dists_3d.size

        # ----------

        if far is True:
            # Far-field strain (ε_f ∝ 1/r)

            # Store only `εₓₓ` radiations, array of shape (n_rec, 1)
            B_fp = xset['FP'].values[0][:, np.newaxis]
            B_fs = xset['FS'].values[0][:, np.newaxis]

            T_fp = np.zeros((n_rec, data_len), dtype=np.float64)
            T_fs = np.zeros_like(T_fp)
            for i_rec in range(n_rec):
                T_fp[i_rec] = pad_stf(tp_all[i_rec], Dddot, deltat, data_len)
                T_fs[i_rec] = pad_stf(ts_all[i_rec], Dddot, deltat, data_len)

            ε_fp = -c['1/4πρ'] * c['1/α']**4 * c['1/r'] * B_fp * T_fp
            ε_fs = +c['1/4πρ'] * c['1/β']**4 * c['1/r'] * B_fs * T_fs

            ε_f = ε_fp + ε_fs
        else:
            ε_f = np.zeros((n_rec, data_len), dtype=np.float64)

        if intermediate_far is True:
            # Intermediate-far field strain (ε_if ∝ 1/r²)

            B_ifp = xset['IFP'].values[0][:, np.newaxis]
            B_ifs = xset['IFS'].values[0][:, np.newaxis]

            T_ifp = np.zeros((n_rec, data_len), dtype=np.float64)
            T_ifs = np.zeros_like(T_ifp)
            for i_rec in range(n_rec):
                T_ifp[i_rec] = pad_stf(tp_all[i_rec], Ddot, deltat, data_len)
                T_ifs[i_rec] = pad_stf(ts_all[i_rec], Ddot, deltat, data_len)

            ε_ifp = +c['1/4πρ'] * c['1/α']**3 * c['1/r2'] * B_ifp * T_ifp
            ε_ifs = -c['1/4πρ'] * c['1/β']**3 * c['1/r2'] * B_ifs * T_ifs

            ε_if = ε_ifp + ε_ifs
        else:
            ε_if = np.zeros((n_rec, data_len), dtype=np.float64)

        if intermediate_near is True:
            # Intermediate-near field strain (ε_in ∝ 1/r³)

            B_inp = xset['INP'].values[0][:, np.newaxis]
            B_ins = xset['INS'].values[0][:, np.newaxis]

            T_inp = np.zeros((n_rec, data_len), dtype=np.float64)
            T_ins = np.zeros_like(T_inp)
            for i_rec in range(n_rec):
                T_inp[i_rec] = pad_stf(tp_all[i_rec], D, deltat, data_len)
                T_ins[i_rec] = pad_stf(ts_all[i_rec], D, deltat, data_len)

            ε_inp = +c['1/4πρ'] * c['1/α']**2 * c['1/r3'] * B_inp * T_inp
            ε_ins = -c['1/4πρ'] * c['1/β']**2 * c['1/r3'] * B_ins * T_ins

            ε_in = ε_inp + ε_ins
        else:
            ε_in = np.zeros((n_rec, data_len), dtype=np.float64)

        if near is True:
            # Near-field strain (ε_n ∝ 1/r⁵)

            B_n = xset['N'].values[0][:, np.newaxis]

            T_n = np.zeros((n_rec, data_len), dtype=np.float64)
            for i_rec in range(n_rec):
                T_n[i_rec] = convolve_stf(
                    tp_all[i_rec], ts_all[i_rec], D, deltat, data_len)

            ε_n = +c['1/4πρ'] * c['1/r5'] * B_n * T_n
        else:
            ε_n = np.zeros((n_rec, data_len), dtype=np.float64)

        # ----------

        # Dimension and coordinate names when saving strains into `xr.DataSet`
        dims = ['i_reciever', 'time']
        coords = {'time': np.arange(data_len) * self.deltat}

        if cable.channel_spacing % cable.grid_spacing == 0.0:
            # Apply moving average

            # Reshape `εₓₓ` arrays (n_rec, data_len)->(1, n_rec, data_len)
            # Stack all along axis=0, x_in.shape == (4, 1, n_rec, data_len)
            x_in = np.stack(
                [e[np.newaxis, :] for e in (ε_f, ε_if, ε_in, ε_n)],
                axis=0)

            # Kernel height is number of grid *points* per GL
            kH = int(round(cable.gauge_len / cable.grid_spacing)) + 1

            # Stride height is number of grid *intervals* per stride
            sH = int(round(cable.channel_spacing / cable.grid_spacing))

            x_out = avg_pool2d(torch.from_numpy(x_in), (kH, 1), stride=(sH, 1))
            x_out = x_out.numpy()

            data_vars = {
                'F': (dims, x_out[0].squeeze(axis=0)),
                'IF': (dims, x_out[1].squeeze(axis=0)),
                'IN': (dims, x_out[2].squeeze(axis=0)),
                'N': (dims, x_out[3].squeeze(axis=0)),
                'total': (dims, x_out.sum(axis=0).squeeze(axis=0))}
        else:
            # Average point strains over separate GLs
            # n_rec = n_channels * n_grids_per_gl
            x_in = [ε_f, ε_if, ε_in, ε_n]
            x_out = np.mean(
                np.stack(np.split(x_in, cable.n_channels, axis=1), axis=1),
                axis=2)

            data_vars = {
                'F': (dims, x_out[0]),
                'IF': (dims, x_out[1]),
                'IN': (dims, x_out[2]),
                'N': (dims, x_out[3]),
                'total': (dims, x_out.sum(axis=0))}

        return xr.Dataset(data_vars=data_vars, coords=coords)


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
        a = [rec.effective_latlon + (rec.depth,) for rec in receivers]
        rlats, rlons, rdepths = zip(*a)
        dists_3d, dircos_vecs = _get_relative_data(
            *event.effective_latlon, event.depth, rlats, rlons, rdepths)

        # Cache reciprocals of powers of 3-D distances.
        self._cache_dist_recips(dists_3d=dists_3d, quantity='displacement')

        # Radiation-pattern factors
        mt_symmat = np.asarray(source.pyrocko_moment_tensor().m())
        xset = calc_radiation_patterns_mt(
            mt_symmat, dircos_vecs, quantity='displacement',
            far=far, intermediate=intermediate, near=near)

        # Time-dependent seismic moment, M(t). It should be normalised
        # to one, otherwise the source magnitude becomes meaningless.
        ramp_stf = SmoothRampSTF(duration=stf_duration, anchor=-1.0)
        _, D = ramp_stf.discretize_t(self.deltat, 0.0, scale=False)
        assert D.max() == 1.0, 'Seismic moment STF should be normalized to 1'

        # Seismic moment-rate, dM(t)/dt
        gaus_stf = GaussianSTF(duration=stf_duration, anchor=-1.0)
        _, Ddot = gaus_stf.discretize_t(self.deltat, 0.0, scale=False)

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
            k: (dims, v)
            for k, v in zip(
                ['F', 'I', 'N', 'total'],
                [u_f, u_i, u_n, u_total])}

        return xr.Dataset(data_vars=data_vars, coords=coords)
