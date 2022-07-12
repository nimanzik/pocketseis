import math

import numpy as np
from scipy.signal import fftconvolve

from pyrocko.guts import Float, Object
from torch.nn.functional import avg_pool2d
import xarray as xr

from pocketseis.compass import compute_relative_data
from pocketseis.hifull import radiation_pattern as rp
from pocketseis.rotation import cartesian_rotmat
from pocketseis.util import time_to_index, time_range
from .meta import HifullMaterial
from .stf import SmoothRampSTF, GaussianSTF, ZeroCrossingSTF

import torch


guts_prefix = 'pf'


def _convolve_stf(tp, ts, stf_amps, deltat):
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
        than sec). This is equal to a time-length of `r/c + T`, where
        `T` is the duration of source-time function in sec, `r` is 3-D
        distance and `c` is the wave velocity.

    Returns
    -------
    out : ndarray of shape (m+n-1,)
        Convolved source-time function.
    """
    tau = time_range(tp, ts, deltat)

    # Repeat end point to prevent boundary effects
    pady = np.pad(stf_amps, (0, tau.size), mode='edge')
    return fftconvolve(pady, tau)[:-tau.size] * deltat


class BaseHifullScenario(Object):
    """
    Base class for forward modelling scenario for homogeneous,
    isotropic, unbounded (full-space) elastic medium.
    """
    deltat = Float.T(help='Time-sampling interval. Unit: [s]')
    material = HifullMaterial.T(help='Isotropic elastic material')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Reciprocals of the material parameters
        self._recips = {
            '1/4πρ': 1.0 / (4.0 * np.pi * self.material.rho),
            '1/α': 1.0 / self.material.vp,
            '1/β': 1.0 / self.material.vs}

    def _cache_distance_reciprocals(self, dists_3d, exponents):
        """
        Cache reciprocals of powers of 3-D distances.

        Parameters
        ----------
        dists_3d : ndarray of shape (n_receivers,)
            3-D distances of observation points from event. Unit: [m]
        exponents : tuple of ints
            The exponents (> 1).

        Returns
        -------
        None
        """
        self._recips['1/r'] = np.reciprocal(dists_3d)
        for exp in exponents:
            self._recips[f'1/r{exp}'] = np.power(self._recips['1/r'], exp)

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


class DispFromMTHifullScenario(BaseHifullScenario):
    """
    Forward-modelling scenario to calculate displacements in an
    homogeneous, isotropic, unbounded (full-space) elastic medium.
    """

    def process(
            self, source, receivers, stf,
            near=True, intermed=True, far=True):
        """
        Parameters
        ----------
        source :
            Seismic moment-tensor point-source object. It must provide
            a method `pyrocko_events()` that returns
            :py:class:`pyrocko.MomentTensor` object.
        receivers : list of :py:class:`pyrocko.model.Station` objects
            Seismic sensors.
        stf : int, float or ndarray
            Either source-time function duration in [s] or its amplitudes.

        Returns
        -------
        ds : :py:class:`xarray.Dataset` object
            Displacements, measured in m, in (x, y, z)=(North, East, Down)
            Cartesian coordinate system. The Dataset keys are
            `{'N', 'I', 'F', 'total'}`. Each key is mapped to a 3-D
            array, whose shape is `(n_receivers, 3, n_times)` and
            dimesion names are `{'i_receiver', 'axis', 'time'}`.
            The indices of the dimension `'axis'` represent
            (0, 1, 2)->(x, y, z)=(North, East, Down).

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
        dists_3d, cosine_vecs = compute_relative_data(
            *event.effective_latlon, event.depth, rlats, rlons, rdepths)

        # Cache powers of 3-D distances, arrays of shape (n_rec, 1, 1)
        self._cache_distance_reciprocals(
            dists_3d[:, np.newaxis, np.newaxis], exponents=(2, 4))

        # Handle STF
        if isinstance(stf, float) or isinstance(stf, int):
            stf_duration = float(stf)

            # Time-dependent seismic moment, M(t). It should be normalised
            # to one, otherwise the source magnitude becomes meaningless.
            ramp_stf = SmoothRampSTF(duration=stf_duration, anchor=-1.0)
            _, D = ramp_stf.discretize_t(self.deltat, 0.0, scale=False)

            # Seismic moment-rate, dM(t)/dt
            gaus_stf = GaussianSTF(duration=stf_duration, anchor=-1.0)
            _, Ddot = gaus_stf.discretize_t(self.deltat, 0.0, scale=False)
        else:
            D = np.asarray(stf)
            D /= np.max(np.abs(D))   # normalize to 1
            Ddot = np.gradient(D, self.deltat)

        # P- and S-wave travel times and indices (flattened arrays)
        tp_values = self._calc_ptimes(dists_3d=dists_3d)
        tp_indices = time_to_index(tp_values, self.deltat)

        ts_values = self._calc_stimes(dists_3d=dists_3d)
        ts_indices = time_to_index(ts_values, self.deltat)

        # Num. of time samples (longest waveform)
        data_len = ts_indices.max() + D.size

        # ----------
        n_rec = dists_3d.size

        # STF values, arrays of shape (n_rec, 1, data_len)
        T_fp = np.zeros((n_rec, 1, data_len))
        T_fs = np.zeros_like(T_fp)
        T_ip = np.zeros_like(T_fp)
        T_is = np.zeros_like(T_fp)
        T_n = np.zeros_like(T_fp)

        for i_rec in range(n_rec):
            # Begin and end indices
            beg_p = tp_indices[i_rec]
            end_p = beg_p + D.size
            beg_s = ts_indices[i_rec]
            end_s = beg_s + D.size
            # Far field
            T_fp[i_rec, 0, beg_p:end_p] = Ddot
            T_fp[i_rec, 0, end_p:] = Ddot[-1]
            T_fs[i_rec, 0, beg_s:end_s] = Ddot
            T_fs[i_rec, 0, end_s:] = Ddot[-1]
            # Intermediate field
            T_ip[i_rec, 0, beg_p:end_p] = D
            T_ip[i_rec, 0, end_p:] = D[-1]
            T_is[i_rec, 0, beg_s:end_s] = D
            T_is[i_rec, 0, end_s:] = D[-1]
            # Near field
            convy = _convolve_stf(
                tp_values[i_rec], ts_values[i_rec], D, self.deltat)
            stop_n = beg_p + convy.size
            T_n[i_rec, 0, beg_p:stop_n] = convy
            T_n[i_rec, 0, stop_n:] = convy[-1]

        # ----------
        c = self._recips

        # Radiation-patterns
        rp_u = rp.disp_from_mt(
            source.pyrocko_moment_tensor().m(), cosine_vecs,
            far=far, intermed=intermed, near=near)

        # Near-field displacement (u_n ∝ 1/r⁵)
        if near is True:

            A_n = rp_u['N'].values[..., np.newaxis]

            u_n = +c['1/4πρ'] * c['1/r4'] * A_n * T_n
        else:
            u_n = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        # Intermediate-field displacement (u_i ∝ 1/r²)
        if intermed is True:

            A_ip = rp_u['IP'].values[..., np.newaxis]
            A_is = rp_u['IS'].values[..., np.newaxis]

            u_ip = +c['1/4πρ'] * c['1/α']**2 * c['1/r2'] * A_ip * T_ip
            u_is = -c['1/4πρ'] * c['1/β']**2 * c['1/r2'] * A_is * T_is

            u_i = u_ip + u_is
        else:
            u_i = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        # Far-field displacement (u_f ∝ 1/r)
        if far is True:
            # Radiation patterns, arrays of shape (n_rec, 3, 1)
            A_fp = rp_u['FP'].values[..., np.newaxis]
            A_fs = rp_u['FS'].values[..., np.newaxis]

            u_fp = +c['1/4πρ'] * c['1/α']**3 * c['1/r'] * A_fp * T_fp
            u_fs = -c['1/4πρ'] * c['1/β']**3 * c['1/r'] * A_fs * T_fs

            u_f = u_fp + u_fs
        else:
            u_f = np.zeros((n_rec, 3, data_len), dtype=np.float64)

        # Total-field displacement
        u_total = u_n + u_i + u_f

        # ----------
        # Save results into a `xr.DataSet` with following dimensions and
        # coordinates. Each array is of shape (n_rec, 3, data_len)
        dims = ['i_receiver', 'axis', 'time']
        coords = {
            'axis': ['x', 'y', 'z'],
            'time': np.arange(data_len) * self.deltat}
        data_vars = {
            k: (dims, v)
            for k, v in zip(
                ['N', 'I', 'F', 'total'],
                [u_n, u_i, u_f, u_total])}

        return xr.Dataset(data_vars=data_vars, coords=coords)


class StrainFromMTHifullScenario(BaseHifullScenario):
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
            self, source, cable, stf, near=True, near_intermed=True,
            intermed_far=True, far=True):
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
        stf : int, float or ndarray
            Either source-time function duration in [s] or amplitudes.

        Returns
        -------
        ds : :py:class:`xarray.Dataset` object
            Longitudinal strains measured *along the cable axis*.
            The Dataset keys are `{'N', 'NI', 'IF', 'F', 'total'}`.
            Each key is mapped to a 2-D array, whose shape is
            `(n_channels, n_times)` and dimesion names are
            `{'i_receiver', 'time'}`.

        Notes
        -----
          * The system time of fisrt sample is zero (i.e. relative to
          source onset time), and the system time of last sample depends
          on the STF length. These values can be easily adapted later
          with respect to P-wave arrival time (``tp``) and S-wave arrival
          time plus STF duration (``ts+T``).
        """
        event = source.pyrocko_event()
        dists_3d, cosine_vecs = \
            cable.get_event_relative_data(event, level='grid')

        # Cache powers of 3-D distances, arrays of shape (n_rec, 1)
        self._cache_distance_reciprocals(
            dists_3d[:, np.newaxis], exponents=(2, 3, 5))

        # Handle STF
        if isinstance(stf, float) or isinstance(stf, int):
            stf_duration = float(stf)

            # Time-dependent seismic moment, M(t). It should be normalized
            # to 1, otherwise the source magnitude becomes meaningless.
            ramp_stf = SmoothRampSTF(duration=stf_duration, anchor=-1.0)
            _, D = ramp_stf.discretize_t(self.deltat, 0.0, scale=False)

            # Seismic moment-rate, dM(t)/dt
            gaus_stf = GaussianSTF(duration=stf_duration, anchor=-1.0)
            _, Ddot = gaus_stf.discretize_t(self.deltat, 0.0, scale=False)

            # Time-derivative of the moment rate, d²M(t)/dt²
            zcros_stf = ZeroCrossingSTF(duration=stf_duration, anchor=-1.0)
            _, Dddot = zcros_stf.discretize_t(self.deltat, 0.0, scale=False)
        else:
            D = np.asarray(stf)
            D /= np.max(np.abs(D))   # normalize to 1
            Ddot = np.gradient(D, self.deltat)
            Dddot = np.gradient(Ddot, self.deltat)

        # P- and S-wave travel times and indices (flattened arrays)
        tp_values = self._calc_ptimes(dists_3d=dists_3d)
        tp_indices = time_to_index(tp_values, self.deltat)

        ts_values = self._calc_stimes(dists_3d=dists_3d)
        ts_indices = time_to_index(ts_values, self.deltat)

        # Num. of time samples (longest waveform)
        data_len = ts_indices.max() + D.size

        # ----------
        n_rec = dists_3d.size

        # STF values, arrays of shape (n_rec, data_len)
        T_fp = np.zeros((n_rec, data_len))
        T_fs = np.zeros_like(T_fp)
        T_ifp = np.zeros_like(T_fp)
        T_ifs = np.zeros_like(T_fp)
        T_inp = np.zeros_like(T_fp)
        T_ins = np.zeros_like(T_fp)
        T_n = np.zeros_like(T_fp)

        for i_rec in range(n_rec):
            # Begin and end indices
            beg_p = tp_indices[i_rec]
            end_p = beg_p + D.size
            beg_s = ts_indices[i_rec]
            end_s = beg_s + D.size
            # Far field
            T_fp[i_rec, beg_p:end_p] = Dddot
            T_fp[i_rec, end_p:] = Dddot[-1]
            T_fs[i_rec, beg_s:end_s] = Dddot
            T_fs[i_rec, end_s:] = Dddot[-1]
            # Intermediate-far field
            T_ifp[i_rec, beg_p:end_p] = Ddot
            T_ifp[i_rec, end_p:] = Ddot[-1]
            T_ifs[i_rec, beg_s:end_s] = Ddot
            T_ifs[i_rec, end_s:] = Ddot[-1]
            # Intermediate-near field
            T_inp[i_rec, beg_p:end_p] = D
            T_inp[i_rec, end_p:] = D[-1]
            T_ins[i_rec, beg_s:end_s] = D
            T_ins[i_rec, end_s:] = D[-1]
            # Near field
            convy = _convolve_stf(
                tp_values[i_rec], ts_values[i_rec], D, self.deltat)
            end_n = beg_p + convy.size
            T_n[i_rec, beg_p:end_n] = convy
            T_n[i_rec, end_n:] = convy[-1]

        # ----------
        # We assume that the DAS cable is oriented in the `x` direction.
        # The axial strain along the cable can be deduced as the normal
        # strain, `εₓₓ`. Therefore, we use a rotated coordinate system
        # in which the cable coincides with the `x` axis.
        if hasattr(cable, 'azimuth'):
            # Surface DAS cable
            rotmat = cartesian_rotmat(np.deg2rad(cable.azimuth), 'z')
        else:
            # Borehole DAS cable
            rotmat = cartesian_rotmat(-np.pi / 2.0, 'y')

        mt_symmat = np.asarray(source.pyrocko_moment_tensor().m())
        mt_symmat_rotated = rotmat.T @ mt_symmat @ rotmat

        # Originally, X.shape must be (3, n_rec) and X′ = (R.T @ X). Here,
        # Y.shape is (n_rec, 3). So, Y′ = (R.T @ X).T = X.T @ R = Y @ R
        cosine_vecs_rotated = cosine_vecs @ rotmat

        # Radiation-pattern factors
        rp_e = rp.normal_strain_from_mt(
            mt_symmat_rotated, cosine_vecs_rotated, far=far,
            intermed_far=intermed_far, near_intermed=near_intermed, near=near)

        # ----------
        c = self._recips

        # Near-field strain (ε_n ∝ 1/r⁵)
        if near is True:

            B_n = rp_e['N'].values[:, [0]]

            ε_n = +c['1/4πρ'] * c['1/r5'] * B_n * T_n
        else:
            ε_n = np.zeros((n_rec, data_len), dtype=np.float64)

        # Near-to-intermediate field strain (ε_ni ∝ 1/r³)
        if near_intermed is True:

            B_nip = rp_e['NIP'].values[:, [0]]
            B_nis = rp_e['NIS'].values[:, [0]]

            ε_nip = +c['1/4πρ'] * c['1/α']**2 * c['1/r3'] * B_nip * T_inp
            ε_nis = -c['1/4πρ'] * c['1/β']**2 * c['1/r3'] * B_nis * T_ins

            ε_ni = ε_nip + ε_nis
        else:
            ε_ni = np.zeros((n_rec, data_len), dtype=np.float64)

        # Intermediate-to-far field strain (ε_if ∝ 1/r²)
        if intermed_far is True:

            B_ifp = rp_e['IFP'].values[:, [0]]
            B_ifs = rp_e['IFS'].values[:, [0]]

            ε_ifp = +c['1/4πρ'] * c['1/α']**3 * c['1/r2'] * B_ifp * T_ifp
            ε_ifs = -c['1/4πρ'] * c['1/β']**3 * c['1/r2'] * B_ifs * T_ifs

            ε_if = ε_ifp + ε_ifs
        else:
            ε_if = np.zeros((n_rec, data_len), dtype=np.float64)

        # Far-field strain (ε_f ∝ 1/r)
        if far is True:
            # Store only `εₓₓ` radiations, arrays of shape (n_rec, 1)
            B_fp = rp_e['FP'].values[:, [0]]
            B_fs = rp_e['FS'].values[:, [0]]

            ε_fp = -c['1/4πρ'] * c['1/α']**4 * c['1/r'] * B_fp * T_fp
            ε_fs = +c['1/4πρ'] * c['1/β']**4 * c['1/r'] * B_fs * T_fs

            ε_f = ε_fp + ε_fs
        else:
            ε_f = np.zeros((n_rec, data_len), dtype=np.float64)

        # ----------
        # Dimension and coordinate names when saving to `xr.DataSet`
        dims = ['i_receiver', 'time']
        coords = {'time': np.arange(data_len) * self.deltat}

        if math.isclose(
                math.remainder(cable.channel_spacing, cable.grid_spacing), 0.0,
                rel_tol=0.0, abs_tol=1e-9):
            # Apply moving average

            # Reshape `εₓₓ` arrays (n_rec, data_len)->(1, n_rec, data_len)
            # Stack all along axis=0, x_in.shape == (4, 1, n_rec, data_len)
            x_in = np.stack(
                [e[np.newaxis, :] for e in (ε_n, ε_ni, ε_if, ε_f)],
                axis=0)

            # Kernel height is number of grid *points* per GL
            kH = int(round(cable.gauge_len / cable.grid_spacing)) + 1

            # Stride height is number of grid *intervals* per stride
            sH = int(round(cable.channel_spacing / cable.grid_spacing))

            x_out = avg_pool2d(torch.from_numpy(x_in), (kH, 1), stride=(sH, 1))
            x_out = x_out.numpy()

            data_vars = {
                'N': (dims, x_out[0].squeeze(axis=0)),
                'NI': (dims, x_out[1].squeeze(axis=0)),
                'IF': (dims, x_out[2].squeeze(axis=0)),
                'F': (dims, x_out[3].squeeze(axis=0)),
                'total': (dims, x_out.sum(axis=0).squeeze(axis=0))}
        else:
            # Average point strains over separate GLs

            # n_rec = n_channels * n_grids_per_gl
            x_in = np.stack([ε_n, ε_ni, ε_if, ε_f], axis=0)
            x_out = np.mean(
                np.stack(np.split(x_in, cable.n_channels, axis=1), axis=1),
                axis=2)

            data_vars = {
                'N': (dims, x_out[0]),
                'NI': (dims, x_out[1]),
                'IF': (dims, x_out[2]),
                'F': (dims, x_out[3]),
                'total': (dims, x_out.sum(axis=0))}

        return xr.Dataset(data_vars=data_vars, coords=coords)


__all__ = ['DispFromMTHifullScenario', 'StrainFromMTHifullScenario']
