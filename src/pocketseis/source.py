"""
Module containing customized seismic sources, source-time functions
(seismic moment and moment-rate functions) that can be used by the
:py:class:`pyrocko.gf.seismosizer.Engine` module to calculate synthetic
seismograms.
"""

import numpy as np
from scipy.interpolate import interp1d

from pyrocko import gf, moment_tensor as pmt
from pyrocko.guts import Float, StringChoice, Int

from pocketseis import rotation, mtensor
from pocketseis.hifull import HIFullMaterial


# ----------------------------------------
# MTQTSource:: map ``beta`` parameter to ``u`` variable
# using eq. 24a,Tape & Tape (2015)

_beta = np.linspace(0.0, np.pi, 5000)
_u = 0.75 * _beta - 0.5 * np.sin(2.0 * _beta) + 0.0625 * np.sin(4.0 * _beta)
f_interp = interp1d(_u, _beta)
# ----------------------------------------


class BaseSTF(gf.STF):
    """
    Base class for STFs.

    Parameters
    ----------
    duration : float
        Source-time function duration in [s].
    anchor : {-1.0, 0.0, 1.0}, optional
        Anchor point with respect to source origin-time (default is -1.0).
        -1.0: left -> source duration [0, T] ~ hypocenter time,
         0.0: center -> source duration [-T/2, T/2] ~ centroid time,
        +1.0: right -> source duration [-T, 0] ~ rupture end time.
    """
    duration = Float.T(
        help='Source time function duration in [s] (also called rise '
             'time). It agrees approximately with the rupture duration.')
    anchor = Float.T(
        default=-1.0,
        help='anchor point with respect to source origin-time: '
             '-1.0: left -> source duration [0, T] ~ hypocenter time, '
             ' 0.0: center -> source duration [-T/2, T/2] ~ centroid time, '
             '+1.0: right -> source duration [-T, 0] ~ rupture end time.')

    def tminmax_stf(self, tref):
        """
        Returns time of first and last samples of the STF.
        """
        tmin_stf = tref - self.duration * (self.anchor + 1.0) * 0.5
        tmax_stf = tref + self.duration * (1.0 - self.anchor) * 0.5
        return tmin_stf, tmax_stf

    def base_key(self):
        """
        Returns STF name and attribute values.
        """
        return (type(self).__name__, self.duration, self.anchor)


class SmoothRampSTF(BaseSTF):
    """
    Smoothly increasing ramp (from zero to maximum amplitude)
    source-time function for near-field displacement.
    STF model is based on analytical seismic moment function, M(t),
    according to either Bruestle & Mueller (1983) or Ohtsu (1995).

    References
    ----------
    .. [1] Br{\"u}stle, W., & M{\"u}ller, G. (1983). Moment and duration of
       shallow earthquakes from Love-wave modelling for regional distances.
       Physics of the Earth and Planetary Interiors, 32(4), 312-324.

    .. [2] Ohtsu, M. (1995). Acoustic emission theory for moment tensor
       analysis. Research in Nondestructive Evaluation, 6(3), 169-184.
    """

    model = StringChoice.T(
        choices=['BrMu83', 'Ohtsu95'],
        default='BrMu83',
        help="Source-time function model. Choices are {'BrMu83', 'Ohtsu95'}"
             "For more details on analytical relations, see "
             "Bruestle & Mueller (1983), and Ohtsu (1995).")

    def discretize_t(self, deltat, tref):
        tmin_stf, tmax_stf = self.tminmax_stf(tref)
        tmin = round(tmin_stf / deltat) * deltat
        tmax = round(tmax_stf / deltat) * deltat
        nt = int(round((tmax - tmin) / deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_dummy = np.linspace(tmin - 0.5 * deltat, tmax + 0.5 * deltat, nt)
            t_edges = np.maximum(tmin_stf, np.minimum(tmax_stf, t_dummy))
            omega_t = (t_edges - tmin_stf) * np.pi / self.duration

            if self.model == 'BrMu83':
                # Bruestle & Mueller (1983) model
                amplitudes = (
                    1. - np.cos(omega_t) + (np.cos(3. * omega_t) - 1.) / 9.)
            elif self.model == 'Ohtsu95':
                # Ohtsu (1995) model
                amplitudes = (self.duration / (32.0 * np.pi)) * \
                    (12.0 * omega_t
                     - 8.0 * np.sin(2.0 * omega_t)
                     + np.sin(4.0 * omega_t))

            # Normalized M(t) -> its maximum amplitude == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output
            # in post-processing step is not multiplied by deltat)
            amplitudes *= (deltat / np.max(amplitudes))
        else:
            amplitudes = np.ones(1)

        return times, amplitudes


class GaussianSTF(BaseSTF):
    """
    Gaussian-shaped type source-time function for far-field displacement
    or near-field velocity.
    STF model is based on analytical moment-rate function, dM(t)/dt,
    according to either Bruestle & Mueller (1983) or Ohtsu (1995).

    References
    ----------
    .. [1] Br{\"u}stle, W., & M{\"u}ller, G. (1983). Moment and duration of
       shallow earthquakes from Love-wave modelling for regional distances.
       Physics of the Earth and Planetary Interiors, 32(4), 312-324.

    .. [2] Ohtsu, M. (1995). Acoustic emission theory for moment tensor
       analysis. Research in Nondestructive Evaluation, 6(3), 169-184.
    """

    model = StringChoice.T(
        choices=['BrMu83', 'Ohtsu95'],
        default='BrMu83',
        help="Source-time function model. Choices are {'BrMu83', 'Ohtsu95'}"
             "For more details on analytical relations, see "
             "Bruestle & Mueller (1983), and Ohtsu (1995).")

    def discretize_t(self, deltat, tref):
        tmin_stf, tmax_stf = self.tminmax_stf(tref)
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        nt = int(round((tmax-tmin)/deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_dummy = np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt+1)
            t_edges = np.maximum(tmin_stf, np.minimum(tmax_stf, t_dummy))
            omega_t = (t_edges-tmin_stf) * np.pi / self.duration

            if self.model == 'BrMu83':
                # ## Bruestle & Mueller (1983) model
                fint = 1. - np.cos(omega_t) + (np.cos(3.*omega_t)-1.)/9.
                # Numerical differentiation
                amplitudes = fint[1:] - fint[:-1]
            elif self.model == 'Ohtsu95':
                # ## Ohtsu (1995) model
                fint = (self.duration/(32.*np.pi)) * (
                    12.*omega_t - 8.*np.sin(2.*omega_t) + np.sin(4.*omega_t))
                # Numerical differentiation
                amplitudes = fint[1:] - fint[:-1]

            # Normalized dM(t)/dt -> its numerical integration == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output in
            # post-processing step is not multiplied by deltat)
            amplitudes /= np.sum(amplitudes)
        else:
            amplitudes = np.ones(1)

        return times, amplitudes


class ZeroCrossingSTF(BaseSTF):
    """
    First derivative of a Gaussian-shaped source-time function for
    far-filed velocity.
    STF model is based on the time derivative of analytical moment-rate
    function according to either Bruestle & Mueller (1983) or Ohtsu (1995).

    References
    ----------
    .. [1] Br{\"u}stle, W., & M{\"u}ller, G. (1983). Moment and duration of
       shallow earthquakes from Love-wave modelling for regional distances.
       Physics of the Earth and Planetary Interiors, 32(4), 312-324.

    .. [2] Ohtsu, M. (1995). Acoustic emission theory for moment tensor
       analysis. Research in Nondestructive Evaluation, 6(3), 169-184.
    """

    model = StringChoice.T(
        choices=['BrMu83', 'Ohtsu95'],
        default='BrMu83',
        help="Source-time function model. Choices are {'BrMu83', 'Ohtsu95'}"
             "For more details on analytical relations, see "
             "Bruestle & Mueller (1983), and Ohtsu (1995).")

    def discretize_t(self, deltat, tref):
        tmin_stf, tmax_stf = self.tminmax_stf(tref)
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        nt = int(round((tmax-tmin)/deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_dummy = np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt+1)
            t_edges = np.maximum(tmin_stf, np.minimum(tmax_stf, t_dummy))
            omega = np.pi / self.duration
            omega_t = (t_edges-tmin_stf) * omega

            if self.model == 'BrMu83':
                # ## Bruestle & Mueller (1983) model
                fint = (omega/3.) * (3.*np.sin(omega_t) - np.sin(3.*omega_t))
            elif self.model == 'Ohtsu95':
                # ## Ohtsu (1995) model
                fint = (3. - 4.*np.cos(2.*omega_t) + np.cos(4.*omega_t)) / 8.

            # Normalized dM(t)/dt -> its numerical integration == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output in
            # post-processing step is not multiplied by deltat)
            fint /= np.sum(fint)

            # Numerical differentiation
            amplitudes = (fint[1:] - fint[:-1]) / deltat
        else:
            amplitudes = np.ones(1)

        return times, amplitudes


class StepResponseSTF(BaseSTF):
    """
    Unit-step response solution of a critically damped 2nd order mechanical
    system (with static friction and a dynamic friction proportional to slip
    velocity) for near-field displacement source-time function (dislocation
    history).

    References
    ----------
    .. [1] Harkrider, D. G. (1976). Potentials and displacements for two
       theoretical seismic sources. Geophysical Journal International,
       47(1), 97-133.
    """
    damping_factor = Int.T(
        help="Parameter with the dimension of inverse time, which controls "
             "the width (or frequency content) and amplitude of the "
             "source-time function. It a positive integer. The larger this "
             "parameter the narrower the pulse.")

    def discretize_t(self, deltat, tref):
        tmin_stf, tmax_stf = self.tminmax_stf(tref)
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        nt = int(round((tmax-tmin)/deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_dummy = np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt)
            t_edges = np.maximum(tmin_stf, np.minimum(tmax_stf, t_dummy))
            omega_t = (t_edges - tmin_stf) * self.damping_factor
            amplitudes = 1. - (1. + omega_t) * np.exp(-1.*omega_t)

            # Normalized M(t) -> its maximum amplitude == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output in
            # post-processing step is not multiplied by deltat)
            amplitudes *= (deltat / np.max(amplitudes))
        else:
            amplitudes = np.ones(1)

        return times, amplitudes


class ImpulseResponseSTF(BaseSTF):
    """
    Impulse-response solution of a critically damped 2nd order mechanical
    system (with static friction and a dynamic friction proportional to slip
    velocity) for far-field displacement source-time function (
    dislocation-rate history).

    References
    ----------
    .. [1] Harkrider, D. G. (1976). Potentials and displacements for two
       theoretical seismic sources. Geophysical Journal International,
       47(1), 97-133.
    """
    damping_factor = Int.T(
        help="Parameter with the dimension of inverse time, which controls "
             "the width (or frequency content) and amplitude of the "
             "source-time function. It a positive integer. The larger this "
             "parameter the narrower the pulse.")

    def discretize_t(self, deltat, tref):
        tmin_stf, tmax_stf = self.tminmax_stf(tref)
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        nt = int(round((tmax-tmin)/deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_dummy = np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt)
            t_edges = np.maximum(tmin_stf, np.minimum(tmax_stf, t_dummy))
            omega_t = (t_edges - tmin_stf) * self.damping_factor
            amplitudes = self.damping_factor * omega_t * np.exp(-1.*omega_t)

            # Normalized dM(t)/dt -> its numerical integration == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output in
            # post-processing step is not multiplied by deltat)
            amplitudes /= np.sum(amplitudes)
        else:
            amplitudes = np.ones(1)

        return times, amplitudes


class MTQTSource(gf.SourceWithMagnitude):
    """
    Class for moment-tensor point source following Q-T domain
    parametrization constructed by Tape & Tape [2015] (hereafter TT15).
    """

    u = Float.T(
        default=0.0,
        help='Lunar co-latitude transformed to Q domain.'
             'Interval of definition: [0, 3pi/4]')

    v = Float.T(
        default=0.0,
        help='Lunar longitude transformed to Q domain.'
             'Interval of definition: [-1/3, 1/3]')

    kappa = Float.T(
        default=0.0,
        help='Strike angle in T domain.'
             'Interval of definition: [0, 2pi]')

    sigma = Float.T(
        default=0.0,
        help='Slip angle in T domain.'
             'Interval of definition: [-pi/2, pi/2]')

    h = Float.T(
        default=0.0,
        help='Cosine of dip angle in T domain.'
             'Interval of definition: [0, 1]')

    discretized_source_class = gf.DiscretizedMTSource

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._beta = None
        self._gamma = None
        self._delta = None
        self._theta = None
        self._lune_lambda_triple = None
        self._lune_lambda_matrix = None
        self._rotmat_kappa = None
        self._rotmat_theta = None
        self._rotmat_sigma = None
        self._rotmat_V = None
        self._rotmat_U = None
        self._m9_nwu = None
        self._m6_nwu = None
        self._m6_nwu_astuple = None
        self._m9_ned = None
        self._m6_ned = None
        self._m6_ned_astuple = None

        # TODO
        # if 'm6' in kwargs or 'm6_ned' in kwargs:
        #     pass

    @property
    def beta(self):
        """
        Lunar co-latitude as a function of :py:attr:`.u` (TT15, eq. 24a)
        """
        if self._beta is None:
            self._beta = f_interp(self.u).item()
        return self._beta

    @property
    def gamma(self):
        """
        Lunar longitude as a function of :py:attr:`.v` (TT15, eq. 24b)
        """
        if self._gamma is None:
            self._gamma = (1.0 / 3.0) * np.arcsin(3.0 * self.v)
        return self._gamma

    @property
    def delta(self):
        """
        The coordinate $\\delta$ explicitly measures the departure from
        being deviatoric, and it puts the closest double couple at
        $\\gamma = \\delta = 0$.
        """
        if self._delta is None:
            self._delta = np.pi/2. - self.beta
        return self._delta

    @property
    def theta(self):
        """
        Dip angle as a function of :py:attr:`.h` (TT15, eq. 24c)
        """
        if self._theta is None:
            self._theta = np.arccos(self.h)
        return self._theta

    @property
    def lune_lambda_triple(self):
        """
        Lune eigenvalue triples (TT15, eq. 7)
        """
        if self._lune_lambda_triple is None:
            sqrt2 = np.sqrt(2.)
            sqrt3 = np.sqrt(3.)
            sqrt6 = sqrt2 * sqrt3
            a = np.array([[sqrt3, -1.0, sqrt2],
                          [0.0, 2.0, sqrt2],
                          [-sqrt3, -1.0, sqrt2]], dtype=np.float64) / sqrt6

            b = np.array([np.sin(self.beta)*np.cos(self.gamma),
                          np.sin(self.beta)*np.sin(self.gamma),
                          np.cos(self.beta)])

            self._lune_lambda_triple = np.matmul(a, b)

        return self._lune_lambda_triple

    @property
    def lune_lambda_matrix(self):
        """
        Diagonalized moment tensor (TT15, eq. 4a).
        This matrix describes a beachball moment tensor with eigenvalues
        $\\lambda_1, \\lambda_2, \\lambda_3$ and with corresponding
        eigenvectors in the x, y, z corrdinate direction.
        """
        if self._lune_lambda_matrix is None:
            self._lune_lambda_matrix = np.diag(self.lune_lambda_triple)
        return self._lune_lambda_matrix

    @property
    def rotmat_kappa(self):
        """
        Rotation through angle :py:attr:`.kappa` about the z-axis
        (TT15, eq. 9)
        """
        if self._rotmat_kappa is None:
            self._rotmat_kappa = rotation.rotmat_cartesian(-self.kappa, 'z')
        return self._rotmat_kappa

    @property
    def rotmat_theta(self):
        """
        Rotation through angle :py:meth:`.theta` about the x-axis
        (TT15, eq. 9)
        """
        if self._rotmat_theta is None:
            self._rotmat_theta = rotation.rotmat_cartesian(self.theta, 'x')
        return self._rotmat_theta

    @property
    def rotmat_sigma(self):
        """
        Rotation through angle :py:attr:`.sigma` about the z-axis
        (TT15, eq. 9)
        """
        if self._rotmat_sigma is None:
            self._rotmat_sigma = rotation.rotmat_cartesian(self.sigma, 'z')
        return self._rotmat_sigma

    @property
    def rotmat_V(self):
        """
        Rotation matrix ``V`` defined in TT15, eq. 9.
        """
        if self._rotmat_V is None:
            self._rotmat_V = (
                self.rotmat_kappa @ self.rotmat_theta @ self.rotmat_sigma)
        return self._rotmat_V

    @property
    def rotmat_U(self):
        """
        Rotation matrix ``U`` defined in TT15, eq. 10.
        """
        if self._rotmat_U is None:
            self._rotmat_U = (
                self.rotmat_V @ rotation.rotmat_cartesian(-np.pi / 4.0, 'y'))
        return self._rotmat_U

    @property
    def m9_nwu(self):
        """
        Moment tensor of *unit norm* in north-west-up (north-west-zenith)
        basis convention (the x-y-z Cartesian coordinate system used in
        TT15).
        This matrix describes the same beachball as
        :py:meth:`.lune_lambda_matrix` after being rotated (transformed) by
        the rotation matrix :py:meth:`.self.rotmat_U`.
        """
        if self._m9_nwu is None:
            self._m9_nwu = (self.rotmat_U
                            @ self.lune_lambda_matrix
                            @ np.linalg.inv(self.rotmat_U))
        return self._m9_nwu

    @property
    def m6_nwu(self):
        """
        Non-redundant components from symmetric 3-by-3 moment tensor of
        *unit norm* that is constructed in north-west-up basis.

        Returns
        -------
        1-D NumPy array with entries ordered as (Mnn, Mww, Muu, Mnw,
        Mnu, Mwu).
        """
        if self._m6_nwu is None:
            self._m6_nwu = pmt.to6(self.m9_nwu)
        return self._m6_nwu

    @property
    def m6_nwu_astuple(self):
        """
        Same as :py:meth:`.m6_nwu` but returned as a tuple.
        """
        if self._m6_nwu_astuple is None:
            self._m6_nwu_astuple = tuple(self.m6_nwu.tolist())
        return self._m6_nwu_astuple

    @property
    def m9_ned(self):
        """
        Moment tensor of *unit norm* in north-east-down (north-east-nadir)
        basis convention (the x-y-z Cartesian coordinate system convention
        used by Aki & Richards (2002) and in Pyrocko to construct a moment
        tensor).
        """
        if self._m9_ned is None:
            self._m9_ned = rotation.rotate_mt(self.m9_nwu, 'NWU->NED')
        return self._m9_ned

    @property
    def m6_ned(self):
        """
        Non-redundant components from symmetric 3-by-3 moment tensor of
        *unit norm* that is constructed in north-east-down basis.

        Returns
        -------
        1-D NumpY array with entries ordered as (Mnn, Mee, Mdd, Mne,
        Mnd, Med).
        """
        if self._m6_ned is None:
            self._m6_ned = pmt.to6(self.m9_ned)
        return self._m6_ned

    @property
    def m6_ned_astuple(self):
        """
        Same as :py:meth:`.m6_ned` but returned as a tuple.
        """
        if self._m6_ned_astuple is None:
            self._m6_ned_astuple = tuple(self.m6_ned.tolist())
        return self._m6_ned_astuple

    @property
    def m9(self):
        """
        An alias to :py:meth:`.m9_ned`
        """
        return self.m9_ned

    @property
    def m6(self):
        """
        An alias to :py:meth:`.m6_ned`
        """
        return self.m6_ned

    @property
    def m6_astuple(self):
        """
        An alias to :py:meth:`.m6_ned_astuple`
        """
        return self.m6_ned_astuple

    @property
    def moment(self):
        return mtensor.magnitude_to_moment(self.magnitude)

    @moment.setter
    def moment(self, value):
        self.magnitude = mtensor.moment_to_magnitude(value)

    def pyrocko_moment_tensor(self):
        """
        Returns
        -------
        mmt : :py:class:`pyrocko.moment_tensor.MomentTensor` object
            *Norm-preserving* moment tensor, i.e. the size of the seismic
            event applied.
        """
        # From unit-norm to norm-preserving moment tensor (NED convention)
        m9_denorm = mtensor.denormalize_mt(self.m9_ned, self.moment)
        return pmt.MomentTensor(m=np.asmatrix(m9_denorm))

    def pyrocko_event(self, **kwargs):
        mmt = self.pyrocko_moment_tensor()
        return super().pyrocko_event(moment_tensor=mmt, **kwargs)

    def base_key(self):
        mmt = self.pyrocko_moment_tensor()
        return super().base_key() + tuple(mmt.m6().tolist())

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, self.time)

        # `m6s` is an ndarray of shape (n_samples, 6)
        mmt = self.pyrocko_moment_tensor()
        m6s = mmt.m6()[np.newaxis, :] * amplitudes[:, np.newaxis]

        return gf.DiscretizedMTSource(m6s=m6s,
                                      **self._dparams_base_repeated(times))

    # TODO
    # @classmethod
    # def from_pyrocko_event(cls, event, **kwargs):
    #     d = dict()
    #     mmt = event.moment_tensor
    #     if mmt:
    #         d.update(
    #             magnitude=float(mmt.magnitude),
    #             _m6_ned=tuple(map(float, mmt.m6()/mmt.moment/np.sqrt(2.0))))
    #     d.update(kwargs)
    #     # In order for the following to work, the
    #     # ``from_pyrocko_event`` method in base class
    #     # ``SourceWithMagnitude`` has to be a class method!
    #     return super(MTQTSource, cls).from_pyrocko_event(event, **d)


class TensileSource(gf.Source):
    strike = Float.T(
        default=0.0,
        help='Strike direction measured clockwise from North. Unit: [deg]')
    dip = Float.T(
        default=0.0,
        help='Dip angle measured from the surface to the fault in the '
             'direction of dip. Unit: [deg]')
    rake = Float.T(
        default=0.0,
        help='Rake angle measured on the fault surface from the strike '
             'direction counter-clockwise to the slip vector. Unit: [deg]')
    dislocation_slope = Float.T(
        default=0.0,
        help='Angle α defining the deviation of the dislocation vector '
             'from the fault plane. '
             'α ∈ (0, +90] -> extensive (opening) source, '
             'α = 0 -> shear source, '
             'α ∈ [-90, 0) -> compressive (closing) source.')
    length = Float.T(
        default=1.0,
        help='Length of rectangular source area. Unit [m]')

    width = Float.T(
        default=1.0,
        help='Width of rectangular source area. Unit: [m]')
    dislocation = Float.T(
        default=1.0,
        help='The magnitude of dislocation vector. Unit: [m]')
    material = HIFullMaterial.T(
        default=HIFullMaterial(),
        help='Isotropic elastic material')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fault_area = self.length * self.width
        self.potency = self.dislocation * self.fault_area
        self._potency_tensor = None

    @property
    def potency_tensor(self):
        if self._potency_tensor is None:
            sinϕ, cosϕ = np.sin(self.strike), np.cos(self.strike)
            sinδ, cosδ = np.sin(self.dip), np.cos(self.dip)
            sinλ, cosλ = np.sin(self.rake), np.cos(self.rake)
            sinα, cosα = (
                np.sin(self.dislocation_slope), np.cos(self.dislocation_slope))

            n = np.array([-sinδ * sinϕ, sinδ * cosϕ, -cosδ])[:, np.newaxis]

            v1 = (
                (cosλ * cosϕ + cosδ * sinλ * sinϕ) * cosα
                - (sinδ * sinϕ * sinα))
            v2 = (
                (cosλ * sinϕ - cosδ * sinλ * cosϕ) * cosα
                + (sinδ * cosϕ * sinα))
            v3 = (-sinλ * sinδ * cosα) - (cosδ * sinα)
            v = np.array([v1, v2, v3])[:, np.newaxis]

            D = (self.potency / 2.0) * (np.outer(n, v) + np.outer(v, n))
            self._potency_tensor = D

        return self._potency_tensor

    def pyrocko_moment_tensor(self):
        """Moment tensor in an isotropic medium."""

        D = self.potency_tensor
        Dkk = D.trace()
        λ, μ = self.material.lame()
        m6 = (
            (λ * Dkk) + 2.0 * μ * D[0, 0],
            (λ * Dkk) + 2.0 * μ * D[1, 1],
            (λ * Dkk) + 2.0 * μ * D[2, 2],
            2.0 * μ * D[0, 1],
            2.0 * μ * D[0, 2],
            2.0 * μ * D[1, 2])
        return pmt.MomentTensor.from_values(m6)


__all__ = """
    SmoothRampSTF
    GaussianSTF
    ZeroCrossingSTF
    StepResponseSTF
    ImpulseResponseSTF
    MTQTSource
    TensileSource
""".split()
