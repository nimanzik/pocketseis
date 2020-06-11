"""
Module containing customized source-time functions (seismic moment and moment
rate functions) that can be used by the `pyrocko.gf.seismosizer.Engine` module
to calculate synthetic seismograms.
"""

import numpy as np

from pyrocko.gf import STF
from pyrocko.guts import Float, StringChoice, Int


class BaseSTF(STF):
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
        tmin_stf = tref - self.duration*(self.anchor+1.0)*0.5
        tmax_stf = tref + self.duration*(1.0-self.anchor)*0.5
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
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        nt = int(round((tmax-tmin)/deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_dummy = np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt)
            t_edges = np.maximum(tmin_stf, np.minimum(tmax_stf, t_dummy))
            omega_t = (t_edges-tmin_stf) * np.pi / self.duration

            if self.model == 'BrMu83':
                # ## Bruestle & Mueller (1983) model
                amplitudes = 1. - np.cos(omega_t) + (np.cos(3.*omega_t)-1.)/9.
            elif self.model == 'Ohtsu95':
                # ## Ohtsu (1995) model
                amplitudes = (self.duration/(32.*np.pi)) * (
                    12.*omega_t - 8.*np.sin(2.*omega_t) + np.sin(4.*omega_t))

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


__all__ = """
    SmoothRampSTF
    GaussianSTF
    ZeroCrossingSTF
    StepResponseSTF
    ImpulseResponseSTF
""".split()
