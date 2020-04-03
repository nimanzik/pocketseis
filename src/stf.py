import numpy as np

from pyrocko.gf import STF
from pyrocko.guts import Float


class GaussianSTF(STF):
    """
    Gaussian-shaped type source time function for far-field displacement
    or near-field velocity.
    This is based on analytical moment-rate function, dM(t)/dt, proposed
    by Bruestle & Mueller (1983).

    Notes
    -----
    .. [1] Bruestle, W., and G. Mueller. "Moment and duration of shallow
       earthquakes from Love-wave modelling for regional distances."
       Physics of the Earth and Planetary Interiors 32.4 (1983): 312-324.
    """
    duration = Float.T(
        help='Source time function duration in [s] (also called rise '
             'time). It agrees approximately with the rupture duration.')
    anchor = Float.T(
        default=-1.0,
        help='anchor point with respect to source.time: ( '
             '-1.0: left -> source duration [0, T] ~ hypocenter time, '
             ' 0.0: center -> source duration [-T/2, T/2] ~ centroid time, '
             '+1.0: right -> source duration [-T, 0] ~ rupture end time)')

    def discretize_t(self, deltat, tref):
        tmin_stf = tref - self.duration*(self.anchor+1.)*0.5
        tmax_stf = tref + self.duration*(1.-self.anchor)*0.5
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        d = round((tmax-tmin)/deltat) * deltat
        nt = int(d/deltat) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_edges = np.maximum(
                tmin_stf,
                np.minimum(
                    tmax_stf,
                    np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt+1)))
            omega_t = (t_edges-tmin_stf) * np.pi / self.duration
            fint = 1.0 - np.cos(omega_t) + (np.cos(3*omega_t)-1.0)/9.0
            # Numerical differentiation
            amplitudes = fint[1:] - fint[:-1]

            # Normalized dM(t)/dt -> its numerical integration == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output in
            # post-processing step is not multiplied by deltat)
            amplitudes /= np.sum(amplitudes)
        else:
            amplitudes = np.ones(1)

        return times, amplitudes

    def base_key(self):
        return (type(self).__name__, self.duration, self.anchor)


class GaussianDerivativeSTF(STF):
    """
    First derivative of a Gaussian-shaped source-time function for
    far-filed velocity.

    This is based on the time derivative of analytical moment-rate
    function, d(dM(t)/dt)/dt, proposed by Bruestle & Mueller (1983).

    Notes
    -----
    .. [1] Bruestle, W., and G. Mueller. "Moment and duration of shallow
       earthquakes from Love-wave modelling for regional distances."
       Physics of the Earth and Planetary Interiors 32.4 (1983): 312-324.
    """
    duration = Float.T(
        help='Source time function duration in [s] (also called rise '
             'time). It agrees approximately with the rupture duration.')
    anchor = Float.T(
        default=-1.0,
        help='anchor point with respect to source.time: ( '
             '-1.0: left -> source duration [0, T] ~ hypocenter time, '
             ' 0.0: center -> source duration [-T/2, T/2] ~ centroid time, '
             '+1.0: right -> source duration [-T, 0] ~ rupture end time)')

    def discretize_t(self, deltat, tref):
        tmin_stf = tref - self.duration*(self.anchor+1.)*0.5
        tmax_stf = tref + self.duration*(1.-self.anchor)*0.5
        tmin = round(tmin_stf/deltat) * deltat
        tmax = round(tmax_stf/deltat) * deltat
        d = round((tmax-tmin)/deltat) * deltat
        nt = int(d/deltat) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            t_edges = np.maximum(
                tmin_stf,
                np.minimum(
                    tmax_stf,
                    np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt+1)))
            omega = np.pi / self.duration
            omega_t = (t_edges-tmin_stf) * omega
            fint = (omega/3.0) * (3.0*np.sin(omega_t) - np.sin(3.0*omega_t))
            # Normalized dM(t)/dt -> its numerical integration == deltat
            # (in `pyrocko.gf.seismoseizer`, the convolution output in
            # post-processing step is not multiplied by deltat)
            fint /= np.sum(fint)
            # Numerical differentiation
            amplitudes = (fint[1:] - fint[:-1]) / deltat
        else:
            amplitudes = np.ones(1) 

        return times, amplitudes

    def base_key(self):
        return (type(self).__name__, self.duration, self.anchor)
