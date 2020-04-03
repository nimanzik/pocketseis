class MomentRateSTFOhtsu95(gf.STF):
    """
    Analytical moment rate function, dM(t)/dt, according to
    Ohtsu (1995).

    Notes
    -----
    .. [1] Ohtsu, M. "Acoustic emission theory for moment tensor analysis."
       Research in Nondestructive Evaluation 6.3 (1995): 169-184.
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
            fint = (self.duration/32./np.pi) * (
                12.*omega_t - 8.*np.sin(2.*omega_t) + np.sin(4.*omega_t))
            # Numerical differentiation
            amplitudes = fint[1:] - fint[:-1]
            # Normalized dM(t)/dt -> numerical integration equals to deltat
            amplitudes /= np.sum(amplitudes)
        else:
            amplitudes = np.ones(1)

        return times, amplitudes

    def base_key(self):
        return (type(self).__name__, self.duration, self.anchor)


class SeismicMomentSTFOhtsu95(gf.STF):
    """
    Analytical seismic moment function, M(t), according to
    Ohtsu (1995).

    Notes
    -----
    .. [1] Ohtsu, M. "Acoustic emission theory for moment tensor analysis."
       Research in Nondestructive Evaluation 6.3 (1995): 169-184.
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
                    np.linspace(tmin-0.5*deltat, tmax+0.5*deltat, nt)))
            omega_t = (t_edges-tmin_stf) * np.pi / self.duration
            amplitudes = (self.duration/32./np.pi) * (
                12.*omega_t - 8.*np.sin(2.*omega_t) + np.sin(4.*omega_t))
            # Normalized M(t) -> maximum amplitude equals to deltat
            #amplitudes *= (deltat/np.max(amplitudes))
            amplitudes /= np.sum(amplitudes)
        else:
            amplitudes = np.ones(1)

        return times, amplitudes

    def base_key(self):
        return (type(self).__name__, self.duration, self.anchor)
