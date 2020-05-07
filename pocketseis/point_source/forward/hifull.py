import numpy as np
from scipy.signal import fftconvolve

from pocketseis import util as psutil


class DynamicDisplacementField(object):
    def __init__(self, near, intermediate, far, deltat):
        self.near = near
        self.intermediate = intermediate
        self.far = far
        self.deltat = deltat
        self._total = None
        self._times = None

    @property
    def total(self):
        if self._total is None:
            self._total = self.near + self.intermediate + self.far
        return self._total

    @property
    def times(self):
        if self._times is None:
            nt = np.any(self.total) and self.total.shape[1] or 1
            self._times = np.arange(nt) * self.deltat
        return self._times


def dynamic_displacements_from_mt_source(
        vp, vs, rho, x, mt, stf, deltat, want_near=True,
        want_intermediate=True, want_far=True):
    """
    Calculat the elastodynamic displacement field u(x, t) due to a
    moment-tensor point source excitation in a homogeneous, isotropic
    full-space (i.e. infinite medium).

    Parameters
    ----------
    vp : float
        P-wave velocity [m/s].
    vs : float
        S-wave velocity [m/s].
    rho : float
        Density [kg/m^3].
    x : ndarray of shape (3, 1)
        Relative source-receiver position vector in Cartesian coordinate
        system (i.e. vector whose initial and terminal are source and
        receiver positions, respectively).
    mt : ndarray of shape (3, 3)
        Seismic moment tensor.
    stf : array_like
        Seismic moment source-time function.
    deltat : float
        Sampling interval [s]

    TODO
    ----
    The system time of fisrt sample is zero (i.e. relative to event time)
    The system time of last sample depends on the STF length.
    These paraneters should be set relative to tp and ts+STF.
    """

    if not any((want_near, want_intermediate, want_far)):
        return

    x = np.asarray(x, dtype=np.float).flatten()
    if x.size != 3:
        raise ValueError("'x' should be an array-like of size 3")

    mt = np.asarray(mt, dtype=np.float)
    if mt.shape != (3, 3):
        raise ValueError("'mt' shoulb be an array of shape (3, 3)")

    stf = np.asarray(stf, dtype=np.float).flatten()

    # #############

    r = np.sqrt(np.sum(x**2))
    tp = r / vp
    ts = r / vs

    if ts <= tp:
        raise ValueError(
            "unsupported matterial properties; vp={0}, vs={1}".format(vp, vs))

    # ## Unit-vector of direction cosines (column vector)
    gamma = x[:, np.newaxis] / r

    # ## These make the computation faster
    c1 = np.linalg.multi_dot([gamma.T, mt, gamma]).item() * gamma
    c2 = mt.trace() * gamma
    c3 = np.dot(mt.T, gamma)
    c4 = np.dot(mt, gamma)

    for ic in range(1, 5):
        cname = 'c{}'.format(ic)
        c = eval(cname)
        assert c.shape == (3, 1), "'%s' should be of shape (3, 1)" % cname

    # ## Constants used more than once
    irho4pi = 1.0 / (4.0 * np.pi * rho)
    ir = 1.0 / r
    ivp = 1.0 / vp
    ivs = 1.0 / vs

    i_tp = psutil.time2index(tp, deltat)
    i_ts = psutil.time2index(ts, deltat)
    i_ts_tp = i_ts - i_tp

    # #############

    # ## Near-field displacement
    if want_near:
        # Repeat end point to prevent boundary effects
        tau_data = psutil.make_time_array(tp, ts, deltat)
        padded_stf = np.concatenate((stf, np.ones_like(tau_data)*stf[-1]))
        convy = fftconvolve(padded_stf, tau_data)[:-tau_data.size] * deltat
        t_nf = np.concatenate([np.zeros(i_tp), convy])

        a_nf = 3 * (5*c1 - c2 - c3 - c4)
        u_nf = (irho4pi * ir**4) * a_nf * t_nf

    # ## Intermediate-field displcement
    if want_intermediate:
        t_ifp = np.concatenate([np.zeros(i_tp), stf, np.ones(i_ts_tp)*stf[-1]])
        t_ifs = np.concatenate([np.zeros(i_ts), stf])

        a_ifp = 6*c1 - c2 - c3 - c4
        a_ifs = 6*c1 - c2 - c3 - 2*c4

        u_ifp = (irho4pi * ivp**2 * ir**2) * a_ifp * t_ifp
        u_ifs = -(irho4pi * ivs**2 * ir**2) * a_ifs * t_ifs
        u_if = u_ifp + u_ifs

    # ## Far-field displacement
    if want_far:
        stf_rate = np.pad(stf[2:]-stf[:-2], 1) / (2*deltat)
        t_ffp = np.concatenate([np.zeros(i_tp), stf_rate, np.zeros(i_ts_tp)])
        t_ffs = np.concatenate([np.zeros(i_ts), stf_rate])

        a_ffp = c1
        a_ffs = c1 - c4

        u_ffp = (irho4pi * ivp**3 * ir) * a_ffp * t_ffp
        u_ffs = -(irho4pi * ivs**3 * ir) * a_ffs * t_ffs
        u_ff = u_ffp + u_ffs

    # #############

    return DynamicDisplacementField(
        near=int(want_near) and u_nf,
        intermediate=int(want_intermediate) and u_if,
        far=int(want_far) and u_ff,
        deltat=deltat)


__all__ = """
    dynamic_displacements_from_mt_source
""".split()
