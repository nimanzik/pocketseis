import numpy as np
from scipy.interpolate import interp1d

from pyrocko import gf, moment_tensor as pmt
from pyrocko.gf import SourceWithMagnitude, Source
from pyrocko.guts import Float

from ..moment_tensor.rotation import BasicRotationMatrix


# ## Some constants used more
pi = np.pi
sqrt2 = np.sqrt(2.0)
sqrt3 = np.sqrt(3.0)
sqrt6 = sqrt2 * sqrt3

# ## Lune lambda matrix; constant term in Tape & Tape [2015], eq. (7)
_lune_lambda_cte = np.array([
    [sqrt3, -1.0, sqrt2],
    [0.0, 2.0, sqrt2],
    [-sqrt3, -1.0, sqrt2]], dtype=np.float) * (1.0/sqrt6)

# ## Map beta parameter to 'u' variable using Tape & Tape [2015], eq. (24a)
_beta = np.linspace(0.0, pi, 5000)
_u = 0.75*_beta - 0.5*np.sin(2*_beta) + 0.0625*np.sin(4*_beta)
f_interp = interp1d(_u, _beta)

# Elemental rotations
_R = BasicRotationMatrix()


class MTQTSource(SourceWithMagnitude):
    """
    Class for moment-tensor point source following Q-T domain
    parametrization constructed by Tape & Tape [2015] (TT15).
    """

    u = Float.T(
        default=0.0,
        help='Lune co-latitude transformed to Q domain.'
             'Interval of definition: [0, 3pi/4]')

    v = Float.T(
        default=0.0,
        help='Lune longitude transformed to Q domain.'
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
        SourceWithMagnitude.__init__(self, **kwargs)
        self._beta = None
        self._gamma = None
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
        Lunar co-latitude as a function of ``u`` (TT15, eq. 24a)
        """
        if self._beta is None:
            self._beta = f_interp(self.u)
        return self._beta

    @property
    def gamma(self):
        """
        Lunar longitude as a function of ``v`` (TT15, eq. 24b)
        """
        if self._gamma is None:
            self._gamma = (1.0/3.0) * np.arcsin(3.0*self.v)
        return self._gamma

    @property
    def theta(self):
        """
        Dip angle as a function of ``h`` (TT15, eq. 24c)
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
            sin_beta = np.sin(self.beta)
            vec = np.array([
                sin_beta*np.cos(self.gamma),
                sin_beta*np.sin(self.gamma),
                np.cos(self.beta)])

            self._lune_lambda_triple = _lune_lambda_cte.dot(vec)

        return self._lune_lambda_triple

    @property
    def lune_lambda_matrix(self):
        """
        Diagonalized moment tensor (TT15, eq. 4a).
        """
        if self._lune_lambda_matrix is None:
            self._lune_lambda_matrix = (
                np.diag(self.lune_lambda_triple).astype(np.float))
        return self._lune_lambda_matrix

    @property
    def rotmat_kappa(self):
        """
        Rotation through angle ``kappa`` about the z-axis (TT15, eq. 9)
        """
        if self._rotmat_kappa is None:
            self._rotmat_kappa = _R.about_z(-self.kappa)
        return self._rotmat_kappa

    @property
    def rotmat_theta(self):
        """
        Rotation through angle ``theta`` about the x-axis (TT15, eq. 9)
        """
        if self._rotmat_theta is None:
            self._rotmat_theta = _R.about_x(self.theta)
        return self._rotmat_theta

    @property
    def rotmat_sigma(self):
        """
        Rotation through angle ``sigma`` about the z-axis (TT15, eq. 9)
        """
        if self._rotmat_sigma is None:
            self._rotmat_sigma = _R.about_z(self.sigma)
        return self._rotmat_sigma

    @property
    def rotmat_V(self):
        """
        Rotation matrix V defined in TT15, eq. 9.
        """
        if self._rotmat_V is None:
            self._rotmat_V = np.linalg.multi_dot([
                self.rotmat_kappa,
                self.rotmat_theta,
                self.rotmat_sigma])
        return self._rotmat_V

    @property
    def rotmat_U(self):
        """
        Rotation matrix U defined in TT15, eq. 10.
        """
        if self._rotmat_U is None:
            self._rotmat_U = self.rotmat_V.dot(_R.about_y(-pi/4.0))
        return self._rotmat_U

    @property
    def m9_nwu(self):
        """
        Moment tensor of *unit norm* in north-west-up (north-west-zenith)
        basis convention (this is xyz coordinate system used in
        Tape & Tape (2015)).
        """
        if self._m9_nwu is None:
            self._m9_nwu = np.linalg.multi_dot([
                self.rotmat_U,
                self.lune_lambda_matrix,
                np.linalg.inv(self.rotmat_U)])
        return self._m9_nwu

    @property
    def m6_nwu(self):
        """
        Non-redundant components from symmetric 3-by-3 moment tensor of
        *unit norm* that is constructed in north-west-up basis and
        returned as a 1-D NumPy array with entries ordered as (Mnn, Mww,
        Muu, Mnw, Mnu, Mwu).
        """
        if self._m6_nwu is None:
            self._m6_nwu = pmt.to6(self.m9_nwu)
        return self._m6_nwu

    @property
    def n6_nwu_astuple(self):
        """
        Same as `m6_nwu` but returned as a tuple.
        """
        if self._m6_nwu_astuple is None:
            self._m6_nwu_astuple = tuple(self.m6_nwu.tolist())
        return self._n6_nwu_astuple

    @property
    def m9_ned(self):
        """
        Moment tensor of *unit norm* in north-east-down (north-east-nadir)
        basis convention (this is the coordinate system convention used
        in Pyrocko to construct a moment tensor).
        """
        if self._m9_ned is None:
            rotx_pi = _R.about_x(pi)
            self._m9_ned = np.linalg.multi_dot([
                rotx_pi, self.m9_nwu, rotx_pi.T])
        return self._m9_ned

    @property
    def m6_ned(self):
        """
        Non-redundant components from symmetric 3-by-3 moment tensor of
        *unit norm* that is constructed in north-east-down basis and
        returned as 1-D NumpY array with entries ordered as (Mnn, Mee,
        Mdd, Mne, Mnd, Med).
        """
        if self._m6_ned is None:
            self._m6_ned = pmt.to6(self.m9_ned)
        return self._m6_ned

    @property
    def m6_ned_astuple(self):
        """
        Same as `m6_ned` but returned as a tuple.
        """
        if self._m6_ned_astuple is None:
            self._m6_ned_astuple = tuple(self.m6_ned.tolist())
        return self._m6_ned_astuple

    @property
    def m9(self):
        """
        An alias to ``m9_ned``
        """
        return self.m9_ned

    @property
    def m6(self):
        """
        An alias to ``m6_ned``
        """
        return self.m6_ned

    @property
    def m6_astuple(self):
        """
        An alias to ``m6_ned_astuple``
        """
        return self.m6_ned_astuple

    def pyrocko_moment_tensor(self):
        return pmt.MomentTensor.from_values(self.m6_astuple+(self.magnitude,))

    def pyrocko_event(self, **kwargs):
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=self.magnitude,
            **kwargs)

    def base_key(self):
        mot = self.pyrocko_moment_tensor()
        return SourceWithMagnitude.base_key(self) + tuple(mot.m6().tolist())

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, self.time)

        # m6s is an ndarray of shape Nsamples-by-6
        mot = self.pyrocko_moment_tensor()
        m6s = mot.m6()[np.newaxis, :] * amplitudes[:, np.newaxis]

        return gf.DiscretizedMTSource(
            m6s=m6s, **self._dparams_base_repeated(times))

    # TODO
    # @classmethod
    # def from_pyrocko_event(cls, event, **kwargs):
    #     d = dict()
    #     mot = event.moment_tensor
    #     if mot:
    #         d.update(
    #             magnitude=float(mot.magnitude),
    #             _m6_ned=tuple(map(float, mot.m6()/mot.moment/np.sqrt(2.0))))
    #     d.update(kwargs)
    #     # In order for the following to work, the
    #     # ``from_pyrocko_event`` method in base class
    #     # ``SourceWithMagnitude`` has to be a class method!
    #     return super(MTQTSource, cls).from_pyrocko_event(event, **d)


__all__ = """
    MTQTSource
""".split()


if __name__ == '__main__':

    # ----- Test
    # Following sample calculation of the uniform moment tensor
    # parametrization is taken from Appendix A, Tape & Tape [2015]

    u = 3.0*pi/8.0
    v = -1.0/9.0
    kappa = 4.0*pi/5.0
    sigma = -pi/2.0
    h = 3.0/4.0

    beta_ref = 1.571
    gamma_ref = -0.113
    theta_ref = 0.723
    lune_lambda_triple_ref = 0.001 * np.array([749, -92, -656])

    rotmat_U_ref = 0.001 * np.array([
        [-587, -809, 37],
        [807, -588, -51],
        [63, 0.0, 998]])

    # Moment tensor of unit norm in north-west-up basis
    m9_nwu_ref = 0.001 * np.array([
        [196, -397, -52],
        [-397, 455, 71],
        [-52, 71, -651]])

    # Moment tensor of unit norm in north-east-down basis
    m9_ned_ref = 0.001 * np.array([
        [196, 397, 52],
        [397, 455, 71],
        [52, 71, -651]])

    qtsrc = MTQTSource(u=u, v=v, kappa=kappa, sigma=sigma, h=h)

    rtol, atol = 0.0, 1e-3
    np.testing.assert_allclose(beta_ref, qtsrc.beta, rtol=rtol, atol=atol)
    np.testing.assert_allclose(gamma_ref, qtsrc.gamma, rtol=rtol, atol=atol)
    np.testing.assert_allclose(theta_ref, qtsrc.theta, rtol=rtol, atol=atol)

    np.testing.assert_allclose(
        lune_lambda_triple_ref, qtsrc.lune_lambda_triple, rtol=rtol, atol=atol)

    np.testing.assert_allclose(
        rotmat_U_ref, qtsrc.rotmat_U, rtol=rtol, atol=atol)

    np.testing.assert_allclose(qtsrc.m9_nwu, m9_nwu_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(qtsrc.m9_ned, m9_ned_ref, rtol=rtol, atol=atol)

    print('....... Testing "MTQTSource" has been successful .......')

    # ----- Test
    # Compare synthetic seismograms from MTQTSource with Pyrocko's MTSource
    import matplotlib.pyplot as plt
    from pyrocko.gf import MTSource, Target, LocalEngine

    # Where are my GF stores?
    store_id = 'coseismiq_200Hz_30km_TypeA'
    store_dir = '/mnt/store/nima/gf_stores'

    # Source parameters
    magnitude = 1.5
    elat, elon = 10.0, 10.0
    slat, slon, sdepth = 10.1, 10.1, 2000

    # Create Pyrocko MTSource object
    m6_astuple = tuple(m9_ned_ref.diagonal().tolist()) + \
        (m9_ned_ref[0, 1], m9_ned_ref[0, 2], m9_ned_ref[1, 2])
    mot = pmt.MomentTensor.from_values(m6_astuple + (magnitude,))
    mtsrc = MTSource(m6=mot.m6())

    # Set magnitude for MTQTSource
    setattr(qtsrc, 'magnitude', magnitude)

    # Set other source parameters for MTSource and MTQTSource
    for src in (mtsrc, qtsrc):
        setattr(src, 'lat', slat)
        setattr(src, 'lon', slon)
        setattr(src, 'depth', sdepth)

    # Set targets (i.e. receivers)
    targets = [
        Target(
            quantity='displacement',
            lat=elat,
            lon=elon,
            store_id=store_id,
            codes=('', 'STA', '', channel_code))
        for channel_code in 'ENZ']

    # Now let's make synthetic seismograms
    engine = LocalEngine(store_superdirs=[store_dir])

    qtresp = engine.process(qtsrc, targets)
    qtsyntrs = qtresp.pyrocko_traces()

    mtresp = engine.process(mtsrc, targets)
    mtsyntrs = mtresp.pyrocko_traces()

    # Plot traces
    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()

    for i in range(3):
        ax = axes[i]
        ax.plot(qtsyntrs[i].get_xdata(), qtsyntrs[i].ydata)
        ax.plot(mtsyntrs[i].get_xdata(), mtsyntrs[i].ydata, '--')

    plt.show()
