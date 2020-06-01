"""
Module for seismic forward modelling in a homogeneous, isotropic full-space.
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.signal import fftconvolve

from pocketseis import util as psutil


class DynamicDispFields(object):
    """
    Base class for dynamic displcement fields.

    Parameters
    ----------
    near, intermediate, far : ndarray
        Displcement amplitudes for near-, intermediate-, and far-filed
        distance ranges. Arrays (for each displcement field term) are of
        shape (n_receivers, 3, data_len), where 3 is the number of receiver
        components.

    deltat : float
        Sampling interval in [s].

    data_len : int
        Number of time samples (it assumes that all seismograms are of
        equal length).
    """
    def __init__(
            self,
            near=0.0,
            intermediate=0.0,
            far=0.0,
            deltat=1.0,
            data_len=1):

        self.near = near
        self.intermediate = intermediate
        self.far = far
        self.deltat = deltat
        self.data_len = int(data_len)
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
            self._times = np.arange(self.data_len) * self.deltat
        return self._times


def _issymmetric(a, tol=1.e-10):
    return np.allclose(a, a.T, atol=tol)


def _column_dstack(array2d):
    n_col = array2d.shape[1]
    return np.moveaxis(np.dstack(np.hsplit(array2d, n_col)), 2, 0)


def dynamic_disp_mt_source(
        vp, vs, rho, rvectors, mt_mat, stf_amps, deltat,
        want_near=True, want_intermediate=True, want_far=True):
    """
    Calculat the elastodynamic body-wave displacement field u(x, t) due
    to a moment-tensor point source excitation in a homogeneous,
    isotropic full-space (infinite/unbounded medium).

    Parameters
    ----------
    vp : float
        P-wave velocity [m/s].

    vs : float
        S-wave velocity [m/s].

    rho : float
        Material density [kg/m^3].

    rvectors : ndarray of shape (3, n_receivers)
        Relative source-receiver position vectors in Cartesian
        coordinate system (i.e. vector whose initial and terminal are
        source and receiver positions, respectively). The indexes of the
        first dimension represent (0, 1, 2)->(x1, x2, x3).

    mt_mat : ndarray of shape (3, 3)
        Seismic moment tensor as a plain 3-by-3 matrix.

    stf_amps : array_like of shape (n_samples,)
        Seismic moment source-time function, M(t).

    deltat : float
        Sampling interval [s].

    Notes
    -----
      * The system time of fisrt sample is zero (i.e. relative to source
      onset time), and the system time of last sample depends on the STF
      length. These values can be easily adapted later with respect to
      P-wave arrival time (``tp``) and S-wave arrivale time plus STF
      duration (ts+T).

    References
    ----------
    .. [1] Pujol, Jose. Elastic wave propagation and generation in
       seismology. Cambridge University Press, 2003.
    """

    if not any((want_near, want_intermediate, want_far)):
        want_far = True

    # #############
    # ## Relative source-receiver vectors

    rvectors = np.asarray(rvectors, dtype=np.float)
    if rvectors.ndim == 1:
        rvectors = rvectors[:, np.newaxis]

    if (rvectors.ndim != 2) or (rvectors.shape[0] != 3):
        raise ValueError(
            "'rvectors' should be an array-like of shape (3, n_receivers)")

    n_receivers = rvectors.shape[1]

    # #############
    # ## Moment tensor and STF

    mt_mat = np.asarray(mt_mat, dtype=np.float)
    if mt_mat.shape != (3, 3):
        raise ValueError("'mt_mat' shoulb be an array of shape (3, 3)")

    stf_amps = np.asarray(stf_amps, dtype=np.float).flatten()

    # #############
    # ## 3-D distances, P & S travel times; all flatttened arrays
    rdistances = np.sqrt(np.sum(rvectors**2, axis=0, keepdims=False))
    ptimes = rdistances / vp
    stimes = rdistances / vs

    if np.any(stimes <= ptimes):
        raise ValueError(
            "unsupported matterial properties; vp={0}, vs={1}".format(vp, vs))

    # #############
    # ## Vectorized terms (Pujol 2003, eqs 9.13.4-8)

    # Unit-vector of direction cosines
    gamma_vectors = rvectors / rdistances

    # Radiation patterns
    c4 = mt_mat @ gamma_vectors
    c3 = c4 if _issymmetric(mt_mat) else mt_mat.T @ gamma_vectors
    c2 = mt_mat.trace() * gamma_vectors
    c1 = np.sum(gamma_vectors*c4, axis=0) * gamma_vectors

    # Reshape from (3, n_receivers) to (n_receivers, 3, 1)
    q4 = _column_dstack(c4)
    q3 = q4 if _issymmetric(mt_mat) else _column_dstack(c3)
    q2 = _column_dstack(c2)
    q1 = _column_dstack(c1)

    # #############
    # ## Multiplicative inverses (MIs)
    minv_4pirho = 1.0 / (4.0*np.pi*rho)
    minv_vp = 1.0 / vp
    minv_vs = 1.0 / vs

    minv_rdistances = 1.0 / rdistances.reshape(-1, 1, 1)
    minv_rdistances2 = minv_rdistances**2
    minv_rdistances4 = minv_rdistances2**2

    # #############
    # ## Phase time indexes (P, S, S-P)
    idx_ptimes = psutil.time2index(ptimes, deltat)
    idx_stimes = psutil.time2index(stimes, deltat)
    idx_sminusp = idx_stimes - idx_ptimes

    # ## Number of time samples (longest waveform)
    ts_max = stimes.max()
    data_len = psutil.time2index(ts_max, deltat) + stf_amps.size

    # #############

    # ## Near-field displacements (NF)
    if want_near is True:

        # Times; receiver-specific convolution with STF;
        # shape of (n_receivers, 1, data_len)
        t_nf = np.zeros((n_receivers, 1, data_len), dtype=np.float)
        for i_rec in range(n_receivers):
            tau_data = psutil.make_time_array(
                ptimes[i_rec], stimes[i_rec], deltat)

            # Repeat end point to prevent boundary effects
            padded_stf = np.pad(stf_amps, (0, tau_data.size), mode='edge')
            convy = fftconvolve(padded_stf, tau_data)[:-tau_data.size] * deltat

            # Pad widths & constat values (left & right, respectively)
            n_prepend = idx_ptimes[i_rec]
            n_append = data_len - (n_prepend + convy.size)
            pad_widths = (n_prepend, n_append)
            constant_values = (0.0, convy[-1])
            t_nf[i_rec] = np.pad(
                convy, pad_widths, mode='constant',
                constant_values=constant_values)

        # Aamplitudes; shape of (n_receivers, 3, 1)
        a_nf = 3 * (5*q1 - q2 - q3 - q4)

        # NF displacements; shape of (n_receivers, 3, data_len)
        u_nf = (minv_4pirho * minv_rdistances4) * a_nf * t_nf

    # ## Intermediate-field displcements (IF)
    if want_intermediate is True:

        # Times; receiver-specific padded seismic moment
        t_ifp = np.zeros((n_receivers, 1, data_len), dtype=np.float)
        t_ifs = np.zeros_like(t_ifp)
        constant_values_if = (0.0, stf_amps[-1])
        for i_rec in range(n_receivers):
            # P-wave times
            n_prepend_p = idx_ptimes[i_rec]
            n_append_p = data_len - (n_prepend_p + stf_amps.size)
            pad_widths_p = (n_prepend_p, n_append_p)
            t_ifp[i_rec] = np.pad(
                stf_amps, pad_widths_p, mode='constant',
                constant_values=constant_values_if)

            # S-wave times
            n_prepend_s = idx_stimes[i_rec]
            n_append_s = data_len - (n_prepend_s + stf_amps.size)
            pad_widths_s = (n_prepend_s, n_append_s)
            t_ifs[i_rec] = np.pad(
                stf_amps, pad_widths_s, mode='constant',
                constant_values=constant_values_if)

        # Amplitudes; shape of (n_receivers, 3, 1)
        a_ifp = 6*q1 - q2 - q3 - q4
        a_ifs = 6*q1 - q2 - q3 - 2*q4

        # IF displacements; shape of (n_receivers, 3, data_len)
        u_ifp = (minv_4pirho * minv_vp**2 * minv_rdistances2) * a_ifp * t_ifp
        u_ifs = -(minv_4pirho * minv_vs**2 * minv_rdistances2) * a_ifs * t_ifs
        u_if = u_ifp + u_ifs

    # ## Far-field displacements (FF)
    if want_far is True:

        # Moment rate
        stf_rate = np.pad(stf_amps[2:]-stf_amps[:-2], 1) / (2.0*deltat)

        # Times; receiver-specific padded moment rate
        t_ffp = np.zeros((n_receivers, 1, data_len), dtype=np.float)
        t_ffs = np.zeros_like(t_ffp)
        constant_values_ff = (0.0, stf_rate[-1])
        for i_rec in range(n_receivers):
            # P-wave times
            n_prepend_p = idx_ptimes[i_rec]
            n_append_p = data_len - (n_prepend_p + stf_rate.size)
            pad_widths_p = (n_prepend_p, n_append_p)
            t_ffp[i_rec] = np.pad(
                stf_rate, pad_widths_p, mode='constant',
                constant_values=constant_values_ff)

            # S-wave times
            n_prepend_s = idx_stimes[i_rec]
            n_append_s = data_len - (n_prepend_s + stf_rate.size)
            pad_widths_s = (n_prepend_s, n_append_s)
            t_ffs[i_rec] = np.pad(
                stf_rate, pad_widths_s, mode='constant',
                constant_values=constant_values_ff)

        # Amplitudes; shape of (n_receivers, 3, 1)
        a_ffp = q1
        a_ffs = q1 - q4

        # FF displacements; shape of (n_receivers, 3, data_len)
        u_ffp = (minv_4pirho * minv_vp**3 * minv_rdistances) * a_ffp * t_ffp
        u_ffs = -(minv_4pirho * minv_vs**3 * minv_rdistances) * a_ffs * t_ffs
        u_ff = u_ffp + u_ffs

    # #############

    return DynamicDispFields(
        near=int(want_near) and u_nf,
        intermediate=int(want_intermediate) and u_if,
        far=int(want_far) and u_ff,
        deltat=deltat,
        data_len=data_len)


class SphereConfigSpace(object):
    """
    Sphere configuration space.

    Parameters
    ----------
    n_theta : int, optional
        Number of inclination (or polar) angles. The range for inclination
        is [0, $\\pi$] radians. Default is 61 (step size of 3 degrees).

    n_phi : int, optional
        Number of azimuth (or azimuthal) angles. The azimuth is restricted
        to the interval [0, 2$\\pi$] radians. Default is 91 (step size of
        4 degrees).

    Notes
    -----
    This class supports meshgrid with only ``matrix indexing``. Therefore,
    the cunstructed space grid points (2-D arrays) are of shape
    ``(n_theta, n_phi)``.
    """
    def __init__(self, n_theta=61, n_phi=91):
        # ## theta: inclination angles, phi: azimuth angles
        self.n_theta = n_theta
        self.n_phi = n_phi

        # ## 2D-grid shapes (depends on meshgrid indexing)
        self.__indexing = 'ij'
        self.sshape = (n_theta, n_phi)

        thetas = np.linspace(0, np.pi, n_theta, dtype=np.float)
        phis = np.linspace(0., 2.*np.pi, n_phi, dtype=np.float)
        vthetas, vphis = np.meshgrid(thetas, phis, indexing=self.__indexing)

        st = np.sin(vthetas)
        ct = np.cos(vthetas)
        sp = np.sin(vphis)
        cp = np.cos(vphis)

        # ## Cartesian points (vectorized)
        # Uses `inclination` angle not elevation!
        self.vx = st * cp
        self.vy = st * sp
        self.vz = ct

        # ## Spherical unit vectors (vectorized)
        self.gamma_uv = np.vstack([
            np.ravel(self.vx),
            np.ravel(self.vy),
            np.ravel(self.vz)])

        self.theta_uv = np.vstack([
            np.ravel(ct*cp),
            np.ravel(ct*sp),
            np.ravel(-st)])

        self.phi_uv = np.vstack([
            np.ravel(-sp),
            np.ravel(cp),
            np.zeros(cp.size)])


class RadiationPattern(SphereConfigSpace):
    """
    Base class for plotting seismic radiation patterns.

    Parameters
    ----------
    values : ndarray
        Radiation pattern values (magnitudes) estimated different spatial
        points in 3-D space.

    field_term : {'near', 'intermediate', 'far'}
        Displacement field term specified by distance from source.

    wave_type : {'P', 'S'}
        Seismic wave type.

    direction : {'radial', 'polar', 'azimuthal'}
        Spherical direction. The radiation pattern is decomposed along
        the directional unit vectors in spherical coordinates, that are:
          * `'radial'` - in source-receiver direction,
          * `'polar'` - tangent to the great circle,
          * `'azimuthal'` - tangent to the small circle parallel to the
            (x1, x2) plane.

    Other parameters
    ----------------
    kwargs : properties of spherical configuration space
        Other keyword arguments passed on to `.SphereConfigSpace`.
        They define number of space points, shape of grid arrays etc.
    """
    def __init__(self, values, field_term, wave_type, direction, **kwargs):
        n_theta = kwargs.get("n_theta", values.shape[0])
        n_phi = kwargs.get("n_phi", values.shape[1])
        super().__init__(n_theta=n_theta, n_phi=n_phi)
        self.values = values
        self.field_term = field_term
        self.wave_type = wave_type
        self.direction = direction
        self._extent = 1.5

        self._mappable_r = ScalarMappable(
            norm=Normalize(-1, 1), cmap=get_cmap('RdBu_r'))

        self._mappable_t = ScalarMappable(
            norm=Normalize(0, 1), cmap=get_cmap('YlGn'))

    def _setup_ax3d(self, ax3d, elev=22.5, azim=47.5):
        """
        Prepare the axes object for final illustration.

        Parameters
        ----------
        ax3d : :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
            Matplotlib ``Axes3D`` object to plot in.

        elev, azim : float
            Set the elevation and azimuth of the axes in degrees (not
            radians).
            `elev` is the angle above (positive) or below
            (negative) the (x1, x2) plane.
            `azim` is a polar angle in the (x1, x2) plane, with positive
            angles indicating counterclockwise rotationof the viewpoint.
            Defalts are 25 and 55 degreed, respectively.
        """
        if not isinstance(ax3d, Axes3D):
            raise ValueError("axes object with projection='3d' is required")

        ax3d.axis('off')
        ax3d.set_aspect('equal')
        ax3d.view_init(elev=elev, azim=azim)
        return ax3d

    def _plot_cartesian_system(self, ax3d, **kwargs):
        """
        Plot Cartesian coordinate system in the given axes object as
        three perpendicular arrows.

        Parameters
        ----------
        ax3d : :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
            Matplotlib ``Axes3D`` object to plot in with
            ``projection='3d'``.

        Returns
        -------
        ax3d : :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
            Axes3D object in which Cartesian system is drawn.
        """
        if not isinstance(ax3d, Axes3D):
            raise ValueError("axes object with projection='3d' is required")

        a = self._extent
        minmax = [-a, a]
        zeros2 = [0., 0.]
        color = kwargs.get('color') or kwargs.get('c') or 'k'

        ax3d.plot(minmax, zeros2, zeros2, color)
        ax3d.plot(zeros2, minmax, zeros2, color)
        ax3d.plot(zeros2, zeros2, minmax, color)

        ax3d.text(a, 0, 0, r'$x_1$', va='top', ha='right')
        ax3d.text(0, a, 0, r'$x_2$', va='top', ha='left')
        ax3d.text(0, 0, a, r'$x_3$', va='bottom', ha='center')
        return ax3d

    def _add_label(self, ax3d):
        """
        Add a 3D text to the given axes consisting three letters to
        indicate:
          * displacement field term (near: N, intermediate: I, far: F),
          * type of propagating wave (P: $\\alpha$, S: $\\beta$),
          * spherical direction: (radial: $\\Gamma$, polar: $\\Theta$,
            azimuthal: $\\Phi$).

        Parameters
        ----------
        ax3d : :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
            Axes3D object to plot in.
        """
        template = r'$\mathcal{{R}}^{{ \mathrm{{{0}}} {{{1}}} {{{2}}} }}$'

        f = self.field_term[0].upper()
        w = {'P': '\\alpha', 'S': '\\beta'}[self.wave_type]
        d = {
            'radial': '\\Gamma',
            'polar': '\\Theta',
            'azimuthal': '\\Phi',
            'tangentional': '\\Psi'}[self.direction]

        a = self._extent
        ax3d.text3D(0, a, a, template.format(f, w, d))
        return ax3d

    def plot_surface(self, ax3d, draw_cartsys=True, add_label=True):
        """
        Plot three-dimensional represetation of seismic radiation
        pattern.

        Parameters
        ----------
        ax3d : :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
            Matplotlib ``Axes3D`` object to plot in.

        draw_cartsys : bool, optional
            Whether to add/plot Cartesian coordinate system to the main
            plot. Default is True.

        add_label : bool, optional
            Whether to  add a legend/label to the main figure indicating
            distance field, wave type, and component projected on.
            Default is True.
        """

        def maxabs_scale(a):
            return a / np.max(np.abs(a))

        # ## Normalize values to [-1, 1] if P, or [0, 1] if S
        if self.direction == 'radial':
            mags = maxabs_scale(self.values)
            mappable = self._mappable_r
        else:
            # Negative shear motion (rotation) is meaningless for
            # illustration purpose
            mags = maxabs_scale(np.sqrt(self.values**2))
            mappable = self._mappable_t

        # ## Indivusual facecolors. 4 == rgba
        facecolors = [mappable.to_rgba(a) for a in mags.ravel()]
        facecolors = np.array(facecolors).reshape(*self.sshape, 4)

        vx = mags * self.vx
        vy = mags * self.vy
        vz = mags * self.vz
        ax3d = self._setup_ax3d(ax3d)
        ax3d.plot_surface(
            vx, vy, vz,
            facecolors=facecolors,
            alpha=0.9,
            linewidth=0,
            edgecolors='k',
            rstride=1,
            cstride=1)

        if draw_cartsys:
            ax3d = self._plot_cartesian_system(ax3d)

        if add_label:
            ax3d = self._add_label(ax3d)

        return ax3d


def radiation_pattern_mt_source(
        mt_symmat,
        qsphere,
        field_term,
        wave_type,
        direction):
    """
    Calculate seismic radiation patterns of body waves generated by a
    general moment tensor point-source in an .

    Parameters
    ----------
    mt_symmat : ndarray of shape (3, 3)
        Seismic moment tensor as a plain **symmetric** matrix.

    qsphere : :py:class:`.SphereConfigSpace` object
        Spherical configuration space with points, unit vectors etc as
        its attributes.

    field_term : {'near', 'intermediate', 'far'}
        Displacement field term.

    wave_type : {'P', 'S'}
        Seismic wave type.

    direction : {'radial', 'polar', 'azimuthal'}
        Spherical direction. The radiation pattern is decomposed
        along the directional unit vectors in spherical coordinates,
        that are:
        `'radial'` - in source-receiver direction,
        `'polar'` - tangent to the great circle,
        `'azimuthal'` - tangent to the small circle parallel to the
        (x1, x2) plane.

    Returns : :py:class:`.RadiationPattern` object
        Seismic radiation pattern derived at given spatial points.

    References
    ----------
    .. [1] Pujol, Jose. Elastic wave propagation and generation in
       seismology. Cambridge University Press, 2003.
    """

    values = np.zeros(int(np.prod(qsphere.sshape)), dtype=np.float)

    # =================================================

    # ## <1> Near-field radiation pattern (NF)
    if field_term == 'near':

        # #############
        # ## <1.1> NF P-wave
        if wave_type == 'P':

            # ## <1.1.a> NF P-wave Radial (Pujol 2003, eq. 9.13.15a)
            if direction == 'radial':
                t1 = 9.0 * np.sum(
                    qsphere.gamma_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)
                t2 = -3.0 * mt_symmat.trace()
                values += (t1 + t2)

            # ## <1.1.b> NF P-wave Polar (Pujol 2003, eq. 9.13.15.b)
            elif direction == 'polar':
                values += -6.0 * np.sum(
                    qsphere.theta_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            # ## <1.1.3> NF P-wave Azimuthal (Pujol 2003, eq. 9.13.15c)
            elif direction == 'azimuthal':
                values += -6.0 * np.sum(
                    qsphere.phi_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

        # #############
        # ## <1.2> NF S-wave
        elif wave_type == 'S':

            # ## <1.2.1> NF S-wave Radial (Pujol 2003, eq. 9.13.15a)
            if direction == 'radial':
                t1 = -9.0 * np.sum(
                    qsphere.gamma_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)
                t2 = 3.0 * mt_symmat.trace()
                values += (t1 + t2)

            # ## <1.2.2> NF S-wave Polar (Pujol 2003, eq. 9.13.15.b)
            elif direction == 'polar':
                values += 6.0 * np.sum(
                    qsphere.theta_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            # ## <1.2.3> NF S-wave Azimuthal (Pujol 2003, eq. 9.13.15c)
            elif direction == 'azimuthal':
                values += 6.0 * np.sum(
                    qsphere.phi_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

    # =================================================

    # ## <2> Intermediate field radiation pattern (IF)
    elif field_term == 'intermediate':

        # #############
        # ## <2.1> IF P-wave
        if wave_type == 'P':

            # ## <2.1.a> IF P-wave Radial (Pujol 2003, eq. 9.13.16a)
            if direction == 'radial':
                t1 = 4.0 * np.sum(
                    qsphere.gamma_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)
                t2 = -mt_symmat.trace()
                values += (t1 + t2)

            # ## <2.1.b> IF P-wave Polar (Pujol 2003, eq. 9.13.16b)
            elif direction == 'polar':
                values += -2.0 * np.sum(
                    qsphere.theta_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            # ## <2.1.c> IF P-wave Azimuthal (Pujol 2003, eq. 9.13.16c)
            elif direction == 'azimuthal':
                values += 3.0 * np.sum(
                    qsphere.phi_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

        # #############
        # ## <2.2> IF S-wave
        elif wave_type == 'S':

            # ## <2.2.a> IF S-wave Radial (Pujol 2003, eq. 9.13.17a)
            if direction == 'radial':
                t1 = -3.0 * np.sum(
                    qsphere.gamma_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)
                t2 = mt_symmat.trace()
                values += (t1 + t2)

            # ## <2.2.b> IF S-wave Polar (Pujol 2003, eq. 9.13.17b)
            elif direction == 'polar':
                values += 3.0 * np.sum(
                    qsphere.theta_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            # ## <2.2.c> IF S-wave Azimuthal (Pujol 2003, eq. 9.13.17c)
            elif direction == 'azimuthal':
                values += 3.0 * np.sum(
                    qsphere.phi_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

    # =================================================

    # ## <3> Far-field radiation pattern (FF)
    elif field_term == 'far':

        # #############
        # ## <3.1> FF P-wave
        if wave_type == 'P':

            # ## <3.1.1> FF P-wave Radial (Pujol 2003, eq. 9.9.18a)
            if direction == 'radial':
                values += np.sum(
                    qsphere.gamma_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            else:
                print(
                    "Far-field P-wave has zero component in "
                    "tangentional directions: '%s'" % direction)
                pass

        # #############
        # ## <3.2>: FF S-wave
        elif wave_type == 'S':

            # ## <3.2.2> FF S-wave Polar (Pujol 2003, eq. 9.9.18b)
            if direction == 'polar':
                values += np.sum(
                    qsphere.theta_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            # ## <3.2.3> FF S-wave Azimuthal (Pujol 2003, eq. 9.9.18c)
            elif direction == 'azimuthal':
                values += np.sum(
                    qsphere.phi_uv * (mt_symmat @ qsphere.gamma_uv),
                    axis=0)

            else:
                print(
                    "Far-field S-wave has zero component in radial "
                    "direction: '%s'" % direction)
                pass

    return RadiationPattern(
        n_theta=qsphere.n_theta,
        n_phi=qsphere.n_phi,
        values=values.reshape(qsphere.sshape),
        field_term=field_term,
        wave_type=wave_type,
        direction=direction)


__all__ = """
    DynamicDispFields
    dynamic_disp_mt_source
    SphereConfigSpace
    RadiationPattern
    radiation_pattern_mt_source
""".split()
