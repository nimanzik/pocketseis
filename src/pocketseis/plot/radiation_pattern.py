from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from pocketseis.hifull import radiation_pattern as rp
from pocketseis.plot import fetch_sci_cmap


class UnitSphereFullSurface(object):
    """
    Unit-sphere configuration space.

    Parameters
    ----------
    n_theta : int, optional
        Number of inclination (or polar) angles. The range for
        inclination is [0, $\\pi$]. Default is 61 (step size of 3 degrees).

    n_phi : int, optional
        Number of azimuth (or azimuthal) angles. The azimuth is restricted
        to the interval [0, 2$\\pi$]. Default is 91 (step size of 4 degrees).

    Notes
    -----
    This class supports meshgrid with only ``matrix indexing``. Therefore,
    the cunstructed space grid points (2-D arrays) are of shape
    ``(n_theta, n_phi)``.

    References
    ----------
    .. [1] Pujol, J. (2003). Elastic wave propagation and generation in
       seismology. Cambridge University Press.
    """

    def __init__(self, n_theta=61, n_phi=91):
        # theta: inclination angle, phi: azimuth angle
        self.n_theta = n_theta
        self.n_phi = n_phi

        thetas = np.linspace(0.0, np.pi, n_theta, dtype=np.float64)
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, dtype=np.float64)
        vthetas, vphis = np.meshgrid(thetas, phis, indexing='ij')

        st = np.sin(vthetas)
        ct = np.cos(vthetas)
        sp = np.sin(vphis)
        cp = np.cos(vphis)

        # Cartesian points (vectorized). Create directional cosines. Uses
        # *inclination* angle, not elevation! See eq. 9.9.16 of Pujol (2003)
        self.vx = st * cp
        self.vy = st * sp
        self.vz = ct

        # Spherical unit vectors. Arrays of shape (n_theta * n_phi, 3)
        self.gamma_uv = np.stack(
            [self.vx.flatten(), self.vy.flatten(), self.vz.flatten()], axis=1)

        self.theta_uv = np.stack(
            [(ct * cp).flatten(), (ct * sp).flatten(), -st.flatten()], axis=1)

        self.phi_uv = np.stack(
            [-sp.flatten(), cp.flatten(), np.zeros(cp.size)], axis=1)

    @property
    def surface_grid_shape(self):
        # 2D-grid shape (depends on meshgrid indexing)
        return (self.n_theta, self.n_phi)


def _check_isaxes3d(ax):
    if not isinstance(ax, Axes3D):
        raise ValueError("requires axes object with projection='3d'")


def _draw_xyz_axes(ax3d, extent=1.5, **kwargs):
    """
    Plot Cartesian coordinate system in the given axes object as
    three perpendicular arrows.

    Parameters
    ----------
    ax3d : :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
        Matplotlib ``Axes3D`` object to plot in with
        ``projection='3d'``.
    extent : float
        A non-zero value specifying the axes limits [-extent, extent].

    Returns
    -------
    None
    """
    _check_isaxes3d(ax3d)
    minmax = np.array([-extent, extent])
    zeros2 = np.array([0.0, 0.0])
    color = kwargs.get('color') or kwargs.get('c') or 'k'

    f = 0.8   # to draw z-axis a bit shorter
    ax3d.plot(minmax, zeros2, zeros2, color)
    ax3d.plot(zeros2, minmax, zeros2, color)
    ax3d.plot(zeros2, zeros2, minmax * f, color)

    text_kwargs = dict(backgroundcolor='none', alpha=0.5)
    ax3d.text(extent, 0, 0, r'$x$', ha='left', va='top', **text_kwargs)
    ax3d.text(0, extent, 0, r'$y$', ha='right', va='top', **text_kwargs)
    ax3d.text(0, 0, extent * f, r'$z$', ha='left', va='bottom', **text_kwargs)


def _set_aspect_equal(ax3d):
    """
    Set the aspect of the axis scaling to 1, i.e. same scaling for x, y
    and z. Axes3D currently does not support the aspect argument 'equal'.

    Returns
    -------
    None
    """
    _check_isaxes3d(ax3d)
    x_range = ax3d.xy_dataLim.width
    y_range = ax3d.xy_dataLim.height
    z_range = ax3d.zz_dataLim.width
    ax3d.set_box_aspect((x_range, y_range, z_range * 0.925))   # Bug in mpl?


def plot_rp_dueto_mt(
        ax3d, mt_symmat, wave_type, wave_field='F', quantity='displacement',
        direction=None, qsphere=None, view_angles=(25, 45),
        draw_cartesian_axes=True):
    """
    Plots the radiation patterns for P and S waves due to an arbitrary
    moment-tensor source.

    Parameters
    ----------
    mt_symmat : ndarray of shape (3, 3)
        Seismic moment tensor as a plain *symmetric* matrix.

    wave_type : {'P', 'S'}
        Seismic wave type.

    wave_field : str, optional
        Displacement or strain wave-field component.
        - If `quantity='displacement'`, choices are {'N', 'I', 'F'}
        - If `quantity='strain'`, choices are {'N', 'NI', 'IF', 'F'}
        (default is 'F').

    quantity : {'displacement', 'strain'}, optional
        Measurement quantity type (default is 'displacement').

    direction : {'radial', 'polar', 'azimuthal'}, optional
        Spherical direction. The radiation pattern is decomposed along
        the directional unit vectors in spherical coordinates, that are:
        - `'radial'`: in source-receiver direction
        - `'polar'`: tangent to the great circle
        - `'azimuthal'`: tangent to the small circle parallel to `x-y` plane.

    qsphere : :py:class:`.UnitSphereFullSurface` object, optional
        Spherical configuration space with points, unit vectors etc as
        its attributes.

    view_angles : tuple or list of 2 float, optional
        View angles `(elevation, azimuth)`. Units: [deg].
        Set the elevation and azimuth of the axes in degrees (not
        radians). `elev` is the angle above (positive) or below
        (negative) the horizontal plane. `azim` is a polar angle in the
        horizontal plane, with positive angles indicating
        counter-clockwise rotation of the view-point (default is
        `(elev, azim)=(15, 35)`.
    """
    if direction is not None and \
            direction not in (valid_dirs := ('radial', 'azimuthal', 'polar')):
        raise ValueError(f"valid values for 'direction' are: '{valid_dirs}'")

    if qsphere is None:
        qsphere = UnitSphereFullSurface()

    if (wave_type := wave_type.upper()) not in (valid_waves := ('P', 'S')):
        raise ValueError(f"valid values for 'wave_type' are: {valid_waves}")

    key = wave_field if wave_field == 'N' else wave_field + wave_type

    if quantity == 'displacement':
        F = rp.disp_from_mt(mt_symmat, qsphere.gamma_uv)[key].values
    else:
        F = rp.normal_strain_from_mt(mt_symmat, qsphere.gamma_uv)[key].values

    # Project radiation pattern factors
    if direction is None:
        if wave_type == 'P':
            # P radiation pattern in radial direction
            direction = 'radial'
            R = np.sum(F * qsphere.gamma_uv, axis=1)
        else:
            # S radiation pattern in absolute value (eq. 9.12.5 of Pujol, 2003)
            R_sv = np.sum(F * qsphere.theta_uv, axis=1)
            R_sh = np.sum(F * qsphere.phi_uv, axis=1)
            R = np.sqrt(R_sv**2 + R_sh**2)
    else:
        # Decompose along given direction (eqs. 9.13.12-14 of Pujol, 2003)
        if direction == 'radial':
            unit_vecs = qsphere.gamma_uv
        elif direction == 'polar':
            unit_vecs = qsphere.theta_uv
        elif direction == 'azimuthal':
            unit_vecs = qsphere.phi_uv

        R = np.sum(F * unit_vecs, axis=1)

    # Reshape to match the meshgrid
    R = R.reshape(qsphere.surface_grid_shape)

    def maxabs_scale(a):
        return a / np.max(np.abs(a))

    # Negative shear motion (rotation) is meaningless for illustration purpose
    mags = maxabs_scale(np.sqrt(R**2))

    if wave_type == 'P':
        cmap = fetch_sci_cmap('berlin')
        norm = Normalize(-1, 1)
    else:
        if direction is None:
            # S radiation pattern in absolute values
            cmap = fetch_sci_cmap('bamako')
            norm = Normalize(0, 1)
        else:
            cmap = fetch_sci_cmap('lisbon')
            norm = Normalize(-1, 1)

    # Indivusual facecolors. Fourth dimension -> rgba
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    facecolors = [mappable.to_rgba(a) for a in maxabs_scale(R).flatten()]
    facecolors = np.array(facecolors).reshape(*qsphere.surface_grid_shape, 4)

    vx = mags * qsphere.vx
    vy = mags * qsphere.vy
    vz = mags * qsphere.vz
    ax3d.plot_surface(
        vx, vy, vz, facecolors=facecolors, alpha=0.9, linewidth=0,
        edgecolors='k', rstride=1, cstride=1)

    elev, azim = view_angles
    ax3d.view_init(elev=elev, azim=azim)
    _set_aspect_equal(ax3d)

    if draw_cartesian_axes is True:
        ax3d.axis('off')
        _draw_xyz_axes(ax3d)
    else:
        ax3d.set(xlabel='x', ylabel='y', zlabel='z')


__all__ = ['UnitSphereFullSurface', 'plot_rp_dueto_mt']
