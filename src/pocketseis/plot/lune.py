from functools import partial

from matplotlib.path import Path as mpath
from matplotlib import transforms
import numpy as np

from cartopy import crs


MM2INCHES = 1 / 25.4
LUNE_PROJECTION = crs.LambertAzimuthalEqualArea()
DATA_CRS = crs.PlateCarree()


def draw_lune(ax, grid=True, fontsize=12):
    """
    Plot fundamental lune and its corresponding annotations.
    """

    assert ax.projection == LUNE_PROJECTION, \
        "Axes projection must be 'LambertAzimuthalEqualArea'"

    # Draw boundary around the lune
    lats = np.concatenate([np.arange(-90, 91), np.arange(90, -91, -1), [-90]])
    lons = np.concatenate([np.tile(30, 181), np.tile(-30, 181), [30]])
    codes = np.hstack([mpath.MOVETO, [mpath.LINETO] * 361, mpath.CLOSEPOLY])

    verts = np.column_stack([lons, lats])
    path = mpath(verts, codes)
    ax.set_boundary(path, transform=DATA_CRS)
    ax.set_extent([-30, 30, -90, 90], crs=DATA_CRS)

    # Add gridlines
    if grid:
        ax.gridlines(
            xlocs=np.arange(-30, 40, 10, dtype=np.int64),
            ylocs=np.arange(-90, 100, 10, dtype=np.int64),
            zorder=0)

    # Annotation tranformars with added offsets
    trans_offset_copy = partial(
        transforms.offset_copy,
        trans=crs.PlateCarree()._as_mpl_transform(ax),
        units='inches', fig=ax.get_figure())

    d = 1.75 * MM2INCHES
    offset_left = trans_offset_copy(x=-d)
    offset_right = trans_offset_copy(x=+d)
    offset_top = trans_offset_copy(y=+d)
    offset_bottom = trans_offset_copy(y=-d)
    offset_topright = trans_offset_copy(y=-d, x=-d)

    text_list = [
        ('+ISO', 0.0, +90.0, offset_top, 'center', 'bottom'),
        ('-ISO', 0.0, -90.0, offset_bottom, 'center', 'top'),
        ('+Crack', -30.0, +60.5038, offset_left, 'right', 'center'),
        ('-Crack', +30.0, -60.5038, offset_right, 'left', 'center'),
        ('+LVD', -30.0, +35.2644, offset_left, 'right', 'center'),
        ('-LVD', +30.0, -35.2644, offset_right, 'left', 'center'),
        ('+CLVD', -30.0, 0.0, offset_left, 'right', 'center'),
        ('-CLVD', +30.0, 0.0, offset_right, 'left', 'center'),
        ('DC', 0.0, 0.0, offset_topright, 'center', 'top')]

    for (txt, x, y, tto, ha, va) in text_list:
        ax.plot(
            x, y, 'o', color='k', transform=DATA_CRS, clip_on=False,
            markersize=fontsize / 3.0)
        ax.text(x, y, txt, transform=tto, clip_on=False, ha=ha, va=va)

    # Source type arc
    x1 = [
        -30.00, -29.20, -28.38, -27.55, -26.71, -25.86, -24.98, -24.10, -23.19,
        -22.27, -21.34, -20.39, -19.42, -18.43, -17.42, -16.40, -15.35, -14.29,
        -13.21, -12.10, -10.98, -09.83, -08.67, -07.48, -06.27, -05.04, -03.79,
        -02.51, -01.22, +00.10, +01.44, +02.79, +04.17, +05.57, +06.99, +08.43,
        +09.89, +11.36, +12.85, +14.36, +15.88, +17.41, +18.96, +20.51, +22.08,
        +23.65, +25.24, +26.82, +28.41, +30.00]
    y1 = [
        +35.26, +35.91, +36.55, +37.19, +37.82, +38.44, +39.06, +39.67, +40.27,
        +40.87, +41.46, +42.04, +42.62, +43.18, +43.74, +44.28, +44.82, +45.35,
        +45.87, +46.38, +46.88, +47.36, +47.84, +48.30, +48.75, +49.19, +49.61,
        +50.02, +50.41, +50.80, +51.16, +51.51, +51.85, +52.17, +52.47, +52.75,
        +53.02, +53.27, +53.50, +53.71, +53.90, +54.08, +54.23, +54.36, +54.48,
        +54.57, +54.64, +54.69, +54.73, +54.74]
    ax.plot(x1, y1, 'k--', transform=DATA_CRS)

    x2 = [
        -30.00, -28.41, -26.82, -25.24, -23.65, -22.08, -20.51, -18.96, -17.41,
        -15.88, -14.36, -12.85, -11.36, -09.89, -08.43, -06.99, -05.57, -04.17,
        -02.79, -01.44, -00.10, +01.22, +02.51, +03.79, +05.04, +06.27, +07.48,
        +08.67, +09.83, +10.98, +12.10, +13.21, +14.29, +15.35, +16.40, +17.42,
        +18.43, +19.42, +20.39, +21.34, +22.27, +23.19, +24.10, +24.98, +25.86,
        +26.71, +27.55, +28.38, +29.20, +30.00]
    y2 = [
        -54.74, -54.73, -54.69, -54.64, -54.57, -54.48, -54.36, -54.23, -54.08,
        -53.90, -53.71, -53.50, -53.27, -53.02, -52.75, -52.47, -52.17, -51.85,
        -51.51, -51.16, -50.80, -50.41, -50.02, -49.61, -49.19, -48.75, -48.30,
        -47.84, -47.36, -46.88, -46.38, -45.87, -45.35, -44.82, -44.28, -43.74,
        -43.18, -42.62, -42.04, -41.46, -40.87, -40.27, -39.67, -39.06, -38.44,
        -37.82, -37.19, -36.55, -35.91, -35.26]
    ax.plot(x2, y2, 'k--', transform=DATA_CRS)

    ax.plot([-30.0, +30.0], [+35.2644, -35.2644], 'k--', transform=DATA_CRS)


def project(m):
    """
    Calculates lune coordinates (γ, δ) for a given moment tensor based
    on the formulation of Tape & Tape (2012).

    Parameters
    ----------
    m : ndarray of shape (3, 3)
        Seismic moment tensor as symmetric 2-D array.

    Returns
    -------
    γ : float
        Lune longitude. Unit: deg
    δ : float
        Lune latitute. Unit: deg
    """
    # Eigenvalues in descending order
    lambda_vec = np.linalg.eigvals(m)
    lambda_vec = np.real(np.take(lambda_vec, np.argsort(lambda_vec)[::-1]))
    λ1, λ2, λ3 = lambda_vec
    lambda_norm = np.linalg.norm(lambda_vec)

    if λ1 != λ3:
        γ = np.rad2deg(
            np.arctan((-λ1 + 2 * λ2 - λ3) / (np.sqrt(3.0) * (λ1 - λ3))))
    else:
        γ = 0.0

    if np.sum(lambda_vec) != 0:
        β = max(-1, min(1, np.sum(lambda_vec) / (np.sqrt(3.0) * lambda_norm)))
        δ = 90.0 - np.rad2deg(np.arccos(β))
    else:
        δ = 0.0

    return (γ, δ)


def project_transform(m):
    """
    Calculates lune coordinates (γ, δ) for a given moment tensor based
    on the formulation of Tape & Tape (2012) and transform them to the
    data coordinates (x, y).
    """
    γ, δ = project(m)
    x, y = LUNE_PROJECTION.transform_point(γ, δ, DATA_CRS)
    return (x, y)


__all__ = ['draw_lune', 'project', 'project_transform']
