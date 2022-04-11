"""
This module contains functions and classes to manipulate color codes.
"""

import colorsys
import os.path as op
import pprint

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from pocketseis.util import get_data_file


def list_sci_cmaps():
    """
    This function lists available scientific colormaps provided by
    ``'ScientificColourMaps7'``.

    References
    ----------
    .. [1] Crameri, F. "Scientific colour maps2, Zenodo (2018),
       doi:10.5281/zenodo.1243862
    .. [2] Crameri, F. "Geodynamic diagnostics, scientific visualization,
       and StagLab 3.0." Geodynamic Model Development 11 (2018): 2541-2562.
    """
    pp = pprint.PrettyPrinter(sort_dicts=False)
    pp.pprint({
        "Sequential": [
            'batlow', 'batlowW', 'batlowK',
            'devon', 'lajolla', 'bamako',
            'davos', 'bilbao', 'nuuk',
            'oslo', 'grayC', 'hawaii',
            'lapaz', 'tokyo', 'buda',
            'acton', 'turku', 'imola'],
        "Diverging": [
            'broc', 'cork', 'vik',
            'lisbon', 'tofino', 'berlin',
            'roma', 'bam', 'vanimo'],
        "Multi-sequential": [
            'oleron', 'bukavo', 'fes'],
        "Cyclic": [
            'romaO', 'bamO', 'brocO', 'corkO', 'vikO']
    })


def get_sci_cmap(name):
    """
    Get a colormap instance provided by ``'ScientificColourMaps7'``.

    Parameters
    ----------
    name : str
        Scientific colormap name.

    Returns
    -------
    cmap : :py:class:`matplotlib.colors.LinearSegmentedColormap`
        Colormap object.

    Raises
    ------
    ValueError
        If colormap name is not recognized.

    References
    ----------
    .. [1] Crameri, F. "Geodynamic diagnostics, scientific visualization,
       and StagLab 3.0." Geodynamic Model Development 11 (2018): 2541-2562.
    """
    file_path_name = op.join('scientific_colormaps', name+'.txt')
    cm_file = get_data_file(file_path_name)

    if not op.exists(cm_file):
        raise ValueError(
            f"Colormap '{name}' is not recognised. Use function "
            f"'list_sci_cmaps()' to see possible values.")

    cm_data = np.loadtxt(cm_file)
    cmap = LinearSegmentedColormap.from_list(name, cm_data)
    return cmap


def hex2rgb(c_hex, to_gmtcolor=False):
    """
    Convert Hex to RGB colors.

    Parameters
    ----------
    c_hex : str
        Hex color value starting with ``#``. Example: ``'#88b7c6'``
    to_gmtcolor : bool
        If True, the output color is transformed into the string form
        `R/G/B` which GMT expects. Default is False.

    Returns
    -------
    c_rgb : tuple of 3 int or str
        RGB color value in the range of 0-255. If `to_gmtcolor=True`,
        values are separated by ``/`` and returned in string format.
    """
    c_hex = c_hex.lstrip('#')
    n = len(c_hex)

    # Base 16 integers
    c_rgb = tuple(int(c_hex[i:i+2], 16) for i in range(0, n, 2))

    if to_gmtcolor:
        return '/'.join([str(v) for v in c_rgb])

    return c_rgb


def rgb2hex(c_rgb):
    """
    Convert RGB color to Hex colors.

    Parameters
    ----------
    c_rgb : tuple of 3 int or str
        RGB color value in the range of 0-255. If given as `str`, values
        must be separated by ``/``. Example: ``'136/183/198'``

    Returns
    -------
    c_hex : str
        Hex color value.
    """
    if isinstance(c_rgb, str):
        c_rgb = map(int, c_rgb.split('/'))

    return '#'+''.join(['%02x' % v for v in c_rgb])


def _adjust_lightness(c_rgb, factor):
    """
    Change lightness of a color.

    Parameters
    ----------
    c_rgb : tuple of 3 int
        RGB color value in the range of 0-255.

    factor : float
        Amount of adjustment, between 0-1.

    Returns
    -------
    c_rgb_new : tuple of 3 int
        Adjusted RGB color value in the range of 0-255.
    """

    # RGB -> HLS (hue, lightness, saturation)
    r, g, b = [v / 255.0 for v in c_rgb]
    h, l_old, s = colorsys.rgb_to_hls(r, g, b)
    l_new = max(min(l_old * factor, 1.0), 0.0)
    c_rgb_new = tuple(int(v * 255.0) for v in colorsys.hls_to_rgb(h, l_new, s))
    return c_rgb_new


def lighten_color(color, factor):
    """
    Lighten a color towards white.

    Parameters
    ----------
    color : tuple of 3 int or str
        RGB color value in the range of 0-255, or HEX color code.
        If RGB color is given as `str`, values must be separated by ``/``
        (e.g. ``'136/183/198'``).
        If HEX color is given, it must start with ``#`` (e.g. ``'#88b7c6'``).

    factor : float
        Amount of lightness, between 0-1.

    Returns
    -------
    color_new : tuple of 3 int or str
        RGB color value in the range of 0-255, or HEX color code.
    """
    if isinstance(color, str):
        isstr = True
        if color.startswith('#'):
            ishex = True
            c_rgb = hex2rgb(color)
        else:
            ishex = False
            c_rgb = tuple(map(int, color.split('/')))
    else:
        isstr = False
        ishex = False
        c_rgb = color

    color_new = _adjust_lightness(c_rgb, 1.0 + factor)

    if isstr:
        if ishex:
            return rgb2hex(color_new)
        else:
            return '{}/{}/{}'.format(*color_new)
    return color_new


def darken_color(color, factor):
    """
    Darken a color towards black.

    Parameters
    ----------
    color : tuple of 3 int or str
        RGB color value in the range of 0-255, or HEX color code.
        If RGB color is given as `str`, values must be separated by ``/``
        (e.g. ``'136/183/198'``).
        If HEX color is given, it must start with ``#`` (e.g. ``'#88b7c6'``).

    factor : float
        Amount of darkness, between 0-1.

    Returns
    -------
    color_new : tuple of 3 int or str
        RGB color value in the range of 0-255, or HEX color code.
    """
    if isinstance(color, str):
        isstr = True
        if color.startswith('#'):
            ishex = True
            c_rgb = hex2rgb(color)
        else:
            ishex = False
            c_rgb = tuple(map(int, color.split('/')))
    else:
        isstr = False
        ishex = False
        c_rgb = color

    color_new = _adjust_lightness(c_rgb, 1.0 - factor)

    if isstr:
        if ishex:
            return rgb2hex(color_new)
        else:
            return '{}/{}/{}'.format(*color_new)
    return color_new


__all__ = """
    list_sci_cmaps
    get_sci_cmap
    hex2rgb
    rgb2hex
    lighten_color
    darken_color
""".split()
