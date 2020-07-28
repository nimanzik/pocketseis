"""
This module contains functions and classes to manipulate color codes.
"""

import colorsys


def hex2rgb_tuple(c_hex):
    """
    Convert Hex to RGB colors.

    Parameters
    ----------
    c_hex : str
        Hex color value starting with ``#``. Example: ``'#88b7c6'``

    Returns
    -------
    c_rgb : tuple of 3 int
        RGB color value in the range of 0-255.
    """

    c_hex = c_hex.lstrip('#')
    n = len(c_hex)

    # Base 16 integers
    c_rgb = tuple(int(c_hex[i:i+2], 16) for i in range(0, n, 2))
    return c_rgb


def hex2rgb_str(c_hex):
    """
    Convert Hex to RGB colors.

    Parameters
    ----------
    c_hex : str
        Hex color value starting with ``#``. Example: ``'#88b7c6'``

    Returns
    -------
    c_rgb : str
        RGB color value in the range 0-255, separated by ``/``.
    """

    c_rgb = hex2rgb_tuple(c_hex)
    return '/'.join([str(v) for v in c_rgb])


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
    r, g, b = [v/255. for v in c_rgb]
    h, l_old, s = colorsys.rgb_to_hls(r, g, b)
    l_new = max(min(l_old*factor, 1.), 0.)
    c_rgb_new = tuple(int(v*255.) for v in colorsys.hls_to_rgb(h, l_new, s))
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
        If HEX color is given, it must starts with ``#`` (e.g. ``'#88b7c6'``).

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
            c_rgb = hex2rgb_tuple(color)
        else:
            ishex = False
            c_rgb = tuple(map(int, color.split('/')))
    else:
        isstr = False
        ishex = False
        c_rgb = color

    color_new = _adjust_lightness(c_rgb, 1.+factor)

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
        If HEX color is given, it must starts with ``#`` (e.g. ``'#88b7c6'``).

    factor : float
        Amount of darkness, between 0-1.

    Returns
    -------
    color_new : tuple of 3 int or str
        RGB color value in the range of 0-255, HEX color code.
    """
    if isinstance(color, str):
        isstr = True
        if color.startswith('#'):
            ishex = True
            c_rgb = hex2rgb_tuple(color)
        else:
            ishex = False
            c_rgb = tuple(map(int, color.split('/')))
    else:
        isstr = False
        ishex = False
        c_rgb = color

    color_new = _adjust_lightness(c_rgb, 1.-factor)

    if isstr:
        if ishex:
            return rgb2hex(color_new)
        else:
            return '{}/{}/{}'.format(*color_new)
    return color_new


__all__ = """
    hex2rgb_tuple
    hex2rgb_str
    rgb2hex
    lighten_color
    darken_color
""".split()
