"""
Utility functions for matplotlib.
"""

from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np


def transform_data2axes(ax, points_indata):
    """
    Transforms points in matplotlib ``data`` coordinate system to
    ``axes`` coordinate system.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes.Axes`
        Matplotlib ``Axes`` object to plot in.

    points_indata : array-like
        Single point as a ``n_dimentions``-tuple (e.g. 2-tuple) or many
        points as a list of ``n_dimensions``-tuples or ndarray of shape
        ``(n_points, n_dimensions)``, where ``n_dimensions`` is the
        number of spatial dimensions.

    Returns
    -------
    poins_inaxes : ndarray of shape (n_points, n_dimensions)
        Transformed points.
    """

    # ## First, from data to display coordinate
    points_indisp = ax.transData.transform(points_indata)

    # ## Then from display to axes coordinate
    inv = ax.transAxes.inverted()
    return inv.transform(points_indisp)


def sci_tickformatter(scilimits=(-3, 3)):
    """
    Format matplotlib tick values as scientific notation.

    Parameters
    ----------
    scilimits : (m, n), tuple of two int
        A tuple containing the powers of 10 that determine the
        switch-over threshold. Numbers below ``10**m`` and above
        ``10**n`` will be displayed in scientific notation.
        Use (0, 0) to include all numbers. Default is (-3, 3).

    Returns
    -------
    formatter : py:class:`matplotlib.ticker.ScalarFormatter`
        Formatter that format ticks as scientific numbers.
    """

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(scilimits)
    return formatter


def sci_strformatter(x, decimals=2, precision=None, exponent=None):
    """
    String formatting of scientific notation.

    Returns a string representation of the scientific notation of a
    number formatted for use with LaTeX or Mathtext.

    Paramteres
    ----------
    x : float
        Input value.
    decimals : int, optional
        Number of decimal places (digits) to round to.
    precision : int, optional
        Number of decimal places (digits) to show.
    exponent : int, optional
        Exponet value to use (i.e. 10 to the power of `exponent`).

    Returns
    -------
    s : str
        Scientific notation.

    References
    ----------
    .. [1] https://stackoverflow.com/a/18313780/3202380
    """
    if not exponent:
        exponent = int(np.floor(np.log10(abs(x))))

    coeff = round(x/float(10**exponent), decimals)

    if precision is None:
        precision = decimals

    return r"${0:.{1}f}\times10^{{{2:d}}}$".format(coeff, precision, exponent)


def create_cbar_axes(ax, position='right', size='5%', pad='3%'):
    """
    Create a new axes into which a colorbar can be drawn. Using an axes
    divider, created axes has the same height (or width) of the main axes.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes.Axes`
        Parent axes from which space for a new colorbar axes will be
        stolen.

    position : {'left', 'right', 'bottom', 'top'}, optional
        Position of created new axes relative to the main axes. Default
        is ``'right'``.

    size : float or str, optional
        Height of the created axes. Default is ``'5%'``.

    pad : float or str, optional
        Set the space between the main and new axes. Default is ``'3%'``.

    Returns
    -------
    cax : :py:class:`matplotlib.axes.Axes` object
        Axes into which the colorbar will be drawn.

    Notes
    -----
    An alternative way is to use
    :py:func:`matplotlib.figure.Figure.add_axes` (e.g.
    ``cax=fig.add_axes([left, bottom, width, height]``)
    """

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size, pad=pad)
    return cax


__all__ = """
    transform_data2axes
    sci_tickformatter
    sci_strformatter
    create_cbar_axes
""".split()
