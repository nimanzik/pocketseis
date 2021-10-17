from matplotlib import colors, cm
import numpy as np

from pyrocko.util import time_to_str

from .mpl_util import transform_data2axes, sci_tickformatter
from pocketseis.plot.colors import darken_color
from pocketseis.mtensor import magnitude_to_moment
from pocketseis.util import round_day
from pocketseis.zmap import calculate_abs_fmd, calculate_cum_fmd, \
    gutenberg_richter


KM2M = 1.0e+3
M2KM = 1.0e-3


def plot_cake_model(ax, earth_model, depth_min=None, depth_max=None,
                    vp_kwargs=None, vs_kwargs=None, invert_yaxis=True):
    """
    Helper function to plot a layered velocity model.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    earth_model : :py:class:`pyrocko.cake.LayeredModel` object
        Layered earth model.
    depth_min : float (optional)
        Minimum depth to show. Unit: m.
    depth_max : float (optional)
        Maximum depth to show. Unit: m.
    vp_kwargs : dict (optional)
        Dict with keywords passed to `ax.plot()` call to create Vp line-plot.
    vS_kwargs : dict (optional)
        Dict with keywords passed to `ax.plot()` call to create VS line-plot.
    invert_yaxis : bool (optional, default: True)
        Whether to invert the y-axis (axis of depth values).
    """
    model = earth_model.extract(depth_min=depth_min, depth_max=depth_max)
    z, vp, vs = np.transpose(np.asarray(model.to_scanlines())[:, :3])
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('Depth [km]')

    vp_kw = dict(color='red', alpha=0.8)
    if vp_kwargs is not None:
        vp_kw['color'] = vp_kwargs.pop('c', 'red')
        vp_kw.update(vp_kwargs)

    vs_kw = dict(color='blue', alpha=0.8)
    if vs_kwargs is not None:
        vs_kw['color'] = vs_kwargs.pop('c', 'blue')
        vs_kw.update(vs_kwargs)

    ax.plot(vp * M2KM, z * M2KM, label=r'$V_p$', **vp_kw)
    ax.plot(vs * M2KM, z * M2KM, label=r'$V_s$', **vs_kw)

    if invert_yaxis:
        ax.invert_yaxis()


def plot_cake_models(ax, earth_models, **kwargs):
    """
    Plot many layered velocity models in one single frame.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    earth_models : list of :py:class:`pyrocko.cake.LayeredModel` objects
        Layered earth models.

    Other Parameters
    ----------------
    **kwargs
        Additional parameters are passed along to the `~plot_cake_models`
        method.
    """
    invert_yaxis = kwargs.pop('invert_yaxis', True)

    for earth_model in earth_models:
        plot_cake_model(ax, earth_model, invert_yaxis=False, **kwargs)

    if invert_yaxis:
        ax.invert_yaxis()


def plot_fmd(ax, mags, mc, b_val, a_val, bin_size=0.1, axis_label='both',
             legend=True, mc_indicator=True):
    """
    Plot frequency-magnitude distribution (FMD).

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    mags : array-like
        Seismic event magnitudes.
    mc : float
        Magnitude of completeness (cutoff magnitude).
    b_val : float
        Gutenberg-Richter law b-value.
    a_val : float
        Gutenberg-Richter law a-value.
    bin_size : float, default: 0.1
        Binning of the earthquake magnitudes.
    axis_label : {'x', 'y', 'both', None}, default: 'both'
        Add axis label. Set to `None` to add no axis label.
    legend : bool, default: True
        Whether to add legend.
    mc_indicator : bool, default: True
        Whether to add Mc (cutoff magnitude) indicator symbol.
    """

    marker_kw = dict(mfc='none', mew=1, ms=4.5)
    bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    c1 = darken_color('#988ed5', 0.10)
    c2 = darken_color('#348abd', 0.05)
    c3 = '#777777'

    mags = np.asarray(mags)
    abs_freqs, centers = calculate_abs_fmd(mags, bin_size)
    cum_freqs = calculate_cum_fmd(abs_freqs)

    # === Cumulative FMD ===
    p1 = ax.plot(centers, np.log10(cum_freqs),
                 'o', mec=c1, label='Cum. FMD', **marker_kw)

    # === Absolute FMD ===
    indxs = np.where(abs_freqs != 0.0)
    p2 = ax.plot(centers[indxs], np.log10(abs_freqs[indxs]),
                 'D', mec=c2, label='Abs. FMD', **marker_kw)

    # === Fitted GR model ===
    bot, top = ax.get_ylim()
    ax.plot(centers, np.log10(gutenberg_richter(centers, b_val, a_val)),
            'k--', lw=2, alpha=0.8)
    ax.set_ylim(bot, top)

    # === Add axis label(s) if required ===
    if axis_label:
        xlabel = 'Magnitude'
        ylabel = r'log$_{10} N$, $N=\#$ of events'
        if axis_label == 'both':
            ax.set(xlabel=xlabel, ylabel=ylabel)
        elif axis_label == 'x':
            ax.set(xlabel=xlabel)
        elif axis_label == 'y':
            ax.set(ylabel=ylabel)

    # === Add legend if required ===
    if legend:
        handles = p1 + p2
        labels = [p.get_label() for p in handles]
        ax.legend(
            handles, labels, handlelength=1, handletextpad=0.5, borderpad=0.25)

    # === Add Mc indicator symbol if required ===
    if mc_indicator:
        x_symb, _ = transform_data2axes(ax, (mc, 0.))
        ax.plot(x_symb, 0.96, 'v', color=c3, ms=7, transform=ax.transAxes)
        ax.text(
            x_symb, 1.01, r'$M_c={:.2f}$'.format(mc), transform=ax.transAxes,
            ha='center', va='bottom', bbox=bbox_props)


def set_xticks_as_daymonth(ax, rotation=0.0):
    """
    Helper function to set xtick labels as ``%d %b``
    """
    xticklabels = [time_to_str(x, format='%d %b') for x in ax.get_xticks()]
    ax.set_xticklabels(xticklabels, rotation=rotation)


def set_xticks_as_fulldate(ax, rotation=0.0):
    """
    Helper function to set xtick labels as ``%d %b %Y``
    """
    xticklabels = [time_to_str(x, format='%d %b %Y') for x in ax.get_xticks()]
    ax.set_xticklabels(xticklabels, rotation=rotation)


def plot_catalog_timehist(ax, times, deltat_days=1, hist_kwargs=None):
    """
    Origin-time histogram of catalogue data.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    times : array-like
        Seismic event origin times as floating timestamps.
    deltat_days : int or float
        Binning of the earthquake times as a factor of days (e.g. set
        ``deltat_days=1`` for daily and ``deltat_days=7`` for weekly plot).
    hist_kwargs : dict
        Dict with keywords passed to `ax.hist()` call.
    """
    deltat = deltat_days * (24 * 3600)
    times = np.asarray(times)
    bins = np.arange(round_day(times.min()),
                     round_day(times.max(), ceiling=True) + deltat,
                     deltat)

    hist_kw = dict(align='mid', fc='#ccb974', ec='dimgray')
    if hist_kwargs is not None:
        hist_kw['fc'] = hist_kwargs.pop('facecolor', '#ccb974')
        hist_kw['ec'] = hist_kwargs.pop('edgecolor', 'dimgray')
        _ = hist_kwargs.pop('align', None)   # suppress user's value
        hist_kw.update(hist_kwargs)

    ax.hist(times, bins=bins, **hist_kw)

    ax.set_ylabel('Counts [#]')

    # First set xlim, then set xticklabels!
    ax.set_xticks([round_day(x) for x in ax.get_xticks()])
    ax.set_xlim(times.min() - deltat, times.max() + deltat)


def plot_mags_timeline(ax, mags, times, cmap=None, scatter_kwargs=None):
    """
    Magnitudes time-line.
    """
    mags = np.asarray(mags)
    times = np.asarray(times)

    if not cmap:
        cmap = cm.inferno_r

    norm = colors.Normalize(vmin=mags.min(), vmax=mags.max())
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.zeros(0))

    marker_colors = mappable.to_rgba(mags)

    scat_kw = dict(c=marker_colors, s=50, alpha=0.8)
    if scatter_kwargs is not None:
        _ = scatter_kwargs.pop('c', None)   # suppress user's value
        scat_kw.update(scatter_kwargs)

    ax.scatter(times, mags, **scat_kw)

    # ## Cumulative seismic moment
    moments_cum = np.cumsum(magnitude_to_moment(mags))
    ax2 = ax.twinx()
    ax2.plot(times, moments_cum, 'k', alpha=0.8)
    ax2.set_ylabel('Cumulative seismic moment')
    ax2.yaxis.set_major_formatter(sci_tickformatter(scilimits=(0, 0)))
    ax2.grid(False)

    ax.set_ylabel('Magnitude')

    # First set xlim, then set xticklabels!
    day = 24 * 3600
    ax.set_xlim(times.min() - day, times.max() + day)
