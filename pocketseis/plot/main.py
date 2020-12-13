from matplotlib import colors, cm
import numpy as np

from pyrocko.util import time_to_str

from .mpl_util import transform_data2axes, sci_tickformatter
from pocketseis.point_source.mtensor import magnitude_to_moment
from pocketseis.util import round_day


KM2M = 1000.
M2KM = 1. / KM2M


def plot_cake_model(earth_model, ax, depth_min=None, depth_max=None,
                    vp_kwargs=None, vs_kwargs=None):
    """
    Helper function to plot a layered velocity model.

    Parameters
    ----------
    earth_model : :py:class:`pyrocko.cake.LayeredModel` object
        Layered earth model.
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    depth_min : float (optional)
        Minimum depth to show. Unit: m.
    depth_max : float (optional)
        Maximum depth to show. Unit: m.
    vp_kwargs : dict (optional)
        Dict with keywords passed to `ax.plot()` call to create Vp line-plot.
    vS_kwargs : dict (optional)
        Dict with keywords passed to `ax.plot()` call to create VS line-plot.
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

    ax.plot(vp*M2KM, z*M2KM, label=r'$V_p$', **vp_kw)
    ax.plot(vs*M2KM, z*M2KM, label=r'$V_s$', **vs_kw)
    ax.invert_yaxis()


def plot_catalog_fmd(mags, ax, deltam=0.1):
    """
    Frequency-magnitude distribution.

    b-value is calculated by using maximum-likelihood method of Aki (1965).
    The b-value is estimated by using the maximum-likelihood method of
    Aki (1965) and Godano et al. (2014).

    Magnitude of completeness is calculated by using maximum-curvature method.

    Parameters
    ----------
    mags : array-like
        Seismic event magnitudes.
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    deltam : float, default: 0.1
        Binning of the earthquake magnitudes.

    References
    ----------
    .. [1] Aki, K. (1965). Maximum likelihood estimate of b in the formula
       log N= a-bM and its confidence limits. Bull. Earthq. Res. Inst.,
       Tokyo Univ., 43, 237-239.
    .. [2] Godano, C., Lippiello, E., & de Arcangelis, L. (2014). Variability
       of the b value in the Gutenberg–Richter distribution. Geophysical
       Journal International, 199(3), 1765-1771.
    """
    mags = np.asarray(mags)

    bins = np.arange(mags.min(), mags.max()+deltam, deltam)
    freq, _ = np.histogram(mags, bins=bins)

    # ## Magnitude of completeness (Mc) -> maximum curvature method
    mc = bins[np.where(freq == np.max(freq))].item()

    m_cut = mags.max()
    mags_truncated = mags[((mags >= mc) & (mags <= m_cut))]

    marker_kwargs = dict(mfc='none', mew=2.5, ms=7)

    # ## 1.: Incremental frequency
    ax.semilogy(bins[:-1], freq, 's', mec='#348ABD', label='Incremental',
                **marker_kwargs)

    # ## 2.: Cumulative frequency
    freq_cum = []
    for m in bins[:-1]:
        freq_cum.append(mags[mags >= m].size)

    ax.plot(bins[:-1], freq_cum, 'o', mec='#988ED5', label='Cumulative',
            **marker_kwargs)

    # ## 3. Fit Gutenberg-Richter power law
    b_val = np.log10(np.e) / (mags_truncated.mean() - (mc-0.5*deltam))
    a_val = np.log10(mags_truncated.size) + b_val*mc

    # ## 4. Plot the fitted curve
    def gr_law(mag):
        """Gutenberg-Richter law"""
        return 10.**(a_val - b_val*mag)

    ylim = ax.get_ylim()
    ax.plot(bins, gr_law(bins), '--k', alpha=0.7,
            label=r'$b\,$-value={:.2f}'.format(b_val))
    ax.set_ylim(ylim)

    # ## 5. Mc indicator symbol
    x_symb, _ = transform_data2axes(ax, (mc, 0.))
    ax.plot(x_symb, 0.970, 'v', color='#777777', ms=14, transform=ax.transAxes)
    ax.text(x_symb, 1.005, r'$M_c=${:.1f}'.format(mc), transform=ax.transAxes,
            ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # ## Finishing touch
    ax.legend()
    ax.set_xlabel(r'Magnitude, $M$')
    ax.set_ylabel(r'log$_{10} N$, $N=$\# events w/ magnitude $\geq M$')


def set_xticks_as_daymonth(ax):
    """
    Helper function to set xtick labels as ``%d %b``
    """
    xticklabels = [time_to_str(x, format='%d %b') for x in ax.get_xticks()]
    ax.set_xticklabels(xticklabels)


def plot_catalog_timehist(times, ax, deltat_days=1, hist_kwargs=None):
    """
    Origin-time histogram of catalogue data.

    Parameters
    ----------
    times : array-like
        Seismic event origin times as floating timestamps.
    ax : :py:class:`matplotlib.axes.Axes` object
        Matplotlib axes instance to plot in.
    deltat_days : int or float
        Binning of the earthquake times as a factor of days (e.g. set
        ``deltat_days=1`` for daily and ``deltat_days=7`` for weekly plot).
    hist_kwargs : dict
        Dict with keywords passed to `ax.hist()` call.
    """
    deltat = deltat_days * (24*3600)
    times = np.asarray(times)
    bins = np.arange(round_day(times.min()),
                     round_day(times.max(), ceiling=True)+deltat,
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
    ax.set_xlim(times.min()-deltat, times.max()+deltat)
    set_xticks_as_daymonth(ax)


def plot_mag_timeline(mags, times, ax, cmap=None, scatter_kwargs=None):
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
    ax.set_xlim(times.min()-day, times.max()+day)
    set_xticks_as_daymonth(ax)
