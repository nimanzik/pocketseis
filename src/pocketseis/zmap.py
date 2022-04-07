"""
Utility functions for analysing catalogue data, monstly implemented from
Matlab scripts in ZMAP package.
"""

import numpy as np


def calc_abs_fmd(mags, bin_size=0.1):
    """
    Calculates absolute frequency-magnitude distribution (Abs. FMD).

    Parameters
    ----------
    mags : list or array-like
        Magnitude values.
    bin_size : float, default: 0.1
        Magnitude binning (the rounding of magnitude values).
    """
    mags = np.asarray(mags)

    mag_min = round(mags.min() / bin_size) * bin_size
    mag_max = round(mags.max() / bin_size) * bin_size

    n_centers = int(round((mag_max - mag_min) / bin_size)) + 1
    centers = np.linspace(mag_min, mag_max, n_centers)

    half_bin = bin_size * 0.5
    edges = np.append(centers - half_bin, centers[-1] + half_bin)
    freqs, _ = np.histogram(mags, bins=edges)

    return (freqs, centers)


def calc_cum_fmd(abs_fmd):
    """
    Calculate cumulative frequency-magnitude distribution (Cum. FMD).

    Parameters
    ----------
    abs_fmd : ndarray
        Absotute frequencies of magnitudes returned by
        `~calc_abs_fmd` method.

    Returns
    -------
    cum_fmd : ndarray
        Cumulative frequencies of magnitudes.
    """
    return np.cumsum(abs_fmd[::-1])[::-1]


def calc_mc(mags, bin_size=0.1, correction=0.2):
    """
    Calculate magnitude of completeness (cutoff magnitude) using maximum
    curvature method (Wiemer & Wyss, 2000).

    Parameters
    ----------
    mags : list or array-like
        Magnitude values.
    bin_size : float, default: 0.1
        Magnitude binning (the rounding of magnitude values).
    correction : float, default: 0.2
        Upward adjustment of magnitude unit, to be added to the
        calculated Mc. As suggested by Woessner & Wiemer (2005), the
        maximum curvature method often underestimates Mc on average by
        0.2 magnitude unit for gradually curved frequency-magnitude
        distributions.

    returns
    -------
    mc : float
        Magnitude of completeness.

    References
    ----------
    .. [1] Wiemer, S., & Wyss, M. (2000). Minimum magnitude of
       completeness in earthquake catalogs: Examples from Alaska, the
       western United States, and Japan. Bulletin of the Seismological
       Society of America, 90(4), 859-869.
    """

    mags = np.asarray(mags)
    freqs, centers = calc_abs_fmd(mags, bin_size)
    mc = centers[np.argmax(freqs)]   # Maximum Curvature method
    mc = round((mc + correction) / bin_size) * bin_size
    return mc


def calculate_ba_value(mags, mc, bin_size=0.1):
    """
    Calculate the b-value and a-value of the Gutenberg-Richter law by
    using maximum-likelihood method of Aki (1965) modified by
    Utsu (1965).

    Parameters
    ----------
    mags : list or array-like
        Magnitude values.
    mc : float
        Magnitude of completeness (cutoff magnitude).
    bin_size : float, default: 0.1
        Magnitude binning (the rounding of magnitude values).

    Returns
    -------
    2-tuple: (b-value, a-value)

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
    mags_sel = mags[mags >= mc]
    half_bin = bin_size * 0.5
    b_val = np.log10(np.e) / (mags_sel.mean() - (mc - half_bin))
    a_val = np.log10(mags_sel.size) + b_val * mc

    return (b_val, a_val)


def gutenberg_richter(m, b_val, a_val):
    """Gutenberg-Richter law"""
    return 10.**(a_val - b_val * np.asarray(m))
