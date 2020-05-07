import numpy as np


def time2index(t, deltat, snap=round):
    """
    Returns indedx of  a time value in array of time axis (e.g. of a
    seismic trace) assuming that starting time is zero.

    Parameters
    ----------
    t : float
        Time value.
    deltat : float
        Sampling interval in the same unit as `t`.
    snap : callable
        By default, the index where to put `t` in the array of times is
        determined by rounding of `t` to sampling instances `deltat`
        using Python's buit-in function :py:func:`round`. This behaviour
        can be changed with the `snap` argument.

    Returns
    -------
    idx : int
        Index of the time value.
    """
    return int(snap(t/deltat))


def make_time_array(tmin, tmax, deltat):
    """
    Construct array of time values.

    Parameters
    ----------
    tmin, tmax : float
        Time of first and last samples, respectively.
    deltat : float
        Sampling interval.

    Returns
    -------
    times : ndarray
        Array of time values.
    """
    t1 = round(tmin/deltat) * deltat
    t2 = round(tmax/deltat) * deltat
    nt = int(round((t2-t1) / deltat)) + 1
    return np.linspace(t1, t2, nt)


def tuple6_to_symmat(a):
    """
    Create symmetric 3-by-3 moment-tensor matrix from its 6
    independent values.

    Parameters
    ----------
    a : tuple of 6 float
        A tuple of six independent components of moment tensor in
        (M11, M22, M33, M12, M13, M23) order.

    Returns
    -------
    m : ndarray, shape (3, 3)
        Plain seismic moment tensor as symmetric 2-D array.
    """
    a11, a22, a33, a12, a13, a23 = a
    return np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33]], dtype=np.float)
