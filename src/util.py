"""
Utility functions for PocketSeis.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def time_to_index(t, deltat, snap=np.round):
    """
    Returns index of a time value in array of time axis (for instance,
    of a seismic trace) assuming that starting time is zero.

    Parameters
    ----------
    t : float or array-like
        Time value(s).
    deltat : float
        Sampling interval in the same unit as `t`.
    snap : callable
        By default, the index where to put `t` in the array of times is
        determined by rounding of `t` to sampling instances `deltat`
        using Numpy's function :py:func:`numpy.round`. This behaviour
        can be changed with the `snap` argument (for example, Numpy's
        functions :py:func:`numpy.floor` or :py:func:`numpy.ceil`).

    Returns
    -------
    idx : int or ndarray
        Indexes of the time value(s).
    """
    return snap(t / deltat).astype(np.int)


def time_range(tmin, tmax, deltat):
    """
    Construct 1-D array of time values.

    Parameters
    ----------
    tmin, tmax : float
        Time of first and last samples, respectively.
    deltat : float
        Sampling interval.

    Returns
    -------
    times : ndarray, shape of (n_samples,)
        Array of time values.
    """
    start = round(tmin / deltat) * deltat
    stop = round(tmax / deltat) * deltat
    num = int(round((stop - start) / deltat)) + 1
    return np.linspace(start, stop, num, axis=-1)


def round_day(timestamp, ceiling=False):
    """
    Round timestamp to day (i.e. hour, minute and second are zero).

    Parameters
    ----------
    timestamp : float
        Desired time (UTC time zone) as floating timestamp in s.
    ceiling : bool, default: False
        If True, it is round to the beginning of the after `timestamp`.

    Returns
    -------
    t : float
        Time rounded to day (UTC time zone).
    """
    dt = datetime.utcfromtimestamp(timestamp)
    dt -= timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second,
                    microseconds=dt.microsecond)
    if ceiling:
        dt += timedelta(days=1)

    return datetime.timestamp(dt.replace(tzinfo=timezone.utc))


def isleap(year):
    """
    Returns True if year is a leap year, otherwise False.

    A leap year is exactly divisible by 4 except for century years (years
    ending with 00). A century year is a leap year if it is perfectly
    divisible by 400.

    Notes
    -----
    Python standard module `calendar` already provides `isleap` method.
    """
    year = int(year)
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def round_half_up(a, decimals=0):
    """
    The *rounding half up* strategy rounds every number to the nearest
    number with the specified precision, and breaks ties by rounding up
    (the number 1.25 is called a tie with respect to 1.2 and 1.3. In
    cases like this, we must assign a tiebreaker).

    References
    ----------
    .. [1] https://realpython.com/python-rounding/#rounding-half-up
    """
    multiplier = 10**decimals
    return np.floor(np.asarray(a) * multiplier + 0.5) / multiplier


def get_data_file(filename):
    p = Path(__file__).absolute()
    return p.parent.joinpath('data', filename)


def column_stack_3d(array_2d):
    """
    Stack columns of a 2-D array into a 3-D array.

    Parameters
    ----------
    array_2d : ndarray of shape (M, N)

    Returns
    -------
    array_3d : ndarray of shape (N, M, 1)
    """
    assert array_2d.ndim == 2
    n_rows, n_cols = array_2d.shape
    return np.dstack(array_2d).reshape(n_cols, n_rows, 1)


def issymmetric(a, tol=1.e-10):
    """Check if given array/matrix is symmetric"""
    return np.allclose(a, a.T, atol=tol)
