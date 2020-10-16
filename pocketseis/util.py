"""
Utility functions for PocketSeis.
"""

from datetime import datetime, timedelta, timezone
import os.path as op

import numpy as np


def time2index(t, deltat, snap=np.round):
    """
    Returns indedx of  a time value in array of time axis (e.g. of a
    seismic trace) assuming that starting time is zero.

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
    return snap(t/deltat).astype(np.int)


def make_time_array(tmin, tmax, deltat):
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
    start = round(tmin/deltat) * deltat
    stop = round(tmax/deltat) * deltat
    num = int(round((stop-start) / deltat)) + 1
    return np.linspace(start, stop, num, axis=-1)


def round_to_day(timestamp, ceiling=False):
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


def isleapyear(year):
    """
    A leap year is exactly divisible by 4 except for century years (
    years ending with 00). A century year is a leap year if it is
    perfectly divisible by 400.
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


def get_data_file(filename):
    return op.join(op.split(__file__)[0], 'data', filename)
