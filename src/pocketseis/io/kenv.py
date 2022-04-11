import gzip
import logging
import os.path as op
import tempfile
from zipfile import ZipFile

import numpy as np
import xarray as xr

from pyrocko.util import setup_logging, str_to_time

from pocketseis.util import isleap, time2index


setup_logging('read_kenv_5min')
logger = logging.getLogger(__name__)


def read_kenv_5min(zip_fname, as_xarray=True):
    """
    Read yearly GPS data file ``5-minute Sample Rate Final Solution``
    in *.kenv format (east, north, up time series).

    Parameters
    ----------
    zip_fname : str
        Path/filename. File must be compressed as ``zip`` format and its
        basename have ``<site-id>.<year>.kenv.zip`` pattern.
    as_xarray : bool (default: True)
        If True, returns output as `xarray.DataArray` object, otherwise
        returns `numpy.ndarray`.

    Returns
    -------
    da : `numpy.ndarray` or `xarray.DataArray`
        GPS data (time and amplitude values) for one year sampled at
        5 minute rate. Gaps in data are filled with `numpy.nan`.
        If `as_xarray=False`, ndarray of shape (105120, 4), whose columns
        are 0: time, 1: east, 2: north, and 3: up.
        If `as_xarray=True`, `DataArray`, where its `data` is 2-D array
        of shape (105120, 3) and `dims` (dimension names) are 'time' and
        `component`. Coordinates of 'component' dimension are 'east',
        'north' and 'up'. Coordinates of 'time' dimension are floating
        timestamps (array of shape of (105120,)).
    """
    # Dict of named place-holders
    station_id, year = op.basename(zip_fname).split('.')[:2]
    dph = dict(station_id=station_id, year=int(year))

    n_days = 366 if isleap(dph['year']) else 365

    deltat = 300
    datalen_perday = int(86400 / deltat)
    datalen_peryear = n_days * datalen_perday

    # Allocate data array
    data_yearly = np.zeros((datalen_peryear, 4))

    if not op.exists(zip_fname):
        logger.exception('No such file: {}'.format(zip_fname))

    # Temporary directory to extract *.gz files
    temp_dir = tempfile.TemporaryDirectory(suffix='ps_kenv_')

    with ZipFile(zip_fname) as zip_obj:   # Open `*.zip` file
        for i_doy, doy in enumerate(range(1, n_days + 1)):
            dph['doy'] = doy
            t0 = str_to_time('{year}-{doy:03d} 00:00:00'.format_map(dph),
                             format='%Y-%j %H:%M:%S')

            data_daily = np.zeros((datalen_perday, 4))
            i1 = i_doy * datalen_perday
            i2 = (i_doy+1) * datalen_perday

            try:
                gz_fname = zip_obj.extract(
                    '{station_id}.{year}.{doy:03d}.kenv.gz'.format_map(dph),
                    path=temp_dir.name)
            except KeyError:
                logger.warning('[{station_id}.{year}] No data for '
                               'day {doy:03d}'.format_map(dph))
                data_yearly[i1:i2, 0] = t0 + np.arange(datalen_perday)*deltat
                data_yearly[i1:i2, [1, 2, 3]] = np.nan
            else:
                with gzip.open(gz_fname, 'rb') as fobj:   # Open *.gz file

                    has_data = set()
                    for line in fobj.readlines():
                        if line.startswith(b'site'):   # Skip header line
                            continue
                        toks = line.split()
                        # Sample time and amplitudes (east, north, vertical)
                        t_rel = int(toks[7])
                        t_abs = t0 + t_rel
                        ae, an, av = map(float, toks[8:11])

                        # There might be gaps in data
                        idx = time2index(t_rel, deltat)
                        has_data.add(idx)
                        data_daily[idx] = (t_abs, ae, an, av)

                # Fill-in the gaps
                gap_indx = {*np.arange(datalen_perday)} - has_data
                if len(gap_indx) != 0:
                    gap_indx = np.asarray(list(gap_indx))
                    data_daily[gap_indx, 0] = t0 + gap_indx * deltat
                    data_daily[gap_indx[:, np.newaxis], [1, 2, 3]] = np.nan

                # Feed daily data-array into yearly data-array
                data_yearly[i1:i2] = data_daily
            finally:
                if (doy % 90 == 0) or (doy == n_days):
                    logger.info('[{station_id}.{year}] Parsed file for '
                                'day {doy:3d}'.format_map(dph))

    # Clean up temporary dir
    temp_dir.cleanup()

    if as_xarray is False:
        return data_yearly

    ydata = data_yearly[:, 1:]
    times = data_yearly[:, 0]
    comps = ['east', 'north', 'up']
    return xr.DataArray(ydata, coords=[('time', times), ('component', comps)])
