import numpy as np

from pyrocko import cake


KM2M = 1000.
M2KM = 1. / KM2M


def empirical_density(vp):
    """
    Calculate material density as a function of P-wave velocity by using
    empirical relation of Ludwig et al. (1970) and Brocher (2005).

    Parameters
    ----------
    vp : float
        P-wave velocity in [m/s].

    Returns
    -------
    rho : float
        Density in [kg/m^3].

    References
    ----------
    .. [1] Brocher, T. M. (2005). Empirical relations between elastic
       wavespeeds and density in the Earth's crust. Bulletin of the
       seismological Society of America, 95(6), 2081-2092.

    .. [2] Ludwig, W. J. (1970). Seismic refraction. The sea, 4, 53-84.
    """
    # Convert [m/s] -> [km/s]
    a1 = vp * M2KM
    a2 = a1 * a1
    a3 = a2 * a1
    a4 = a3 * a1
    a5 = a4 * a1
    rho = 1.6612*a1 - 0.4721*a2 + 0.0671*a3 - 0.0043*a4 + 0.000106*a5

    # Convert [g/cm^3] -> [kg/m^3]
    rho *= 1000.
    return rho


def qp_from_infqk(qs, vp, vs):
    """
    Calculate Qp from Qs, Vp and Vs assuming a purly compressive or
    dilational processes (i.e. Bulk quality factor, Qk, is infinity) by
    using equation (14.41) of Udias (1999):

        (Qp/Qs) = (3/4) * (Vp/Vs)^2

    Parameters
    ----------
    qs : float
        S-wave quality factor.
    vp : float
        P-wave velocity in [m/s].
    vs : float
        S-wave velocity in [m/s].

    Returns
    -------
    qp : float
        P-wave quality factor.
    """
    return 0.75 * (vp/vs)**2 * qs


def qs_from_infqk(qp, vp, vs):
    """
    Calculate Qs from Qp, Vp and Vs assuming a purly compressive or
    dilational processes (i.e. Bulk quality factor, Qk, is infinity) by
    using equation (14.41) of Udias (1999):

        (Qp/Qs) = (3/4) * (Vp/Vs)^2

    Parameters
    ----------
    qp : float
        P-wave quality factor.
    vp : float
        P-wave velocity in [m/s].
    vs : float
        S-wave velocity in [m/s].

    Returns
    -------
    qs : float
        S-wave quality factor.
    """
    return qp / (0.75 * (vp/vs)**2)


def _read_minimum1d_model_fh(f, zbot=None, qp=None, qs=None):
    from_scanlines = []
    mabove = None
    for line in f:
        z, vp, vs = map(lambda x: float(x)*KM2M, line.split())
        rho = empirical_density(vp)

        if qp and not qs:
            qs = qs_from_infqk(qp, vp, vs)
        elif qs and not qp:
            qp = qp_from_infqk(qs, vp, vs)

        if mabove:
            from_scanlines.append((z, mabove, None))

        m = cake.Material(vp=vp, vs=vs, rho=rho, qp=qp, qs=qs)
        from_scanlines.append((z, m, None))
        mabove = m

    if zbot:
        from_scanlines.append((zbot, mabove, None))

    f.close()

    for x in from_scanlines:
        yield x


def read_minimum1d_model(fn, zbot=None, qp=None, qs=None):
    """
    Reader for minimum 1-D model style files containing multiple lines of
    three columns data, that are depth (in km), P-wave velocity (in km/s)
    and S-wave velocity (in km/s), respectively.

    This reader can be used as producer in
    :py:meth:`LayeredModel.from_scanlines`.

    Parameters
    ----------
    fn : str
        Path/filename of the velocity model.
    zbot : float (optional)
        Maximum depth to append to the model. Unit: m.
    qp : float (optional)
        Fixed value for P-wave quality factor.
    qs : float (optional)
        Fixed value for S-wave quality factor.

    Yields
    ------
    3-tuple of (depth, material, name).
    """
    with open(fn, 'r') as f:
        for x in _read_minimum1d_model_fh(f, zbot=zbot, qp=qp, qs=qs):
            yield x
