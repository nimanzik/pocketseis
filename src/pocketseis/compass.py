import numpy as np

from pyrocko import orthodrome as od


def calc_relative_data(lat0, lon0, depth0, lats, lons, depths):
    """
    Calculates the 3-D distances and directional cosines of a set of
    observation points (potential receivers) from a reference location
    (potential seismic source) in a Cartesian coordinate system defined
    as (x, y, z)=(North, East, Down).

    Parameters
    ----------
    lat0, lon0, depth0: float
        Geographical coordinates of the reference point.
    lats, lons, depths : list of floats
        Geographical coordinates of the observation points.

    Returns
    -------
    dists_3d : ndarray of shape (n_receivers,)
        3-D distances of observation points from event. Unit: [m]
    cosine_vecs : ndarray of shape (3, n_receivers)
        Unit vectors of direction cosines. Relative event-receiver
        position vectors are in Cartesian coordinate system (i.e.
        vectors whose initial and terminal are event and observation
        points, respectively). The indexes of the first dimension
        represent (0, 1, 2)->(North, East, Down) axes.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)

    deltazs = depths - depth0
    deltaxs, deltays = od.latlon_to_ne_numpy(lat0, lon0, lats, lons)

    # Position vectors relative to reference loc., array of shape (3, n_points)
    pos_vecs = np.vstack([deltaxs, deltays, deltazs])
    assert pos_vecs.shape == (3, lats.size)

    # 3-D distances from reference loc., array of shape (n_points,)
    dists_3d = np.sqrt(np.sum(pos_vecs**2, axis=0, keepdims=False))

    # Unit vectors of direction cosines
    cosine_vecs = pos_vecs / dists_3d

    return (dists_3d, cosine_vecs)


class WGS84(object):
    """
    Earth reference ellipsoid parameters for WGS84 geodetic system.
    """

    def __init__(self):
        self.__a = 6378137.0
        self.__f = 1.0 / 298.257223563

    @property
    def a(self):
        """
        Semi-major axis in [m]
        """
        return self.__a

    @property
    def f(self):
        """
        Earth flattening factor
        """
        return self.__f

    @property
    def b(self):
        """
        Semi-minor axis in [m]
        """
        return self.__a * (1.0 - self.__f)

    @property
    def e2(self):
        """
        First eccentricity squared
        """
        return self.__f * (2.0 - self.__f)

    @property
    def eprime2(self):
        """
        Second eccentricity squared
        """
        return self.__f * (2.0 - self.__f) / (1.0 - self.__f)**2


def ellipsoid_distance(lat1, lon1, lat2, lon2):
    """
    Approximate ellipsoid distance between two points using Vincenty's
    formulae. Although it is called approximate, it is actually much
    more accurate than the great circle calculation.

    Parameters
    ----------
    lat1, lon1: float
        Latitude and longitude of the first point in [deg].
    lat2, lon2: float
        Latitude and longitude of the second point in [deg].

    Returns
    -------
    d : float
        Distance between points a and b in [m].

    References
    ----------
    .. [1] https://www.codeguru.com/cplusplus/geographic-distance-and-azimuth-calculations/   # noqa
    .. [2] Jean Meeus (1998), Astronomical Algorithms, 2nd ed.
    """
    wgs = WGS84()
    a = wgs.a
    f = wgs.f

    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

    F = (lat1+lat2) * 0.5
    G = (lat1-lat2) * 0.5
    L = (lon1-lon2) * 0.5

    sinF, cosF = np.sin(F), np.cos(F)
    sinG, cosG = np.sin(G), np.cos(G)
    sinL, cosL = np.sin(L), np.cos(L)

    S = (sinG**2 * cosL**2) + (cosF**2 * sinL**2)
    C = (cosG**2 * cosL**2) + (sinF**2 * sinL**2)
    W = np.arctan2(np.sqrt(S), np.sqrt(C))

    if W == 0.:
        return 0.0

    R = np.sqrt(S * C) / W
    H1 = (3.0 * R - 1.0) / (2.0 * C)
    H2 = (3.0 * R + 1.0) / (2.0 * S)
    d = 2.0 * W * a * (
        1.0 + (f * H1 * sinF**2 * cosG**2) - (f * H2 * cosF**2 * sinG**2))

    return d


__all__ = ['calc_relative_data', 'WGS84', 'ellipsoid_distance']
