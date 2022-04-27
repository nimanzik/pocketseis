"""
Utility functions for seismic moment tensor.
"""

import numpy as np

from pyrocko import moment_tensor as pmt


def tuple6_to_symmat(a):
    """
    Create symmetric 3-by-3 moment-tensor matrix from its six
    independent components.

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

    return np.array([[a11, a12, a13],
                     [a12, a22, a23],
                     [a13, a23, a33]], dtype=np.float64)


def symmat_to_tuple6(m):
    """
    Get non-redundant components of symmetric 3-by-3 moment-tensor matrix.

    m : ndarray, shape (3, 3)
        Plain seismic moment tensor as symmetric 2-D array.

    Returns
    -------
    a : tuple of 6 float
        A tuple of six independent components of moment tensor in
        (M11, M22, M33, M12, M13, M23) order.
    """
    m = np.asarray(m, dtype=np.float64)
    assert m.shape == (3, 3), 'moment tensor is a symmetric 3-by-3 array'
    a = tuple(m.diagonal()) + (m[0, 1], m[0, 2], m[1, 2])

    return tuple(a)


def moment_to_magnitude(moment):
    """
    Converts scalar moment, $M_0$, to moment magnitude, $M_w$ using
    eq. 9.73, Shearer (2009)::

        $M_w = \frac{2}{3} [log_10 M_0 - 9.1]$,

    where, $M_0$ is scalar moment in Nm.

    Parameters
    ----------
    moment : float
        Scalar moment, $M_0$. Unit: Nm.

    Returns
    -------
    mag : float
        Moment magnitude (Mw).
    """
    return (2.0 / 3.0) * (np.log10(moment) - 9.1)


def magnitude_to_moment(mag):
    """
    Converts moment magnitude, $M_w$, to scalar moment, $M_0$ using
    eq. 9.73, Shearer (2009)::

        $M_w = \frac{2}{3} [log_10 M_0 - 9.1]$,

    where, $M_0$ is scalar moment in Nm.

    Parameters
    ----------
    mag : float
        Moment magnitude, $M_w$.

    Returns
    -------
    moment : float
        Scalar moment, , $M_0$. Unit: Nm.
    """
    return 10**(1.5 * mag + 9.1)


def normalize_mt(m):
    """
    Unit-norm moment tensor.
    Normalizes moment tensor by its Euclidean (Frobenius) norm.

    Parameters
    ----------
    m : ndarray, shape (3, 3)
        Plain seismic moment tensor as symmetric 2-D array.

    Returns
    -------
    m_norm : ndarray, shape (3, 3)
        Normalized moment tensor.
    """
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2 or m.shape != (3, 3):
        raise ValueError("'m' must be an array of shape (3, 3)")

    m_norm = m / np.linalg.norm(m, ord='fro')
    return m_norm


def denormalize_mt(m_norm, moment):
    """
    Construct norm-preserving moment tensor from a unit-norm moment
    tensor and its total moment following Silver and Jordan (1982) and
    using eq. 9.8, Shearer (2009).

    Parameters
    ----------
    m_norm : ndarray, shape (3, 3)
        Unit-norm moment tensor as plain symmetric 2-D array.

    moment : float
        Scalar moment, $M_0$. Unit: Nm.

    Returns
    -------
    m : ndarray, shape (3, 3)
        Norm-preserved moment tensor (denormalized), i.e. the size of the
        seismic event applied.

    References
    ----------
    .. [1] Silver, P. G., & Jordan, T. H. (1982). Optimal estimation of
       scalar seismic moment. Geophysical Journal International, 70(3),
       755-787.
    .. [2] Shearer, P. M. (2019). Introduction to seismology. Cambridge
       university press.
    """
    m_norm = np.asarray(m_norm, dtype=np.float64)
    if m_norm.ndim != 2 or m_norm.shape != (3, 3):
        raise ValueError("'m_norm' must be an array of shape (3, 3)")

    return np.sqrt(2.0) * moment * m_norm


def angular_distance(m1, m2):
    """
    Angular distance between two moment tensors given by eq. 2b of
    Tape & Tape (2019, GJI).

    Parameters
    ----------
    m1, m2 : ndarray, shape (3, 3)
        Plain seismic moment tensors as symmetric 2-D arrays.

    Returns
    -------
    chi : float
        Angular distance between two moment tensors `m1` and `m2`.
    """
    m1m2_inner = np.trace(np.dot(m1.T, m2))
    m1_norm = np.linalg.norm(m1, ord='fro')
    m2_norm = np.linalg.norm(m2, ord='fro')
    chi = np.arccos(m1m2_inner / (m1_norm * m2_norm))
    return chi


def euclidean_distance(m1, m2):
    """
    Euclidean distance between two moment tensors given by eq. 3 of
    Tape & Tape (2019, GJI). Distances calculated by using this equation
    range between 0 and 1, that correspond to identical and opposite
    seismic radiation patterns between the two compared moment tensors,
    respectively.

    Parameters
    ----------
    m1, m2 : ndarray, shape (3, 3)
        Plain seismic moment tensors as symmetric 2-D arrays.

    Returns
    -------
    d : float
        Euclidean distance between two moment tensors `m1` and `m2`.
    """
    d = np.sin(angular_distance(m1, m2) / 2.0)
    return d


def random_mt(n_src=1, method='S5', seed=None):
    """
    Generate uniform distribution of unit-norm moment tensors.

    Parameters
    ----------
    n_src : int, optional
        Number of randomly generated moment-tensor sources (default is 1).
    seed : int, optional
        A seed to initialise the random generator (default is None).
    method : {'S5', 'R6', 'QT'}
        Different ways to generate uniform distributions of unit moment
        tensors:
        - 'S5': Uniform sampling of unit 5-D hyper-sphere using the
          method of Tashiro (1977).
        - 'R6': Uniform sampling of unit 5-D hyper-sphere using the
          method of Tape & Tape (2015, GJI).
        - 'QT' : Uniform sampling of Q-T coordinate block proposed by
          Tape & Tape (2015, GJI).
        Default is 'S5'.

    Returns
    -------
    out : list of `pyrocko.moment_tensor.MomentTensor` objects
        Unit-norm moment tensors in North-East-Down (NED) coordinate
        system.
    """
    if not isinstance(method, str):
        raise TypeError("expects str for method")

    if (method := method.upper()) not in (valid_methods := ['S5', 'R6', 'QT']):
        raise ValueError(f"valid methods are {valid_methods}")

    if (seed is not None) and (not isinstance(seed, int)):
        raise TypeError("expects int for seed")

    rng = np.random.default_rng(seed=seed)
    two_pi = 2.0 * np.pi
    sqrt2 = np.sqrt(2.0)

    # Eq. 6 of Staehler & Sigloch (2014, SE)
    if method == 'S5':
        xs = np.zeros((5, n_src), dtype=np.float64)
        for i_row in range(5):
            xs[i_row] = rng.uniform(low=0.0, high=1.0, size=n_src)

        x1, x2, x3, x4, x5 = xs

        y3 = 1.0
        y2 = np.sqrt(x2)
        y1 = y2 * x1

        c1 = np.sqrt(y1)
        c2 = np.sqrt(y2 - y1)
        c3 = np.sqrt(y3 - y2)

        a_xx = c1 * np.cos(two_pi * x3)
        a_yy = c1 * np.sin(two_pi * x3)
        a_zz = c2 * np.cos(two_pi * x4)
        a_xy = c2 * np.sin(two_pi * x4) / sqrt2
        a_yz = c3 * np.cos(two_pi * x5) / sqrt2
        a_xz = c3 * np.sin(two_pi * x5) / sqrt2

        mt_list = []
        for a_tuple in zip(a_xx, a_yy, a_zz, a_xy, a_xz, a_yz):
            m = pmt.symmat6(*a_tuple)
            mt_list.append(pmt.MomentTensor(m=m))

    # Method 1 of section 9 of Tape & Tape (2015, GJI)
    elif method == 'R6':
        vs = np.zeros((6, n_src), dtype=np.float64)
        for i_row in range(6):
            vs[i_row] = rng.normal(loc=0.0, scale=1.0, size=n_src)

        vhats = vs / np.linalg.norm(vs, axis=0, keepdims=True)
        vhats[3:] /= sqrt2

        mt_list = []
        for a_tuple in vhats.T:
            m = pmt.symmat6(*a_tuple)
            mt_list.append(pmt.MomentTensor(m=m))

    # Method 2 of section 9 of Tape & Tape (2015, GJI)
    else:
        # Differ the import of MTQTSource to avoid circular import
        from pocketseis.source import MTQTSource

        us = rng.uniform(low=0.0, high=0.75 * np.pi, size=n_src)
        vs = rng.uniform(low=-1.0 / 3.0, high=1.0 / 3.0, size=n_src)
        κs = rng.uniform(low=0.0, high=2.0 * np.pi, size=n_src)
        σs = rng.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=n_src)
        hs = rng.uniform(low=0.0, high=1.0, size=n_src)

        mt_list = []
        for (u, v, κ, σ, h) in zip(us, vs, κs, σs, hs):
            m = MTQTSource(u=u, v=v, kappa=κ, sigma=σ, h=h).m9_ned
            mt_list.append(pmt.MomentTensor(m=np.asmatrix(m)))

    return mt_list


__all__ = [
    'tuple6_to_symmat',
    'symmat_to_tuple6',
    'moment_to_magnitude',
    'magnitude_to_moment',
    'normalize_mt',
    'denormalize_mt',
    'angular_distance',
    'euclidean_distance']
