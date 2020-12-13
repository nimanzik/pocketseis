"""
Utility functions for seismic moment tensor.
"""

import numpy as np


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

    return np.array([[a11, a12, a13],
                     [a12, a22, a23],
                     [a13, a23, a33]], dtype=np.float)


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
    m = np.asarray(m, dtype=np.float)
    a = []
    for offset in range(3):
        a.extend(m.diagonal(offset=offset).tolist())

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
    return (2./3.) * (np.log10(moment) - 9.1)


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
    return 10**(1.5*mag + 9.1)


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
    m = np.asarray(m, dtype=np.float)
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
    m_norm = np.asarray(m_norm, dtype=np.float)
    if m_norm.ndim != 2 or m_norm.shape != (3, 3):
        raise ValueError("'m_norm' must be an array of shape (3, 3)")

    return np.sqrt(2.) * moment * m_norm


__all__ = """
    tuple6_to_symmat
    symmat_to_tuple6
    moment_to_magnitude
    magnitude_to_moment
    normalize_mt
    denormalize_mt
""".split()
