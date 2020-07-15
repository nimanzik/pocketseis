"""
This module provides functions and classes related to 2-D and 3-D rotational
transformation.


**Elemental 3-D Rotation Matrices**
-----------------------------------
Basic (elemental) rotation matrices that can be used to rotate about one of
the axes of a right-handed Cartesian coordinate system through a given angle.

**Moment Tensor Rotation**
--------------------------
Convert moment tensor from/to commonly used coordinate system
conventions. Coordinate systems supported are:
    * north-east-down (NED)
    * north-west-up (NWU)
    * east-north-up (ENU)

All methods get plain moment tensor as symmetric 3-by-3 matrix in the 1st
(original) coordinate system and returns the converted moment tensor
defined in the 2nd (rotated) system.

**Wavefor Rotation Matrices (2-D and 3-D)**
-------------------------------------------
2-D and 3-D rotation matrices that can be used to rotate seismic waveform
data into specified coordinate systems.

Notes
-----
The convention used to write the rotation matrices:
*Columns* of these matrices form unit basis vactors of the rotated coordinate
system (primed system). In other words, they are basis vactors of the primed
system written in original coordinate system.

When discussing a rotation, there are two possible conventions: rotation of
the axes (passive transformation), and rotation of the object relative to
fixed axes (active transformation). In the physical sciences, an active
transformation is one which actually changes the physical position of a
system, and makes sense even in the absence of a coordinate system whereas a
passive transformation is a change in the coordinate description of the
physical system (change of basis). The distinction between active and passive
transformations is important.

---
**Passive transformation:**

Transformation (here, rotation) law relating components of the *same*
vector with respect to different Cartesian unit bases is::

    ``u = np.matmul(rotmat.T, v)``,

where, ``v`` denotes the vactor with respect to the unprimed (original)
basis, and ``u`` denotes the *same* vector with respect to the primed
(rotated) basis.

Transformation (here, rotation) law that relates components of the *same*
tensor with respect to different Cartesian unit bases is::

    ``U = np.linalg.multi_dot((rotmat.T, V, rotmat))``,

where, ``V`` and ``U`` are different matrices of the *same* tensor with
respect to original and rotated coordinate frames, respectively.

The `np.matmul` function implements the semantics of the @ operator introduced
in Python 3.5 following PEP465.

---
**Active transformation:**

Components of a transformed (rotated) vector, ``b``, is computed from the
components of original vector, ``a``, and transformation (here rotation)
matrix as::

    ``b = np.matmul(rotmat, a).

Note that in this case the coordinate frame is fixed and a linear
transformation is applied to vector ``a``. Therefore, both vectors ``a``
and ``b`` are in the *same* coordinate system. Components of a transformed
tensor ``B`` is computed from the components of original tensor ``A`` as::

    ``B = np.linalg.multi_dot((rotmat, A, rotmat.T))``

References
----------
.. [1] https://en.wikipedia.org/wiki/Rotation_matrix
.. [2] https://en.wikipedia.org/wiki/Active_and_passive_transformation
"""

import numpy as np


def rotmat_about_x(a):
    """
    Elemental rotation through angle `a` about x-axis of a Cartesian
    coordinate system.

    Parameters
    ----------
    a : float or array-like
        Angle of rotation in **radians**.

    Returns
    -------
    rotmat_x : ndarray, shape (3, 3) or (n_angles, 3, 3) where n_angles >= 2
        Rotation matrix that rotates vectors/matrices by angle of `a`
        about x-axis.
    """
    a = np.asarray(a, dtype=np.float)
    ca = np.cos(a)
    sa = np.sin(a)
    rotmat_x = np.array([[1., 0., 0.],
                         [0., ca, -sa],
                         [0., sa, ca]])
    if rotmat_x.ndim == 3:
        # Reshape to (n_angles, 3, 3)
        rotmat_x = np.moveaxis(rotmat_x, 2, 0)

    return np.squeeze(rotmat_x)


def rotmat_about_y(a):
    """
    Elemental rotation through angle `a` about y-axis of a Cartesian
    coordinate system.

    Parameters
    ----------
    a : float or array-like
        Angle of rotation in **radians**.

    Returns
    -------
    rotmat_y : ndarray, shape (3, 3) or (n_angles, 3, 3) where n_angles >= 2
        Rotation matrix that rotates vectors/matrices by angle of `a`
        about y-axis.
    """
    a = np.asarray(a, dtype=np.float)
    ca = np.cos(a)
    sa = np.sin(a)
    rotmat_y = np.array([[ca, 0., sa],
                         [0., 1., 0.],
                         [-sa, 0., ca]])
    if rotmat_y.ndim == 3:
        # Reshape to (n_angles, 3, 3)
        rotmat_y = np.moveaxis(rotmat_y, 2, 0)

    return np.squeeze(rotmat_y)


def rotmat_about_z(a):
    """
    Elemental rotation through angle `a` about z-axis of Cartesian
    coordinate system.

    Parameters
    ----------
    a : float or array-like
        Angle of rotation in **radians**.

    Returns
    -------
    rotmat_z : ndarray, shape (3, 3) or (n_angles, 3, 3) where n_angles >= 2
        Rotation matrix that rotates vectors/matrices by angle of `a`
        about z-axis.
    """
    a = np.asarray(a, dtype=np.float)
    ca = np.cos(a)
    sa = np.sin(a)
    rotmat_z = np.array([[ca, -sa, 0.],
                         [sa, ca, 0.],
                         [0., 0., 1.]])
    if rotmat_z.ndim == 3:
        # Reshape to (n_angles, 3, 3)
        rotmat_z = np.moveaxis(rotmat_z, 2, 0)

    return np.squeeze(rotmat_z)


def rotate_mt_ned2nwu(m):
    """
    Convert (i.e. rotate) seismic moment tensor from North-East-Down (ned)
    into North-West-Up (nwu) coordinate system.

    Parameters
    ----------
    m : ndarray, shape (3, 3)
        Moment tensor as symmetric 3-by-3 matrix represented in the 1st
        (original) coordinate frame.

    Returns
    -------
    m_prime : ndarray, shape (3, 3)
        Converted (i.e. rotated) moment tensor defined in the 2nd (primed)
        coordinate frame.
    """
    m = np.asarray(m, dtype=np.float)
    rotmat = rotmat_about_x(np.pi)
    return np.linalg.multi_dot((rotmat.T, m, rotmat))


def rotate_mt_nwu2ned(m):
    """
    Convert (i.e. rotate) seismic moment tensor from North-West-Up (nwu)
    into North-East-Down (ned) coordinate system.

    Parameters
    ----------
    m : ndarray, shape (3, 3)
        Moment tensor as symmetric 3-by-3 matrix represented in the 1st
        (original) coordinate frame.

    Returns
    -------
    m_prime : ndarray, shape (3, 3)
        Converted (i.e. rotated) moment tensor defined in the 2nd (primed)
        coordinate frame.
    """
    m = np.asarray(m, dtype=np.float)
    rotmat = rotmat_about_x(-np.pi)
    return np.linalg.multi_dot((rotmat.T, m, rotmat))


def rotate_mt_ned2enu(m):
    """
    Convert (i.e. rotate) seismic moment tensor from North-East-Down (ned)
    into East-North-Up (enu) coordinate system.

    Parameters
    ----------
    m : ndarray, shape (3, 3)
        Moment tensor as symmetric 3-by-3 matrix represented in the 1st
        (original) coordinate frame.

    Returns
    -------
    m_prime : ndarray, shape (3, 3)
        Converted (i.e. rotated) moment tensor defined in the 2nd (primed)
        coordinate frame.
    """
    m = np.asarray(m, dtype=np.float)
    rotmat1 = rotmat_about_x(np.pi)
    rotmat2 = rotmat_about_z(-np.pi/2.)
    rotmat = np.dot(rotmat1, rotmat2)
    return np.linalg.multi_dot((rotmat.T, m, rotmat))


def rotate_mt_enu2ned(m):
    """
    Convert (i.e. rotate) seismic moment tensor from East-North-Up (enu)
    into North-East-Down (ned) coordinate system.

    Parameters
    ----------
    m : ndarray, shape (3, 3)
        Moment tensor as symmetric 3-by-3 matrix represented in the 1st
        (original) coordinate frame.

    Returns
    -------
    m_prime : ndarray, shape (3, 3)
        Converted (i.e. rotated) moment tensor defined in the 2nd (primed)
        coordinate frame.
    """
    m = np.asarray(m, dtype=np.float)
    rotmat1 = rotmat_about_z(np.pi/2.)
    rotmat2 = rotmat_about_x(-np.pi)
    rotmat = np.dot(rotmat1, rotmat2)
    return np.linalg.multi_dot((rotmat.T, m, rotmat))


def rotmat_ne2rt(bazi):
    """
    2-D rotation matrix of horizontal components of a seismogram.

    The rotation is about ``Z`` (vertical) component, so the ``Z``
    component remains unchanged and ``NE`` (North, East) components are
    rotated into ``RT`` (Radial, Transverse).
    Both coordinate systems (``ZNE`` and ``ZRT``) are left-handed.

    Parameters
    ----------
    bazi : float or array-like
        Backazimuth in **degrees**. This is the angle measured clockwise
        from the North and is defined as the angle between vector pointing
        from the station to the North and the vector pointing from the
        station to the source.

    Returns
    -------
    rotmat_2d : ndarray, shape (2, 2) or (n_bazi, 2, 2) where n_bazi >= 2
        Rotation matrix.

    Notes
    -----
    To rotate ``NE`` components into ``RT``, one should use::

        ``np.matmul(rotmat_2d.T, ne)``,

    where ``ne`` is an ndarray of shape ``(2, n_samples)``, whose rows are
    ``N`` and ``E`` components of the seismogram, respectively.
    """

    bazi = np.deg2rad(np.asarray(bazi, dtype=np.float))
    sb = np.sin(bazi)
    cb = np.cos(bazi)

    rotmat_2d = np.array([[-cb, sb],
                          [-sb, -cb]])

    if np.ndim(rotmat_2d) == 3:
        # Reshape to (n_bazi, 2, 2)
        rotmat_2d = np.moveaxis(rotmat_2d, 2, 0)

    return np.squeeze(rotmat_2d)


def rotmat_zne2lqt(bazi, incid):
    """
    3-D rotation of all three components of a seismogram from ``ZNE``
    (Vertical, North, East; left-handed) system into ``LQT`` (P-wave
    propagation, SV direction, SH direction; ray coordinate system,
    right-handed).

    Parameters
    ----------
    bazi : float or array-like
        Backazimuth in **degrees**. This is the angle measured clockwise
        from the North and is defined as the angle between vector pointing
        from the station to the North and the vector pointing from the
        station to the source.

    incid : float or array-like
        Angle of incidence in **degrees**. This is the angle from vertical
        at which an incoming ray arrives. For example, a ray arriving from
        directly below the station would have an angle of incidence of
        zero degrees.

    Returns
    -------
    rmat_3d : ndarray, shape (3, 3) or (n, 3, 3) where n >= 2
        Rotation matrix.

    Notes
    -----
    *Columns* of the 3-D rotation matrix are unit bases of the new
    coordinate system ``LQT``.

    To rotate from ``ZNE`` to ``LQT``, one should use::

        ``np.matmul(rmat_3d.T, zne)``

    where ``zne`` is an ndarray of shape ``(3, n_samples)``, whose rows are
    ``Z``, ``N`` and ``E`` components of the seismogram, respectively.
    """
    bazi = np.deg2rad(np.asarray(bazi, dtype=np.float))
    incid = np.deg2rad(np.asarray(incid, dtype=np.float))

    sb = np.sin(bazi)
    cb = np.cos(bazi)
    si = np.sin(incid)
    ci = np.cos(incid)

    rotmat_3d = np.array([[ci, si, 0.],
                          [-si*cb, ci*cb, sb],
                          [-si*sb, ci*sb, -cb]])
    if rotmat_3d.ndim == 3:
        # Reshape to (n_bazi, 3, 3) or (n_incid, 3, 3)
        rotmat_3d = np.moveaxis(rotmat_3d, 2, 0)

    return np.squeeze(rotmat_3d)


__all__ = """
    rotmat_about_x
    rotmat_about_y
    rotmat_about_z
    rotate_mt_ned2nwu
    rotate_mt_ned2enu
    rotate_mt_enu2ned
    rotate_mt_nwu2ned
    rotmat_ne2rt
    rotmat_zne2lqt
""".split()
