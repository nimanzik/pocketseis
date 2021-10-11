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

**Waveform Rotation Matrices (2-D and 3-D)**
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


def rotmat_cartesian(theta, axis):
    """
    Elemental rotation through angle `theta` about `axis`-axis of a
    Cartesian coordinate system.

    Parameters
    ----------
    theta : float or array-like
        Angle(s) of rotation in **radians**.
    axis : {'x', 'y', 'z'}
        One of the axis of the Cartesian coordinate system, through wich
        the rotation is performed.

    Returns
    -------
    rotmat : ndarray, shape (3, 3) or (n_angles, 3, 3) where n_angles >= 2
        Rotation matrix that rotates vectors/matrices by angle(s) of
        `theta` about the given `axis`.
    """
    if not (axis := axis.lower()) in (valid_axes := ['x', 'y', 'z']):
        raise ValueError(f"Valid axis values are {valid_axes}")

    theta = np.asarray(theta, dtype=np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    if axis == 'x':
        rotmat = np.array([[1.0, 0.0, 0.0],
                           [0.0, +cos_t, -sin_t],
                           [0.0, +sin_t, +cos_t]])
    elif axis == 'y':
        rotmat = np.array([[+cos_t, 0.0, +sin_t],
                           [0.0, 1.0, 0.0],
                           [-sin_t, 0.0, cos_t]])
    elif axis == 'z':
        rotmat = np.array([[+cos_t, -sin_t, 0.0],
                           [+sin_t, +cos_t, 0.0],
                           [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"Invalid axis value {axis}")

    if rotmat.ndim == 3:
        # Reshape to (n_angles, 3, 3)
        rotmat = np.moveaxis(rotmat, 2, 0)

    return np.squeeze(rotmat)


def rotate_mt(m, conv_from, conv_to):
    """
    Convert (i.e. rotate) seismic moment tensor from one Cartesian
    coordinate system into another.

    Parameters
    ----------
    m : ndarray, shape (3, 3)
        Moment tensor as symmetric 3-by-3 matrix represented in the 1st
        (original) coordinate frame.
    conv_from : {'NED', 'NWU', 'ENU'}
        Original coordinate system.
    conv_to : {'NED', 'NWU', 'ENU'}
        Primed coordinate system.

    Returns
    -------
    m_prime : ndarray, shape (3, 3)
        Converted (i.e. rotated) moment tensor defined in the 2nd (primed)
        coordinate frame.
    """
    m = np.asarray(m, dtype=np.float64)
    conv_from = conv_from.upper()
    conv_to = conv_to.upper()
    err_msg = f"Conversion is not supported: {conv_from} -> {conv_to}"

    if conv_from == 'NED':
        if conv_to == 'NWU':
            rotmat = rotmat_cartesian(np.pi, 'x')
        elif conv_to == 'ENU':
            rotmat1 = rotmat_cartesian(np.pi, 'x')
            rotmat2 = rotmat_cartesian(-np.pi / 2.0, 'z')
            rotmat = np.dot(rotmat1, rotmat2)
        else:
            raise ValueError(err_msg)

    elif conv_from == 'NWU':
        if conv_to == 'NED':
            rotmat = rotmat_cartesian(-np.pi, 'x')
        else:
            raise ValueError(err_msg)

    elif conv_from == 'ENU':
        if conv_to == 'NED':
            rotmat1 = rotmat_cartesian(np.pi / 2.0, 'z')
            rotmat2 = rotmat_cartesian(-np.pi, 'x')
            rotmat = np.dot(rotmat1, rotmat2)
        else:
            raise ValueError(err_msg)

    else:
        raise ValueError(err_msg)

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

    bazi = np.deg2rad(np.asarray(bazi, dtype=np.float64))
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
    bazi = np.deg2rad(np.asarray(bazi, dtype=np.float64))
    incid = np.deg2rad(np.asarray(incid, dtype=np.float64))

    sb = np.sin(bazi)
    cb = np.cos(bazi)
    si = np.sin(incid)
    ci = np.cos(incid)

    rotmat_3d = np.array([[ci, si, 0.],
                          [-si * cb, ci * cb, sb],
                          [-si * sb, ci * sb, -cb]])
    if rotmat_3d.ndim == 3:
        # Reshape to (n_bazi, 3, 3) or (n_incid, 3, 3)
        rotmat_3d = np.moveaxis(rotmat_3d, 2, 0)

    return np.squeeze(rotmat_3d)


__all__ = """
    rotmat_cartesian
    rotate_mt
    rotmat_ne2rt
    rotmat_zne2lqt
""".split()
