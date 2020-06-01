"""
Module containing functions and classes related to seismic moment tensor.
"""

import numpy as np


class ElementalRotations3D(object):
    """
    Basic (elemental) rotation that is rotation about one of the axes of a
    right-handed coordinate system through a given angle.
    """
    @classmethod
    def about_x(cls, a):
        """
        Elemental rotation through angle `a` about x-axis of a Cartesian
        coordinate system.

        Parameters
        ----------
        a : float
            Angle of rotation in [rad].

        Returns
        -------
        Rx : ndarray, shape (3, 3)
            Rotation matrix that rotates vectors by an angle of `a` about
            x-axis.
        """
        ca = np.cos(a)
        sa = np.sin(a)
        return np.array([
            [1., 0., 0.],
            [0., ca, -sa],
            [0., sa, ca]], dtype=np.float)

    @classmethod
    def about_y(cls, a):
        """
        Elemental rotation through angle `a` about y-axis of a Cartesian
        coordinate system.

        Parameters
        ----------
        a : float
            Angle of rotation in [rad].

        Returns
        -------
        Ry : ndarray, shape (3, 3)
            Rotation matrix that rotates vectors by an angle of `a` about
            y-axis.
        """
        ca = np.cos(a)
        sa = np.sin(a)
        return np.array([
            [ca, 0., sa],
            [0., 1., 0.],
            [-sa, 0., ca]], dtype=np.float)

    @classmethod
    def about_z(cls, a):
        """
        Elemental rotation through angle `a` about z-axis of Cartesian
        coordinate system.

        Parameters
        ----------
        a : float
            Angle of rotation in [rad].

        Returns
        -------
        Rz : ndarray, shape (3, 3)
            Rotation matrix that rotates vectors by an angle of `a` about
            z-axis.
        """
        ca = np.cos(a)
        sa = np.sin(a)
        return np.array([
            [ca, -sa, 0.],
            [sa, ca, 0.],
            [0., 0., 1.]], dtype=np.float)


class MomentTensorRotation(ElementalRotations3D):
    """
    Convert moment tensor from/to commonly used coordinate system
    conventions.

    Coordinate systems supported are:
        * north-east-down (NED)
        * north-west-up (NWU)
        * east-north-up (ENU)

    All methods get plain moment tensor as symmetric 3-by-3 matrix in the 1st
    (original) coordinate system and returns the converted moment tensor
    defined in the 2nd (rotated) system.
    """
    @classmethod
    def from_ned_to_nwu(cls, m):
        rotmat = super().about_x(np.pi)
        return np.linalg.multi_dot((rotmat.T, m, rotmat))

    @classmethod
    def from_nwu_to_ned(cls, m):
        return cls.from_ned_to_nwu(m)

    @classmethod
    def from_ned_to_enu(cls, m):
        rotmat1 = super().about_x(np.pi)
        rotmat2 = super().about_z(-np.pi/2.0)
        rotmat = np.dot(rotmat1, rotmat2)
        return np.linalg.multi_dot((rotmat.T, m, rotmat))

    @classmethod
    def from_enu_to_ned(cls, m):
        rotmat1 = super().about_z(np.pi/2.0)
        rotmat2 = super().about_x(-np.pi)
        rotmat = np.dot(rotmat1, rotmat2)
        return np.linalg.multi_dot((rotmat.T, m, rotmat))


__all__ = """
    ElementalRotations3D
    MomentTensorRotation
""".split()
