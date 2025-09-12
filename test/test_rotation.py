import unittest
from functools import partial

import numpy as np

from pocketseis import rotation as rot


class RotationTestCase(unittest.TestCase):
    def test_cartesian_rotmat(self):
        theta = np.pi / 2.0
        Rx = rot.cartesian_rotmat(theta, "x")
        Ry = rot.cartesian_rotmat(theta, "y")
        Rz = rot.cartesian_rotmat(theta, "z")

        e1, e2, e3 = np.eye(3, dtype=np.float64)

        aac = partial(np.testing.assert_allclose, rtol=0.0, atol=1.0e-4)
        aac(Rz @ e1, e2)
        aac(Rx @ e2, e3)
        aac(Ry @ e3, e1)


if __name__ == "__main__":
    unittest.main()
