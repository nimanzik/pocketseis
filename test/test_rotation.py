from functools import partial
import unittest

import numpy as np

from pocketseis.rotation import cartesian_rotmat, rotate_mt


class RotationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.aac = partial(np.testing.assert_allclose, rtol=0.0, atol=1.0e-9)

    def test_cartesian_rotmat(self):

        theta = np.pi / 2.0
        Rx = cartesian_rotmat(theta, 'x')
        Ry = cartesian_rotmat(theta, 'y')
        Rz = cartesian_rotmat(theta, 'z')

        e1, e2, e3 = np.eye(3, dtype=np.float64)
        self.aac(Rz @ e1, e2)
        self.aac(Rx @ e2, e3)
        self.aac(Ry @ e3, e1)

    def test_rotate_mt(self):
        m = np.random.default_rng().random(size=(3, 3))
        self.aac(rotate_mt(rotate_mt(m, 'NED->NWU'), 'NWU->NED'), m)
        self.aac(rotate_mt(rotate_mt(m, 'NED->ENU'), 'ENU->NED'), m)


if __name__ == '__main__':
    unittest.main()
