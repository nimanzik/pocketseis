from functools import partial
import unittest

import numpy as np
from pyrocko.orthodrome import distance_accurate50m_numpy

from pocketseis.compass import compute_relative_data, ellipsoid_distance


KM2M = 1e+3
M2KM = 1.0 / KM2M


class CompassTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.aac = partial(np.testing.assert_allclose, rtol=0.0, atol=1e-9)

    def test_calc_relative_data(self):
        p0 = (0.0, 0.0, 0.0)

        step_size = 0.8
        points = np.eye(3) * step_size

        # Euclidean distances & directional cosines
        d, c = compute_relative_data(*p0, *points)

        self.aac(c, np.eye(3))
        self.aac(d[:2], distance_accurate50m_numpy(*p0[:2], *points[:2, :2]))
        self.aac(d[-1], step_size)

    def test_ellipsoid_distance(self):
        p1 = (52.51666666666667, 13.4)   # Berlin lat-lon
        p2 = (35.7, 139.76666666666667)   # Tokyo lat-lon
        self.assertAlmostEqual(
            ellipsoid_distance(*p1, *p2) * M2KM,
            8941.20250458698,
            places=9)


if __name__ == '__main__':
    unittest.main()
