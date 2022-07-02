import unittest

from pocketseis.compass import ellipsoid_distance


KM2M = 1e+3


class CompassTestCase(unittest.TestCase):

    def test_ellipsoid_distance(self):
        # Berlin lat-lon
        p1 = (52.51666666666667, 13.4)
        # Tokyo lat-lon
        p2 = (35.7, 139.76666666666667)
        self.assertAlmostEqual(
            ellipsoid_distance(*p1, *p2) / KM2M, 8941.20250458698, places=9)


if __name__ == '__main__':
    unittest.main()
