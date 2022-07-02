import unittest

import numpy as np
from numpy import testing

from pocketseis.moment_tensor import \
    tuple6_to_symmat, moment_to_magnitude, magnitude_to_moment, \
    normalize_mt, denormalize_mt, angular_distance, euclidean_distance


class MTensorTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng()
        cls.a_list = list()
        cls.m_list = list()
        for _ in range(10):
            cls.a_list.append(rng.random(size=6))
            m = rng.random(size=(3, 3), dtype=np.float64)
            cls.m_list.append((m + m.T) / 2.0)

    def test_tuple6_to_symmat(self):
        for a in self.a_list:
            b = tuple6_to_symmat(tuple(a))
            testing.assert_allclose(b, b.T)

    def test_magnitude_moment(self):
        for i in range(1, 11):
            mag = float(i)
            testing.assert_almost_equal(
                moment_to_magnitude(magnitude_to_moment(mag)), mag, decimal=6,
                err_msg='Magnitude to moment to magnitude test faild')

    def test_normalize_mt(self):
        for m in self.m_list:
            self.assertAlmostEqual(
                np.linalg.norm(normalize_mt(m), ord='fro'), 1.0)

    def test_denormalize_mt(self):
        for m in self.m_list:
            mhat = normalize_mt(m)
            moment = np.linalg.norm(m, ord='fro') / np.sqrt(2.0)
            testing.assert_allclose(denormalize_mt(mhat, moment), m)

    def test_angular_euclidean_distance(self):
        for m in self.m_list:
            testing.assert_almost_equal(angular_distance(m, m), 0.0)
            testing.assert_almost_equal(euclidean_distance(m, m), 0.0)


if __name__ == '__main__':
    unittest.main()
