import unittest

import numpy as np

from pocketseis import mtensor


class MTensorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng()
        cls.a = rng.random(size=6)

        b = rng.random(size=(3, 3), dtype=np.float64)
        cls.m = (b + b.T) / 2.0

    def test_tuple6_to_symmat(self):
        b = mtensor.tuple6_to_symmat(tuple(self.a))
        np.testing.assert_allclose(b, b.T)

    def test_normalize_mt(self):
        m_ = mtensor.normalize_mt(self.m)
        self.assertAlmostEqual(np.linalg.norm(m_, ord="fro"), 1.0)


if __name__ == "__main__":
    unittest.main()
