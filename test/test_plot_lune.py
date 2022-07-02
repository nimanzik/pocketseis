import unittest

import numpy as np

from pocketseis.moment_tensor import tuple6_to_symmat
from pocketseis.plot.lune import project


class LuneTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Following sample calculation is taken from Appendix A of
        # Tape & Tape (2015)
        cls.m = tuple6_to_symmat((196, 455, -651, -397, -52, 71)) * 1e-3

    def test_project(self):
        γ, δ = project(self.m)
        β = 90.0 - δ
        np.testing.assert_almost_equal(γ, np.rad2deg(-0.113), decimal=2)
        np.testing.assert_almost_equal(β, np.rad2deg(+1.571), decimal=2)


if __name__ == '__main__':
    unittest.main()
