import unittest

import numpy as np
from scipy.integrate import simps, cumtrapz

import gf_util


class GFUtilTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.deltat = 0.01
        cls.tref = 10.2

        factors = [10, 20, 41, 164]
        cls.durations = [factor*cls.deltat for factor in factors]

    def assert_ac(self, actual, desired):
        np.testing.assert_allclose(actual, desired, rtol=0.0, atol=1e-4)

    def test_stf_smooth_ramp(self):

        stf = gf_util.SmoothRampSTF(duration=1*self.deltat)
        t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
        self.assertEqual(a.max(), self.deltat)
        self.assert_ac(
            simps(np.diff(a, prepend=0.0)/self.deltat, x=t), 0.5*self.deltat)

        for duration in self.durations:
            stf = gf_util.SmoothRampSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assertEqual(a.max(), self.deltat)
            self.assert_ac(
                simps(np.diff(a, prepend=0.0)/self.deltat, x=t), self.deltat)

    def test_stf_gaussian(self):

        for duration in self.durations:
            stf = gf_util.GaussianSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assert_ac(simps(a, x=t), self.deltat)

    def test_stf_gaussian_derivative(self):

        for duration in self.durations:
            stf = gf_util.GaussianDerivativeSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assert_ac(simps(a, x=t), 0.0)
            self.assert_ac(
                simps(cumtrapz(a, x=t, initial=0.0), x=t), self.deltat)


if __name__ == '__main__':
    unittest.main()
