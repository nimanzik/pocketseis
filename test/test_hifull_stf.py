from functools import partial
import unittest

import numpy as np
from scipy.integrate import simps, cumtrapz

from pocketseis.hifull.stf import SmoothRampSTF, GaussianSTF, ZeroCrossingSTF


class SourceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.deltat = 0.004
        cls.tref = 10.2
        cls.durations = [factor * cls.deltat for factor in (10, 20, 41, 164)]
        cls.assert_allclose = partial(
            np.testing.assert_allclose, rtol=0.0, atol=1.0e-4)

    def test_stf_smooth_ramp(self):

        stf = SmoothRampSTF(duration=1 * self.deltat)
        t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
        self.assertEqual(a.max(), self.deltat)
        self.assert_allclose(
            simps(np.diff(a, prepend=0.0) / self.deltat, x=t),
            0.5 * self.deltat)

        for duration in self.durations:
            stf = SmoothRampSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assertEqual(a.max(), self.deltat)
            self.assert_allclose(
                simps(np.diff(a, prepend=0.0) / self.deltat, x=t),
                self.deltat)

    def test_stf_gaussian(self):

        for duration in self.durations:
            stf = GaussianSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assert_allclose(simps(a, x=t), self.deltat)

    def test_stf_zero_crossing(self):

        for duration in self.durations:
            stf = ZeroCrossingSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assert_allclose(simps(a, x=t), 0.0)
            np.testing.assert_array_less(
                simps(cumtrapz(a, x=t, initial=0.0), x=t),
                self.deltat)


if __name__ == '__main__':
    unittest.main()
