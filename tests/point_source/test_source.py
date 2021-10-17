from functools import partial
import unittest

import numpy as np
from scipy.integrate import simps, cumtrapz

from pocketseis import point_source as ps
from pocketseis.mtensor import tuple6_to_symmat


class PointSourceModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.deltat = 0.01
        cls.tref = 10.2

        factors = [10, 20, 41, 164]
        cls.durations = [factor*cls.deltat for factor in factors]

    def assert_ac(self, actual, desired):
        np.testing.assert_allclose(actual, desired, rtol=0.0, atol=1e-4)

    def test_stf_smooth_ramp(self):

        stf = ps.SmoothRampSTF(duration=1*self.deltat)
        t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
        self.assertEqual(a.max(), self.deltat)
        self.assert_ac(
            simps(np.diff(a, prepend=0.0)/self.deltat, x=t), 0.5*self.deltat)

        for duration in self.durations:
            stf = ps.SmoothRampSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assertEqual(a.max(), self.deltat)
            self.assert_ac(
                simps(np.diff(a, prepend=0.0)/self.deltat, x=t), self.deltat)

    def test_stf_gaussian(self):

        for duration in self.durations:
            stf = ps.GaussianSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assert_ac(simps(a, x=t), self.deltat)

    def test_stf_zero_crossing(self):

        for duration in self.durations:
            stf = ps.ZeroCrossingSTF(duration=duration, anchor=0.0)
            t, a = stf.discretize_t(deltat=self.deltat, tref=self.tref)
            self.assert_ac(simps(a, x=t), 0.0)
            self.assert_ac(
                simps(cumtrapz(a, x=t, initial=0.0), x=t), self.deltat)

    def test_source_mtqt(self):
        # Following sample calculation of uniform moment tensor
        # parametrization is taken from Appendix A, Tape & Tape (2015).
        mtqt_src = ps.MTQTSource(u=3.*np.pi/8.,
                                 v=-1./9.,
                                 kappa=4.*np.pi/5.,
                                 sigma=-np.pi/2.,
                                 h=0.75)

        aac = partial(np.testing.assert_allclose, rtol=0., atol=0.001)

        aac(mtqt_src.beta, 1.571)
        aac(mtqt_src.gamma, -0.113)
        aac(mtqt_src.lune_lambda_triple, np.array([0.749, -0.092, -0.656]))
        aac(mtqt_src.theta, 0.723)

        rotmat_U_ref = 0.001 * np.array([[-587, -809, 37],
                                         [807, -588, -51],
                                         [63, 0, 998]], dtype=np.float)
        aac(mtqt_src.rotmat_U, rotmat_U_ref)

        m9_nwu_ref = 0.001 * tuple6_to_symmat((196, 455, -651, -397, -52, 71))
        aac(mtqt_src.m9_nwu, m9_nwu_ref)

        m9_ned_ref = 0.001 * tuple6_to_symmat((196, 455, -651, 397, 52, 71))
        aac(mtqt_src.m9_ned, m9_ned_ref)


if __name__ == '__main__':
    unittest.main()
