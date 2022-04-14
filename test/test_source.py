from functools import partial
import unittest

import numpy as np
from scipy.integrate import simps, cumtrapz

from pocketseis.hifull import source as ps
from pocketseis.mtensor import tuple6_to_symmat


class SourceTestCase(unittest.TestCase):

    def test_source_mtqt(self):
        # Following sample calculation of uniform moment tensor
        # parametrization is taken from Appendix A, Tape & Tape (2015).
        mtqt_src = ps.MTQTSource(
            u=3.0 * np.pi / 8.0,
            v=-1.0 / 9.0,
            kappa=4.0 * np.pi / 5.0,
            sigma=-np.pi / 2.0,
            h=0.75)

        aac = partial(np.testing.assert_allclose, rtol=0., atol=0.001)
        aac(mtqt_src.beta, 1.571)
        aac(mtqt_src.gamma, -0.113)
        aac(mtqt_src.lune_lambda_triple, np.array([0.749, -0.092, -0.656]))
        aac(mtqt_src.theta, 0.723)

        rotmat_U_ref = np.array([
            [-587, -809, 37],
            [807, -588, -51],
            [63, 0, 998]], dtype=np.float64) * 0.001
        aac(mtqt_src.rotmat_U, rotmat_U_ref)

        m9_nwu_ref = tuple6_to_symmat((196, 455, -651, -397, -52, 71)) * 0.001
        aac(mtqt_src.m9_nwu, m9_nwu_ref)

        m9_ned_ref = tuple6_to_symmat((196, 455, -651, 397, 52, 71)) * 0.001
        aac(mtqt_src.m9_ned, m9_ned_ref)


if __name__ == '__main__':
    unittest.main()
