import math
import unittest

import numpy as np

from pyrocko import orthodrome as od

from pocketseis.hifull import SurfaceDASCable, BoreholeDASCable


class DASCableTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cable_kw = dict(nominal_len=9.0, channel_spacing=3.0, gauge_len=12.0)
        cls.sc = SurfaceDASCable(**cable_kw)
        cls.bc = BoreholeDASCable(**cable_kw)

    def test_das_cable(self):
        for cable in (self.sc, self.bc):
            self.assertEqual(cable.n_channels, 4)
            self.assertEqual(cable.effective_len, 21.0)

            # Overlapping grids
            cable.set_grid_spacing(1.5)
            self.assertEqual(len(cable.grid_locs), 15)

            # Non-overlapping grids
            cable.set_grid_spacing(2.0)
            self.assertEqual(len(cable.grid_locs), 28)

    def test_surface_das_cable(self):
        azims = [45.0, 135.0, 225.0, 315.0]
        a = (self.sc.gauge_len / 2.0) / np.sqrt(2.0)
        elats, elons = od.ne_to_latlon(
            self.sc.nominal_refloc.effective_lat,
            self.sc.nominal_refloc.effective_lon,
            north_m=np.array([-a, +a, +a, -a]),
            east_m=np.array([-a, -a, +a, +a]))   # according to cable azimuths

        for i_azim, azim in enumerate(azims):
            setattr(self.sc, 'azimuth', azim)
            d = od.distance_accurate50m(
                elats[i_azim], elons[i_azim],
                *self.sc.effective_refloc.effective_latlon)

            # Distances less than abs_tol are accepted
            self.assertTrue(math.isclose(d, 0.0, rel_tol=0.0, abs_tol=0.1))


if __name__ == '__main__':
    unittest.main()
