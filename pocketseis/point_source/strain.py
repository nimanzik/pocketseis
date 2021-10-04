import numpy as np

from pyrocko import model as pmodel, orthodrome as pod
from pyrocko.guts import Object, Float


class DASCable(Object):
    nominal_zerooffset_loc = pmodel.Location.T(
        help='Geographycal location of first channel')
    nominal_length = Float.T(help='Unit: m')
    azimuth = Float.T(help='Unit: deg')
    channel_spacing = Float.T(help='Unit: m')
    gauge_length = Float.T(help='Unit: m')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_channels = None
        self._channel_locs = None
        self._real_length = None
        self._real_zerooffset_loc = None
        self._grid_spacing = None
        self._grid_locs = None

    @property
    def n_channels(self):
        if self._n_channels is None:
            self._n_channels = (
                int(round(self.nominal_length / self.channel_spacing)) + 1)
        return self._n_channels

    @property
    def channel_locs(self):
        if self._channel_locs is None:
            cha_offsets = np.linspace(
                0.0, self.nominal_length, self.n_channels)

            cha_latlons = pod.azidist_to_latlon(
                self.nominal_zerooffset_loc.effective_lat,
                self.nominal_zerooffset_loc.effective_lon,
                np.ones(self.n_channels) * self.azimuth,
                cha_offsets * pod.m2d)

            cha_locs = []
            for clat, clon in cha_latlons.T:
                cha_locs.append(pmodel.Location(lat=clat, lon=clon))

            self._channel_locs = cha_locs

        return self._channel_locs

    @property
    def real_length(self):
        """Cable total length, from x=(Cha_1 - GL/2) to x=(Cha_N + GL/2)"""
        if self._real_length is None:
            self._real_length = (
                (self.n_channels - 1) * self.channel_spacing
                + self.gauge_length)

        return self._real_length

    @property
    def real_zerooffset_loc(self):
        """Geographical location of x=(Cha_1 - GL/2)"""
        if self._real_zerooffset_loc is None:
            lat0, lon0 = pod.azidist_to_latlon(
                self.nominal_zerooffset_loc.effective_lat,
                self.nominal_zerooffset_loc.effective_lon,
                self.azimuth + 180.0,
                self.gauge_length / 2.0 * pod.m2d)

            self._real_zerooffset_loc = pmodel.Location(lat=lat0, lon=lon0)

        return self._real_zerooffset_loc

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @property
    def grid_locs(self):
        return self._grid_locs

    @grid_spacing.setter
    def grid_spacing(self, grid_spacing):
        self._grid_spacing = grid_spacing

        n_grids = int(round(self.real_length / grid_spacing)) + 1
        grid_offsets = np.linspace(0.0, self.real_length, n_grids)
        grid_latlons = pod.azidist_to_latlon(
            self.real_zerooffset_loc.effective_lat,
            self.real_zerooffset_loc.effective_lon,
            np.ones(n_grids) * self.azimuth,
            grid_offsets * pod.m2d)

        grid_locs = []
        for glat, glon in grid_latlons.T:
            grid_locs.append(pmodel.Location(lat=glat, lon=glon))

        self._grid_locs = grid_locs
