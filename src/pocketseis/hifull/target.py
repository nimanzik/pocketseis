import math

import numpy as np

from pyrocko import model as pmodel, orthodrome as od
from pyrocko.guts import Float, Object

from pocketseis.compass import compute_relative_data


guts_prefix = 'pf'


class BaseDASCable(Object):
    nominal_refloc = pmodel.Location.T(
        default=pmodel.Location(lat=0.0, lon=0.0, depth=0.0),
        help='Geographycal location of the first channel')
    nominal_len = Float.T(
        default=1000.0,
        help='Cable length between the first and last channels. Unit: [m]')
    channel_spacing = Float.T(
        default=1.0,
        help='Channel spacing. Unit: [m]')
    gauge_len = Float.T(
        default=10.0,
        help='Gauge length. Unit: [m]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_spacing = None
        self.grid_locs = list()

    @property
    def n_channels(self):
        return int(round(self.nominal_len / self.channel_spacing) + 1)

    @property
    def effective_len(self):
        """
        Returns
        -------
        Effective length in m, from `xi=(first_channel - GL/2)` to
        `xf =(last_channel + GL/2)`, where `GL` is the gauge length.
        """
        return self.nominal_len + self.gauge_len

    def _get_grid_offsets(self, new_grid_spacing):
        """
        Returns
        -------
        grid_offsets : ndarray of shape (n_grids,)
            Grid point offsets from the effective reference location.
        """
        if math.isclose(
                math.remainder(self.channel_spacing, new_grid_spacing), 0.0,
                rel_tol=0.0, abs_tol=1e-9):
            # Some grids overlap. We can use moving average
            n_grids = int(round(self.effective_len / new_grid_spacing) + 1)
            grid_offsets = np.linspace(0.0, self.effective_len, n_grids)
        else:
            # Grids do not overlap. Must take average over separate GLs
            # In this case, n_grids = n_channels * n_grids_per_gl
            n_grids_per_gl = int(round(self.gauge_len / new_grid_spacing) + 1)
            a = np.arange(self.n_channels)[:, None] * self.channel_spacing
            b = np.linspace(0.0, self.gauge_len, n_grids_per_gl)
            grid_offsets = (a + b).flatten()

        return grid_offsets


class SurfaceDASCable(BaseDASCable):
    azimuth = Float.T(
        default=0.0,
        help='Azimuth measured clockwise from the North(=x) and is '
             'defined as the angle between vector pointing from the '
             'interrogator to the North and the vector pointing from '
             'the interrogator to the other end of the cable. Unit: [deg]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cable depth below surface (positive downward). Unit: [m]
        self.depth = self.nominal_refloc.depth

    @property
    def effective_refloc(self):
        """
        Geographical location of point `x=(first_channel - GL/2)`, where
        `GL` is the gauge length.

        Returns
        -------
        :py:class:`pyrocko.model.Location` object
        """
        lat0, lon0 = od.azidist_to_latlon(
            self.nominal_refloc.effective_lat,
            self.nominal_refloc.effective_lon,
            self.azimuth + 180.0,
            self.gauge_len / 2.0 * od.m2d)

        return pmodel.Location(lat=lat0, lon=lon0, depth=self.depth)

    @property
    def channel_locs(self):
        """
        Returns
        -------
        out : list of :py:class:`pyrocko.model.Location` objects
            Geographical locations of the channels.
        """
        cha_offsets = np.linspace(0.0, self.nominal_len, self.n_channels)

        # Channel lats & lons, 2-tuple with arrays of shape (n_channels,)
        cha_lats, cha_lons = od.azidist_to_latlon(
            self.nominal_refloc.effective_lat,
            self.nominal_refloc.effective_lon,
            np.full(self.n_channels, self.azimuth),
            cha_offsets * od.m2d)

        cha_locs = [
            pmodel.Location(lat=clat, lon=clon, depth=self.depth)
            for clat, clon in zip(cha_lats, cha_lons)]

        return cha_locs

    def set_grid_spacing(self, new_grid_spacing):
        """
        Set grid spacing in [m] that is used for discretising the gauge
        length to calculate point strains.
        """
        grid_offsets = self._get_grid_offsets(new_grid_spacing)

        # Grid lats & lons, 2-tuple of arrays of shape (n_grids,)
        grid_lats, grid_lons = od.azidist_to_latlon(
            self.effective_refloc.effective_lat,
            self.effective_refloc.effective_lon,
            np.full_like(grid_offsets, self.azimuth),
            grid_offsets * od.m2d)

        grid_locs = [
            pmodel.Location(lat=glat, lon=glon, depth=self.depth)
            for glat, glon in zip(grid_lats, grid_lons)]

        self.grid_spacing = new_grid_spacing
        self.grid_locs = grid_locs

    def get_event_relative_data(self, event, level='channel'):
        """
        Parameters
        ----------
        event : :py:class:`pyrocko.model.Event` object
            Seismic event. Its location is considered as the origin of
            the Cartesian system to obtain relative data.
        level : {'channel', 'grid'}
            Determines the observation points for which relative data
            are calculated.

        Returns
        -------
        dists_3d : ndarray of shape (n_receivers,)
            3-D distances of observation points from event. Unit: [m]
        cosine_vecs : ndarray of shape (n_receivers, 3)
            Unit vectors of direction cosines. Relative event-receiver
            (either channels or grids) position vectors are in Cartesian
            coordinate system (i.e. vectors whose initial and terminal
            are event and observation  points, respectively). The
            indexes of the second dimension represent
            (0, 1, 2)->(North, East, Down) axes.
        """
        if level not in (valid_levels := ('channel', 'grid')):
            raise ValueError(f"Valid levels are {valid_levels}")

        if level == 'grid':
            assert self.grid_spacing is not None, (
                "cannot set relative data for grids when 'grid_spacing' "
                "is None. Run 'set_grid_spacing()' first")

        if level == 'channel':
            rdepths = np.full_like(self.channel_locs, self.depth)
            rlats, rlons = zip(
                *[c.effective_latlon for c in self.channel_locs])
        else:
            rdepths = np.full_like(self.grid_locs, self.depth)
            rlats, rlons = zip(*[g.effective_latlon for g in self.grid_locs])

        return compute_relative_data(
            *event.effective_latlon, event.depth, rlats, rlons, rdepths)


class BoreholeDASCable(BaseDASCable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def effective_refloc(self):
        """
        Geographical location of point `x=(first_channel - GL/2)`, where
        `GL` is the gauge length.

        Returns
        -------
        :py:class:`pyrocko.model.Location` object
        """
        lat0, lon0 = self.nominal_refloc.effective_latlon
        depth0 = self.nominal_refloc.depth - (self.gauge_len / 2.0)
        return pmodel.Location(lat=lat0, lon=lon0, depth=depth0)

    @property
    def channel_locs(self):
        """
        Returns
        -------
        out : list of :py:class:`pyrocko.model.Location` objects
            Geographical locations of the channels.
        """
        cha_lats = np.full(self.n_channels, self.nominal_refloc.effective_lat)
        cha_lons = np.full(self.n_channels, self.nominal_refloc.effective_lon)

        cha_offsets = np.linspace(0.0, self.nominal_len, self.n_channels)
        cha_depths = self.nominal_refloc.depth + cha_offsets

        cha_locs = [
            pmodel.Location(lat=clat, lon=clon, depth=cdepth)
            for clat, clon, cdepth in zip(cha_lats, cha_lons, cha_depths)]

        return cha_locs

    def set_grid_spacing(self, new_grid_spacing):
        """
        Set grid spacing in [m] that is used for discretising the gauge
        length to calculate point strains.
        """
        grid_offsets = self._get_grid_offsets(new_grid_spacing)
        grid_depths = self.effective_refloc.depth + grid_offsets
        lat0, lon0 = self.effective_refloc.effective_latlon
        grid_lats = np.full_like(grid_offsets, lat0)
        grid_lons = np.full_like(grid_offsets, lon0)

        grid_locs = [
            pmodel.Location(lat=glat, lon=glon, depth=gdepth)
            for glat, glon, gdepth in zip(grid_lats, grid_lons, grid_depths)]

        self.grid_spacing = new_grid_spacing
        self.grid_locs = grid_locs

    def get_event_relative_data(self, event, level='channel'):
        if level not in (valid_levels := ('channel', 'grid')):
            raise ValueError(f"Valid levels are {valid_levels}")

        if level == 'grid':
            assert self.grid_spacing is not None, (
                "cannot set relative data for grids when 'grid_spacing' "
                "is None. Run 'set_grid_spacing()' first")

        lat0, lon0 = self.nominal_refloc.effective_latlon

        if level == 'channel':
            rlats = np.full_like(self.channel_locs, lat0)
            rlons = np.full_like(self.channel_locs, lon0)
            rdepths = [c.depth for c in self.channel_locs]
        else:
            rlats = np.full_like(self.grid_locs, lat0)
            rlons = np.full_like(self.grid_locs, lon0)
            rdepths = [g.depth for g in self.grid_locs]

        return compute_relative_data(
            *event.effective_latlon, event.depth, rlats, rlons, rdepths)


__all__ = ['SurfaceDASCable', 'BoreholeDASCable']
