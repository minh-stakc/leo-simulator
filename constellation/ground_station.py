"""
Ground station model.

Each ground station is fixed on the Earth's surface and characterised by its
geodetic coordinates and antenna properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import EARTH_RADIUS_KM, EARTH_ROTATION_RATE, GROUND_STATIONS, LINK_BUDGET


@dataclass
class GroundStation:
    """Model of a single ground station.

    Attributes
    ----------
    gs_id : int
        Unique ground station identifier.
    name : str
        Human-readable name.
    lat_deg : float
        Geodetic latitude [deg].
    lon_deg : float
        Geodetic longitude [deg].
    alt_km : float
        Altitude above sea level [km].
    min_elevation_deg : float
        Minimum elevation angle for link acquisition [deg].
    antenna_gain_dbi : float
        Receive antenna gain [dBi].
    system_noise_temp_k : float
        Receiver system noise temperature [K].
    """

    gs_id: int
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = LINK_BUDGET["min_elevation_deg"]
    antenna_gain_dbi: float = LINK_BUDGET["gs_antenna_gain_dbi"]
    system_noise_temp_k: float = LINK_BUDGET["gs_system_noise_temp_k"]

    # ECEF position (fixed, computed once)
    position_ecef: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.position_ecef = self._lla_to_ecef(self.lat_deg, self.lon_deg, self.alt_km)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
        """Convert geodetic (lat, lon, alt) to ECEF [km] (spherical model)."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        r = EARTH_RADIUS_KM + alt_km
        return np.array([
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        ])

    def position_eci(self, dt_s: float) -> np.ndarray:
        """Return ECI position at elapsed time *dt_s* from epoch.

        The ground station co-rotates with the Earth, so we reverse the
        ECEF -> ECI rotation.
        """
        theta = EARTH_ROTATION_RATE * dt_s
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return R @ self.position_ecef

    def elevation_angle(self, sat_pos_eci: np.ndarray, dt_s: float) -> float:
        """Compute the elevation angle [deg] of a satellite as seen from this station.

        Parameters
        ----------
        sat_pos_eci : np.ndarray
            Satellite ECI position [km].
        dt_s : float
            Elapsed seconds since epoch (needed for station ECI position).

        Returns
        -------
        float
            Elevation angle in degrees.  Negative means below horizon.
        """
        gs_eci = self.position_eci(dt_s)
        diff = sat_pos_eci - gs_eci
        range_km = np.linalg.norm(diff)
        if range_km < 1e-6:
            return 90.0

        # Unit vector from station to satellite
        u_diff = diff / range_km

        # Local "up" direction at the station (radial)
        u_up = gs_eci / np.linalg.norm(gs_eci)

        # Elevation = complement of zenith angle
        sin_el = np.dot(u_diff, u_up)
        elevation_deg = math.degrees(math.asin(np.clip(sin_el, -1.0, 1.0)))
        return elevation_deg

    def slant_range_km(self, sat_pos_eci: np.ndarray, dt_s: float) -> float:
        """Distance from station to satellite [km]."""
        gs_eci = self.position_eci(dt_s)
        return float(np.linalg.norm(sat_pos_eci - gs_eci))

    def __repr__(self) -> str:
        return f"GroundStation('{self.name}', lat={self.lat_deg:.2f}, lon={self.lon_deg:.2f})"


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_ground_stations() -> List[GroundStation]:
    """Build ground stations from :pymod:`config` parameters."""
    stations = []
    for i, gs_cfg in enumerate(GROUND_STATIONS):
        stations.append(
            GroundStation(
                gs_id=i,
                name=gs_cfg["name"],
                lat_deg=gs_cfg["lat"],
                lon_deg=gs_cfg["lon"],
                alt_km=gs_cfg.get("alt_km", 0.0),
            )
        )
    return stations
