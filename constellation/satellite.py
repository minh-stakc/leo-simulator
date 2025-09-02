"""
Satellite model.

Each satellite carries its orbital elements, current state vectors, and an
identifier encoding its plane and position within the constellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from config import EARTH_RADIUS_KM, CONSTELLATION
from constellation.orbit import KeplerianOrbit


@dataclass
class Satellite:
    """Model of a single LEO satellite.

    Attributes
    ----------
    sat_id : int
        Unique satellite identifier.
    plane_id : int
        Orbital plane index (0-based).
    plane_index : int
        Index within the plane (0-based).
    orbit : KeplerianOrbit
        Orbital elements for this satellite.
    """

    sat_id: int
    plane_id: int
    plane_index: int
    orbit: KeplerianOrbit

    # Mutable state -- updated each time step
    position_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_ecef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_km: float = 0.0
    _current_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_constellation_slot(
        cls,
        plane_id: int,
        plane_index: int,
        num_planes: int,
        sats_per_plane: int,
        altitude_km: float,
        inclination_deg: float,
        phase_offset: int = 1,
        pattern: str = "star",
    ) -> "Satellite":
        """Create a satellite from its Walker constellation slot.

        Parameters follow the Walker-Star (or Delta) convention:
          - RAAN is evenly spaced across 360 deg (delta) or 180 deg (star).
          - True anomaly is evenly spaced within each plane, with an
            inter-plane phase offset.
        """
        sat_id = plane_id * sats_per_plane + plane_index

        raan_spread = 180.0 if pattern == "star" else 360.0
        raan = (plane_id * raan_spread) / num_planes

        ta_spacing = 360.0 / sats_per_plane
        phase_shift = (phase_offset * plane_id * 360.0) / (num_planes * sats_per_plane)
        true_anomaly = (plane_index * ta_spacing + phase_shift) % 360.0

        orbit = KeplerianOrbit(
            semi_major_axis_km=EARTH_RADIUS_KM + altitude_km,
            eccentricity=0.0,
            inclination_deg=inclination_deg,
            raan_deg=raan,
            arg_perigee_deg=0.0,
            true_anomaly_deg=true_anomaly,
        )

        return cls(
            sat_id=sat_id,
            plane_id=plane_id,
            plane_index=plane_index,
            orbit=orbit,
        )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, dt_s: float) -> None:
        """Propagate the orbit and refresh all position fields.

        Parameters
        ----------
        dt_s : float
            Elapsed seconds since the simulation epoch.
        """
        self._current_time_s = dt_s
        self.position_eci, self.velocity_eci = self.orbit.propagate(dt_s)
        self.position_ecef = KeplerianOrbit.eci_to_ecef(self.position_eci, dt_s)
        self.lat_deg, self.lon_deg, self.alt_km = KeplerianOrbit.ecef_to_lla(
            self.position_ecef
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def coverage_half_angle_deg(self) -> float:
        """Half-cone angle of the satellite's nadir-pointed coverage area.

        Based on the minimum elevation angle from config.
        """
        from config import HANDOVER

        el_min = np.radians(HANDOVER["min_elevation_deg"])
        rho = np.arcsin(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + self.alt_km))
        eta = np.arccos(np.sin(el_min) / np.sin(rho))
        lam = np.pi / 2.0 - el_min - eta
        return float(np.degrees(lam))

    @property
    def coverage_radius_km(self) -> float:
        """Ground coverage radius in km (great-circle distance)."""
        return np.radians(self.coverage_half_angle_deg) * EARTH_RADIUS_KM

    def distance_to(self, other_pos_eci: np.ndarray) -> float:
        """Euclidean distance in km to another ECI position."""
        return float(np.linalg.norm(self.position_eci - other_pos_eci))

    def __repr__(self) -> str:
        return (
            f"Satellite(id={self.sat_id}, plane={self.plane_id}, "
            f"idx={self.plane_index}, alt={self.alt_km:.1f} km)"
        )


# ------------------------------------------------------------------
# Constellation factory
# ------------------------------------------------------------------

def create_constellation() -> list[Satellite]:
    """Build the full satellite constellation from :pymod:`config` parameters."""
    cfg = CONSTELLATION
    satellites = []
    for p in range(cfg["num_planes"]):
        for s in range(cfg["sats_per_plane"]):
            sat = Satellite.from_constellation_slot(
                plane_id=p,
                plane_index=s,
                num_planes=cfg["num_planes"],
                sats_per_plane=cfg["sats_per_plane"],
                altitude_km=cfg["altitude_km"],
                inclination_deg=cfg["inclination_deg"],
                phase_offset=cfg["walker_phase_offset"],
                pattern=cfg["pattern"],
            )
            satellites.append(sat)
    return satellites
