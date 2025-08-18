"""
Orbital mechanics module.

Provides Keplerian orbit propagation with J2 secular perturbation corrections.
Coordinates are in the Earth-Centered Inertial (ECI) frame unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from config import EARTH_MU, EARTH_RADIUS_KM, EARTH_J2


@dataclass
class KeplerianOrbit:
    """Classical Keplerian orbital elements with J2 secular drift.

    Parameters
    ----------
    semi_major_axis_km : float
        Semi-major axis [km].
    eccentricity : float
        Eccentricity (0 for circular LEO).
    inclination_deg : float
        Orbital inclination [deg].
    raan_deg : float
        Right Ascension of the Ascending Node [deg].
    arg_perigee_deg : float
        Argument of perigee [deg].
    true_anomaly_deg : float
        True anomaly at epoch [deg].
    """

    semi_major_axis_km: float
    eccentricity: float = 0.0
    inclination_deg: float = 53.0
    raan_deg: float = 0.0
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    # Derived (computed in __post_init__)
    _period_s: float = field(init=False, repr=False)
    _mean_motion: float = field(init=False, repr=False)  # rad/s
    _j2_raan_rate: float = field(init=False, repr=False)  # rad/s
    _j2_argp_rate: float = field(init=False, repr=False)  # rad/s

    def __post_init__(self) -> None:
        a_m = self.semi_major_axis_km * 1e3
        self._period_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / EARTH_MU)
        self._mean_motion = 2.0 * math.pi / self._period_s

        # J2 secular perturbation rates
        inc = math.radians(self.inclination_deg)
        p = a_m * (1.0 - self.eccentricity ** 2)
        re = EARTH_RADIUS_KM * 1e3
        n = self._mean_motion
        factor = -1.5 * n * EARTH_J2 * (re / p) ** 2

        self._j2_raan_rate = factor * math.cos(inc)  # rad/s
        self._j2_argp_rate = factor * (2.0 - 2.5 * math.sin(inc) ** 2)  # rad/s

    @property
    def period_s(self) -> float:
        """Orbital period in seconds."""
        return self._period_s

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion in rad/s."""
        return self._mean_motion

    def propagate(self, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate the orbit forward by *dt_s* seconds from epoch.

        Returns ECI position [km] and velocity [km/s] as 3-vectors.
        """
        a = self.semi_major_axis_km
        e = self.eccentricity
        inc = math.radians(self.inclination_deg)
        raan = math.radians(self.raan_deg) + self._j2_raan_rate * dt_s
        argp = math.radians(self.arg_perigee_deg) + self._j2_argp_rate * dt_s

        # Mean anomaly at time dt_s
        M0 = self._true_to_mean(math.radians(self.true_anomaly_deg), e)
        M = M0 + self._mean_motion * dt_s

        # Solve Kepler's equation for eccentric anomaly
        E = self._solve_kepler(M, e)

        # True anomaly
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + e) * math.sin(E / 2.0),
            math.sqrt(1.0 - e) * math.cos(E / 2.0),
        )

        # Distance
        r_mag = a * (1.0 - e * math.cos(E))

        # Position in orbital plane
        r_orb = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])

        # Velocity in orbital plane
        p = a * (1.0 - e ** 2)
        mu_km = EARTH_MU * 1e-9  # km^3/s^2
        h = math.sqrt(mu_km * p)
        v_orb = np.array([
            -(mu_km / h) * math.sin(nu),
            (mu_km / h) * (e + math.cos(nu)),
            0.0,
        ])

        # Rotation matrix: orbital plane -> ECI
        R = self._rotation_matrix(raan, argp, inc)
        pos_eci = R @ r_orb
        vel_eci = R @ v_orb

        return pos_eci, vel_eci

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
        """Solve Kepler's equation  M = E - e*sin(E)  via Newton-Raphson."""
        E = M if e < 0.8 else math.pi
        for _ in range(max_iter):
            dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < tol:
                break
        return E

    @staticmethod
    def _true_to_mean(nu: float, e: float) -> float:
        """Convert true anomaly to mean anomaly."""
        E = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu / 2.0),
            math.sqrt(1.0 + e) * math.cos(nu / 2.0),
        )
        M = E - e * math.sin(E)
        return M

    @staticmethod
    def _rotation_matrix(raan: float, argp: float, inc: float) -> np.ndarray:
        """Build the 3-1-3 rotation matrix from orbital frame to ECI."""
        cos_O, sin_O = math.cos(raan), math.sin(raan)
        cos_w, sin_w = math.cos(argp), math.sin(argp)
        cos_i, sin_i = math.cos(inc), math.sin(inc)

        return np.array([
            [
                cos_O * cos_w - sin_O * sin_w * cos_i,
                -cos_O * sin_w - sin_O * cos_w * cos_i,
                sin_O * sin_i,
            ],
            [
                sin_O * cos_w + cos_O * sin_w * cos_i,
                -sin_O * sin_w + cos_O * cos_w * cos_i,
                -cos_O * sin_i,
            ],
            [
                sin_w * sin_i,
                cos_w * sin_i,
                cos_i,
            ],
        ])

    @staticmethod
    def eci_to_ecef(pos_eci: np.ndarray, dt_s: float) -> np.ndarray:
        """Rotate ECI position to ECEF given elapsed time from epoch."""
        from config import EARTH_ROTATION_RATE

        theta = EARTH_ROTATION_RATE * dt_s
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array([
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return R @ pos_eci

    @staticmethod
    def ecef_to_lla(pos_ecef: np.ndarray) -> Tuple[float, float, float]:
        """Convert ECEF [km] to geodetic (lat_deg, lon_deg, alt_km).

        Uses a simplified spherical Earth model.
        """
        x, y, z = pos_ecef
        r = np.linalg.norm(pos_ecef)
        lat = math.degrees(math.asin(z / r))
        lon = math.degrees(math.atan2(y, x))
        alt = r - EARTH_RADIUS_KM
        return lat, lon, alt
