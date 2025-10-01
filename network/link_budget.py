"""
Link budget calculations.

Computes free-space path loss, received signal strength, SNR, and achievable
data rate for both ground-to-satellite and inter-satellite links.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import LINK_BUDGET, SPEED_OF_LIGHT, BOLTZMANN_DB


@dataclass
class LinkResult:
    """Result of a link budget calculation."""

    slant_range_km: float
    free_space_loss_db: float
    total_loss_db: float
    received_power_dbw: float
    noise_power_dbw: float
    snr_db: float
    capacity_mbps: float
    propagation_delay_ms: float


class LinkBudgetCalculator:
    """Compute link budgets for satellite communication links."""

    def __init__(
        self,
        frequency_ghz: float = LINK_BUDGET["frequency_ghz"],
        sat_tx_power_dbw: float = LINK_BUDGET["sat_tx_power_dbw"],
        sat_antenna_gain_dbi: float = LINK_BUDGET["sat_antenna_gain_dbi"],
        gs_antenna_gain_dbi: float = LINK_BUDGET["gs_antenna_gain_dbi"],
        system_noise_temp_k: float = LINK_BUDGET["gs_system_noise_temp_k"],
        atmospheric_loss_db: float = LINK_BUDGET["atmospheric_loss_db"],
        pointing_loss_db: float = LINK_BUDGET["pointing_loss_db"],
        rain_margin_db: float = LINK_BUDGET["rain_margin_db"],
        bandwidth_mhz: float = LINK_BUDGET["bandwidth_mhz"],
    ) -> None:
        self.frequency_ghz = frequency_ghz
        self.sat_tx_power_dbw = sat_tx_power_dbw
        self.sat_antenna_gain_dbi = sat_antenna_gain_dbi
        self.gs_antenna_gain_dbi = gs_antenna_gain_dbi
        self.system_noise_temp_k = system_noise_temp_k
        self.atmospheric_loss_db = atmospheric_loss_db
        self.pointing_loss_db = pointing_loss_db
        self.rain_margin_db = rain_margin_db
        self.bandwidth_hz = bandwidth_mhz * 1e6

        # Wavelength in metres
        self._wavelength_m = (SPEED_OF_LIGHT * 1e3) / (frequency_ghz * 1e9)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground_to_sat(
        self,
        slant_range_km: float,
        elevation_deg: float,
    ) -> LinkResult:
        """Compute downlink budget from satellite to ground station.

        Parameters
        ----------
        slant_range_km : float
            Slant range between station and satellite [km].
        elevation_deg : float
            Elevation angle at the ground station [deg].
        """
        fspl = self._free_space_path_loss(slant_range_km)

        # Elevation-dependent atmospheric loss (increases at low elevations)
        atm_loss = self.atmospheric_loss_db / max(math.sin(math.radians(elevation_deg)), 0.1)

        total_loss = (
            fspl
            + atm_loss
            + self.pointing_loss_db
            + self.rain_margin_db
        )

        received_power = (
            self.sat_tx_power_dbw
            + self.sat_antenna_gain_dbi
            + self.gs_antenna_gain_dbi
            - total_loss
        )

        noise_power = BOLTZMANN_DB + 10 * math.log10(self.system_noise_temp_k) + 10 * math.log10(self.bandwidth_hz)
        snr = received_power - noise_power

        # Shannon capacity  C = B * log2(1 + SNR_linear)
        snr_linear = 10 ** (snr / 10.0)
        capacity_bps = self.bandwidth_hz * math.log2(1.0 + snr_linear)
        capacity_mbps = capacity_bps / 1e6

        delay_ms = (slant_range_km / SPEED_OF_LIGHT) * 1e3

        return LinkResult(
            slant_range_km=slant_range_km,
            free_space_loss_db=fspl,
            total_loss_db=total_loss,
            received_power_dbw=received_power,
            noise_power_dbw=noise_power,
            snr_db=snr,
            capacity_mbps=capacity_mbps,
            propagation_delay_ms=delay_ms,
        )

    def inter_satellite(self, distance_km: float) -> LinkResult:
        """Compute an inter-satellite link (ISL) budget.

        ISLs operate in free space (no atmospheric losses) and use optical
        or high-frequency RF terminals.  We model a simplified RF ISL here.
        """
        fspl = self._free_space_path_loss(distance_km)

        # ISL: both ends use satellite-class antennas, no atmosphere
        total_loss = fspl + self.pointing_loss_db

        received_power = (
            self.sat_tx_power_dbw
            + self.sat_antenna_gain_dbi
            + self.sat_antenna_gain_dbi
            - total_loss
        )

        noise_power = BOLTZMANN_DB + 10 * math.log10(self.system_noise_temp_k) + 10 * math.log10(self.bandwidth_hz)
        snr = received_power - noise_power
        snr_linear = 10 ** (snr / 10.0)
        capacity_bps = self.bandwidth_hz * math.log2(1.0 + snr_linear)
        capacity_mbps = capacity_bps / 1e6

        delay_ms = (distance_km / SPEED_OF_LIGHT) * 1e3

        return LinkResult(
            slant_range_km=distance_km,
            free_space_loss_db=fspl,
            total_loss_db=total_loss,
            received_power_dbw=received_power,
            noise_power_dbw=noise_power,
            snr_db=snr,
            capacity_mbps=capacity_mbps,
            propagation_delay_ms=delay_ms,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _free_space_path_loss(self, distance_km: float) -> float:
        """Free-space path loss [dB].

        FSPL = 20*log10(4*pi*d/lambda)  where d and lambda in same units.
        """
        d_m = distance_km * 1e3
        fspl = 20.0 * math.log10(4.0 * math.pi * d_m / self._wavelength_m)
        return fspl
