"""
Visibility window calculation.

Determines when each satellite is visible from each ground station based on
elevation angle, and computes the remaining time in each visibility window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import HANDOVER, SIMULATION
from constellation.satellite import Satellite
from constellation.ground_station import GroundStation


@dataclass
class VisibilityWindow:
    """A contiguous interval during which a satellite is visible from a station."""

    sat_id: int
    gs_id: int
    start_s: float
    end_s: float
    peak_elevation_deg: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def remaining_s(self, current_time_s: float) -> float:
        """Seconds remaining in this window from the given time."""
        return max(0.0, self.end_s - current_time_s)


class VisibilityCalculator:
    """Computes and caches visibility windows for all satellite-station pairs."""

    def __init__(
        self,
        satellites: List[Satellite],
        ground_stations: List[GroundStation],
        min_elevation_deg: Optional[float] = None,
        time_step_s: Optional[float] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        self.satellites = satellites
        self.ground_stations = ground_stations
        self.min_elevation = min_elevation_deg or HANDOVER["min_elevation_deg"]
        self.time_step = time_step_s or SIMULATION["time_step_s"]
        self.duration = duration_s or SIMULATION["duration_s"]

        # Cache: (gs_id, sat_id) -> list of VisibilityWindow
        self._windows: Dict[Tuple[int, int], List[VisibilityWindow]] = {}
        self._computed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_windows(self) -> None:
        """Pre-compute visibility windows for every station-satellite pair.

        This scans through the simulation timeline at the configured time step
        and detects elevation-angle crossings.
        """
        times = np.arange(0.0, self.duration + self.time_step, self.time_step)

        for gs in self.ground_stations:
            for sat in self.satellites:
                key = (gs.gs_id, sat.sat_id)
                self._windows[key] = self._find_windows(gs, sat, times)

        self._computed = True

    def visible_satellites(
        self, gs: GroundStation, current_time_s: float
    ) -> List[Tuple[Satellite, float, float]]:
        """Return satellites currently visible from *gs*.

        Returns a list of (satellite, elevation_deg, remaining_visibility_s) tuples,
        sorted by descending elevation.
        """
        results = []
        for sat in self.satellites:
            sat.update(current_time_s)
            el = gs.elevation_angle(sat.position_eci, current_time_s)
            if el >= self.min_elevation:
                remaining = self._remaining_visibility(gs.gs_id, sat.sat_id, current_time_s)
                results.append((sat, el, remaining))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_windows(self, gs_id: int, sat_id: int) -> List[VisibilityWindow]:
        """Return pre-computed windows for a station-satellite pair."""
        return self._windows.get((gs_id, sat_id), [])

    def is_visible(self, gs: GroundStation, sat: Satellite, time_s: float) -> bool:
        """Quick check whether *sat* is visible from *gs* at *time_s*."""
        el = gs.elevation_angle(sat.position_eci, time_s)
        return el >= self.min_elevation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_windows(
        self,
        gs: GroundStation,
        sat: Satellite,
        times: np.ndarray,
    ) -> List[VisibilityWindow]:
        """Detect visibility windows by scanning elevation over time."""
        windows: List[VisibilityWindow] = []
        in_window = False
        start_t = 0.0
        peak_el = -90.0

        for t in times:
            sat.update(t)
            el = gs.elevation_angle(sat.position_eci, t)

            if el >= self.min_elevation:
                if not in_window:
                    in_window = True
                    start_t = t
                    peak_el = el
                else:
                    peak_el = max(peak_el, el)
            else:
                if in_window:
                    windows.append(
                        VisibilityWindow(
                            sat_id=sat.sat_id,
                            gs_id=gs.gs_id,
                            start_s=start_t,
                            end_s=t - self.time_step,
                            peak_elevation_deg=peak_el,
                        )
                    )
                    in_window = False

        # Close any open window at end of simulation
        if in_window:
            windows.append(
                VisibilityWindow(
                    sat_id=sat.sat_id,
                    gs_id=gs.gs_id,
                    start_s=start_t,
                    end_s=times[-1],
                    peak_elevation_deg=peak_el,
                )
            )

        return windows

    def _remaining_visibility(
        self, gs_id: int, sat_id: int, current_time_s: float
    ) -> float:
        """Estimate remaining seconds in the current visibility window.

        If pre-computed windows are available, use them; otherwise fall back
        to a forward-looking scan.
        """
        if self._computed:
            for w in self._windows.get((gs_id, sat_id), []):
                if w.start_s <= current_time_s <= w.end_s:
                    return w.remaining_s(current_time_s)
            return 0.0

        # Fallback: quick forward scan (up to 10 min)
        gs = self.ground_stations[gs_id]
        sat = self.satellites[sat_id]
        for dt in np.arange(0, 600, self.time_step):
            t = current_time_s + dt
            sat.update(t)
            el = gs.elevation_angle(sat.position_eci, t)
            if el < self.min_elevation:
                return dt
        return 600.0
