"""
Satellite handover logic.

Implements several handover strategies for ground stations switching from one
serving satellite to another as orbital geometry changes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import HANDOVER
from constellation.satellite import Satellite
from constellation.ground_station import GroundStation
from constellation.visibility import VisibilityCalculator
from network.link_budget import LinkBudgetCalculator, LinkResult

logger = logging.getLogger(__name__)


@dataclass
class HandoverEvent:
    """Record of a single handover."""

    time_s: float
    gs_id: int
    from_sat_id: Optional[int]
    to_sat_id: int
    reason: str
    elevation_deg: float
    remaining_visibility_s: float


class HandoverManager:
    """Manages serving-satellite assignments for ground stations.

    Supports three strategies (configured via ``config.HANDOVER["strategy"]``):
      - ``best_elevation``: hand over to the satellite with the highest elevation.
      - ``longest_visibility``: prefer the satellite with the longest remaining
        visibility window.
      - ``best_snr``: choose the satellite yielding the best link SNR.
    """

    def __init__(
        self,
        satellites: List[Satellite],
        ground_stations: List[GroundStation],
        visibility_calc: VisibilityCalculator,
        link_calc: Optional[LinkBudgetCalculator] = None,
    ) -> None:
        self.satellites = {s.sat_id: s for s in satellites}
        self.ground_stations = {gs.gs_id: gs for gs in ground_stations}
        self.visibility_calc = visibility_calc
        self.link_calc = link_calc or LinkBudgetCalculator()

        self.strategy = HANDOVER["strategy"]
        self.hysteresis = HANDOVER["hysteresis_deg"]
        self.min_remaining = HANDOVER["min_remaining_visibility_s"]
        self.handover_delay_ms = HANDOVER["handover_delay_ms"]
        self.max_rate = HANDOVER["max_handovers_per_min"]

        # Current serving satellite per ground station
        self.serving: Dict[int, Optional[int]] = {
            gs.gs_id: None for gs in ground_stations
        }

        # Handover history
        self.events: List[HandoverEvent] = []

        # Rate limiting: timestamps of recent handovers per GS
        self._recent_handovers: Dict[int, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, time_s: float) -> List[HandoverEvent]:
        """Evaluate handover decisions for all ground stations at *time_s*.

        Returns a list of handover events that occurred.
        """
        new_events: List[HandoverEvent] = []

        for gs in self.ground_stations.values():
            event = self._evaluate_station(gs, time_s)
            if event is not None:
                new_events.append(event)
                self.events.append(event)

        return new_events

    def get_serving(self, gs_id: int) -> Optional[int]:
        """Return the sat_id currently serving the given ground station."""
        return self.serving.get(gs_id)

    # ------------------------------------------------------------------
    # Core handover logic
    # ------------------------------------------------------------------

    def _evaluate_station(
        self, gs: GroundStation, time_s: float
    ) -> Optional[HandoverEvent]:
        """Check whether *gs* needs a handover at *time_s*."""
        visible = self.visibility_calc.visible_satellites(gs, time_s)

        if not visible:
            # No satellites visible -- force disconnect
            if self.serving[gs.gs_id] is not None:
                old_sat = self.serving[gs.gs_id]
                self.serving[gs.gs_id] = None
                return HandoverEvent(
                    time_s=time_s,
                    gs_id=gs.gs_id,
                    from_sat_id=old_sat,
                    to_sat_id=-1,
                    reason="no_visible_satellite",
                    elevation_deg=0.0,
                    remaining_visibility_s=0.0,
                )
            return None

        # Rank candidates by strategy
        best = self._select_best(gs, visible, time_s)
        if best is None:
            return None

        best_sat, best_el, best_remaining = best
        current_sat_id = self.serving[gs.gs_id]

        # Initial acquisition
        if current_sat_id is None:
            self.serving[gs.gs_id] = best_sat.sat_id
            return HandoverEvent(
                time_s=time_s,
                gs_id=gs.gs_id,
                from_sat_id=None,
                to_sat_id=best_sat.sat_id,
                reason="initial_acquisition",
                elevation_deg=best_el,
                remaining_visibility_s=best_remaining,
            )

        # Check if current satellite is still visible
        current_visible = [v for v in visible if v[0].sat_id == current_sat_id]

        if not current_visible:
            # Current satellite no longer visible -- forced handover
            if self._rate_limited(gs.gs_id, time_s):
                return None
            self.serving[gs.gs_id] = best_sat.sat_id
            self._record_handover_time(gs.gs_id, time_s)
            return HandoverEvent(
                time_s=time_s,
                gs_id=gs.gs_id,
                from_sat_id=current_sat_id,
                to_sat_id=best_sat.sat_id,
                reason="current_satellite_lost",
                elevation_deg=best_el,
                remaining_visibility_s=best_remaining,
            )

        # Current satellite still visible -- check if handover is beneficial
        cur_sat, cur_el, cur_remaining = current_visible[0]

        # Trigger handover if remaining visibility is below threshold
        needs_handover = False
        reason = ""

        if cur_remaining < self.min_remaining and best_remaining > cur_remaining:
            needs_handover = True
            reason = "low_remaining_visibility"
        elif best_sat.sat_id != current_sat_id and best_el > cur_el + self.hysteresis:
            if self.strategy == "best_elevation":
                needs_handover = True
                reason = "better_elevation"
            elif self.strategy == "longest_visibility" and best_remaining > cur_remaining + 60:
                needs_handover = True
                reason = "longer_visibility"
            elif self.strategy == "best_snr":
                best_snr = self._compute_snr(gs, best_sat, time_s)
                cur_snr = self._compute_snr(gs, cur_sat, time_s)
                if best_snr > cur_snr + 2.0:  # 2 dB hysteresis
                    needs_handover = True
                    reason = "better_snr"

        if needs_handover and not self._rate_limited(gs.gs_id, time_s):
            self.serving[gs.gs_id] = best_sat.sat_id
            self._record_handover_time(gs.gs_id, time_s)
            return HandoverEvent(
                time_s=time_s,
                gs_id=gs.gs_id,
                from_sat_id=current_sat_id,
                to_sat_id=best_sat.sat_id,
                reason=reason,
                elevation_deg=best_el,
                remaining_visibility_s=best_remaining,
            )

        return None

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_best(
        self,
        gs: GroundStation,
        visible: List[Tuple[Satellite, float, float]],
        time_s: float,
    ) -> Optional[Tuple[Satellite, float, float]]:
        """Pick the best candidate based on the active strategy."""
        if not visible:
            return None

        if self.strategy == "best_elevation":
            return max(visible, key=lambda v: v[1])
        elif self.strategy == "longest_visibility":
            return max(visible, key=lambda v: v[2])
        elif self.strategy == "best_snr":
            return max(visible, key=lambda v: self._compute_snr(gs, v[0], time_s))
        else:
            return visible[0]

    def _compute_snr(self, gs: GroundStation, sat: Satellite, time_s: float) -> float:
        """Compute SNR for a ground-to-satellite link."""
        slant = gs.slant_range_km(sat.position_eci, time_s)
        el = gs.elevation_angle(sat.position_eci, time_s)
        if el < 1.0:
            return -999.0
        result = self.link_calc.ground_to_sat(slant, el)
        return result.snr_db

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limited(self, gs_id: int, time_s: float) -> bool:
        """Check if the station has exceeded the handover rate limit."""
        recent = self._recent_handovers[gs_id]
        cutoff = time_s - 60.0
        recent[:] = [t for t in recent if t > cutoff]
        return len(recent) >= self.max_rate

    def _record_handover_time(self, gs_id: int, time_s: float) -> None:
        self._recent_handovers[gs_id].append(time_s)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def total_handovers(self) -> int:
        return len([e for e in self.events if e.from_sat_id is not None])

    def handovers_per_station(self) -> Dict[int, int]:
        counts: Dict[int, int] = defaultdict(int)
        for e in self.events:
            if e.from_sat_id is not None:
                counts[e.gs_id] += 1
        return dict(counts)
