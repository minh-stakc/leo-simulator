"""
Traffic generation models.

Generates network traffic between ground station pairs using configurable
arrival patterns: Poisson, constant bit-rate (CBR), or bursty.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import TRAFFIC
from constellation.ground_station import GroundStation


class TrafficGenerator:
    """Generates traffic demands between ground station pairs.

    Each call to :meth:`generate` returns a list of (source_gs_id, dest_gs_id,
    packet_size_bytes) tuples representing packets created during the time step.
    """

    def __init__(
        self,
        ground_stations: List[GroundStation],
        model: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.ground_stations = ground_stations
        self.model = model or TRAFFIC["model"]
        self._rng = np.random.default_rng(seed or 42)

        # Build list of valid station pairs
        gs_ids = [gs.gs_id for gs in ground_stations]
        self.pairs: List[Tuple[int, int]] = []
        for i in range(len(gs_ids)):
            for j in range(i + 1, len(gs_ids)):
                self.pairs.append((gs_ids[i], gs_ids[j]))

        # Assign flows to pairs round-robin
        self.num_flows = min(TRAFFIC["num_flows"], len(self.pairs))
        self.flow_pairs = [
            self.pairs[i % len(self.pairs)] for i in range(self.num_flows)
        ]

        # Bursty state
        self._burst_timer = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, time_s: float, dt_s: float
    ) -> List[Tuple[int, int, int]]:
        """Generate packets for the interval [time_s, time_s + dt_s).

        Returns
        -------
        list of (src_gs_id, dst_gs_id, size_bytes)
        """
        if self.model == "poisson":
            return self._poisson(dt_s)
        elif self.model == "cbr":
            return self._cbr(dt_s)
        elif self.model == "bursty":
            return self._bursty(time_s, dt_s)
        else:
            raise ValueError(f"Unknown traffic model: {self.model}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _poisson(self, dt_s: float) -> List[Tuple[int, int, int]]:
        """Poisson arrival process: each flow generates packets independently."""
        packets = []
        rate = TRAFFIC["mean_arrival_rate_hz"] / self.num_flows
        size = TRAFFIC["packet_size_bytes"]

        for src, dst in self.flow_pairs:
            n = self._rng.poisson(rate * dt_s)
            for _ in range(n):
                packets.append((src, dst, size))

        return packets

    def _cbr(self, dt_s: float) -> List[Tuple[int, int, int]]:
        """Constant bit-rate: fixed number of packets per interval per flow."""
        packets = []
        size = TRAFFIC["packet_size_bytes"]
        rate_bps = TRAFFIC["cbr_rate_mbps"] * 1e6
        packets_per_step = int((rate_bps * dt_s) / (size * 8))
        pkts_per_flow = max(1, packets_per_step // self.num_flows)

        for src, dst in self.flow_pairs:
            for _ in range(pkts_per_flow):
                packets.append((src, dst, size))

        return packets

    def _bursty(
        self, time_s: float, dt_s: float
    ) -> List[Tuple[int, int, int]]:
        """Bursty traffic: periodic bursts of packets."""
        packets = []
        interval = TRAFFIC["burst_interval_s"]
        burst_size = TRAFFIC["burst_size"]
        size = TRAFFIC["packet_size_bytes"]

        # Check if a burst should fire during this interval
        step_start = time_s
        step_end = time_s + dt_s

        burst_time = self._burst_timer
        while burst_time < step_end:
            if burst_time >= step_start:
                for src, dst in self.flow_pairs:
                    for _ in range(burst_size):
                        packets.append((src, dst, size))
            burst_time += interval

        self._burst_timer = burst_time

        return packets
