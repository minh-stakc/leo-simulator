"""
Performance metrics collection and reporting.

Tracks latency, packet loss ratio, throughput, jitter, and handover statistics
throughout a simulation run.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from network.routing import Packet
from network.handover import HandoverEvent


@dataclass
class TopologySnapshot:
    time_s: float
    num_isl: int
    num_gsl: int


class MetricsCollector:
    """Collects and summarises network performance metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all collected data."""
        # Packet-level records
        self._latencies_ms: List[float] = []
        self._delivered: int = 0
        self._dropped: int = 0
        self._total_bytes_delivered: int = 0
        self._drop_reasons: Dict[str, int] = defaultdict(int)

        # Per-flow latency tracking (for jitter)
        self._flow_latencies: Dict[int, List[float]] = defaultdict(list)

        # Handover records
        self._handover_events: List[HandoverEvent] = []

        # Topology snapshots
        self._topology_snapshots: List[TopologySnapshot] = []

        # Time tracking
        self._first_packet_time: Optional[float] = None
        self._last_packet_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_packet(self, packet: Packet) -> None:
        """Record the outcome of a single packet."""
        if self._first_packet_time is None:
            self._first_packet_time = packet.created_at_s

        self._last_packet_time = packet.created_at_s

        if packet.dropped:
            self._dropped += 1
            self._drop_reasons[packet.drop_reason] += 1
        else:
            latency_ms = (packet.delivered_at_s - packet.created_at_s) * 1000.0
            self._latencies_ms.append(latency_ms)
            self._delivered += 1
            self._total_bytes_delivered += packet.size_bytes
            self._flow_latencies[packet.flow_id].append(latency_ms)

    def record_handover(self, event: HandoverEvent) -> None:
        """Record a handover event."""
        self._handover_events.append(event)

    def record_topology_snapshot(
        self, time_s: float, num_isl: int, num_gsl: int
    ) -> None:
        """Record a topology state snapshot."""
        self._topology_snapshots.append(
            TopologySnapshot(time_s=time_s, num_isl=num_isl, num_gsl=num_gsl)
        )

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarise(self) -> Dict:
        """Compute summary statistics.

        Returns
        -------
        dict
            Keys:
            - ``mean_latency_ms``, ``median_latency_ms``, ``p95_latency_ms``,
              ``p99_latency_ms``
            - ``packet_loss_ratio``
            - ``throughput_mbps``
            - ``mean_jitter_ms``
            - ``total_packets``, ``delivered``, ``dropped``
            - ``drop_reasons``
            - ``handover_count``
            - ``topology_stats``
        """
        total = self._delivered + self._dropped
        latencies = np.array(self._latencies_ms) if self._latencies_ms else np.array([0.0])

        # Throughput
        if self._first_packet_time is not None and self._last_packet_time is not None:
            duration_s = max(self._last_packet_time - self._first_packet_time, 1.0)
        else:
            duration_s = 1.0
        throughput_mbps = (self._total_bytes_delivered * 8) / (duration_s * 1e6)

        # Jitter (mean inter-packet delay variation per flow)
        jitter_values = []
        for flow_id, lats in self._flow_latencies.items():
            if len(lats) > 1:
                arr = np.array(lats)
                jitter_values.append(np.mean(np.abs(np.diff(arr))))
        mean_jitter = float(np.mean(jitter_values)) if jitter_values else 0.0

        # Topology stats
        if self._topology_snapshots:
            isl_counts = [s.num_isl for s in self._topology_snapshots]
            gsl_counts = [s.num_gsl for s in self._topology_snapshots]
            topo_stats = {
                "mean_isl": float(np.mean(isl_counts)),
                "mean_gsl": float(np.mean(gsl_counts)),
                "min_gsl": int(np.min(gsl_counts)),
                "max_gsl": int(np.max(gsl_counts)),
            }
        else:
            topo_stats = {}

        return {
            "total_packets": total,
            "delivered": self._delivered,
            "dropped": self._dropped,
            "packet_loss_ratio": self._dropped / max(total, 1),
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "throughput_mbps": throughput_mbps,
            "mean_jitter_ms": mean_jitter,
            "drop_reasons": dict(self._drop_reasons),
            "handover_count": len(
                [e for e in self._handover_events if e.from_sat_id is not None]
            ),
            "topology_stats": topo_stats,
        }

    def get_latency_timeseries(self) -> np.ndarray:
        """Return raw latency samples as an array."""
        return np.array(self._latencies_ms)

    def get_topology_timeseries(self) -> Dict[str, np.ndarray]:
        """Return topology counts over time."""
        if not self._topology_snapshots:
            return {"time": np.array([]), "isl": np.array([]), "gsl": np.array([])}
        return {
            "time": np.array([s.time_s for s in self._topology_snapshots]),
            "isl": np.array([s.num_isl for s in self._topology_snapshots]),
            "gsl": np.array([s.num_gsl for s in self._topology_snapshots]),
        }
