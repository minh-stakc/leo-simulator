"""
Packet routing through the satellite constellation.

Uses Dijkstra's algorithm on the dynamic topology graph, with configurable
weight metrics (propagation delay, hop count, or inverse capacity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config import SPEED_OF_LIGHT
from network.topology import TopologyManager

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """A computed route through the network."""

    source: str
    destination: str
    path: List[str]
    total_delay_ms: float
    hop_count: int
    bottleneck_capacity_mbps: float
    valid: bool = True

    @property
    def path_str(self) -> str:
        return " -> ".join(self.path)


@dataclass
class Packet:
    """A network packet traversing the constellation."""

    packet_id: int
    flow_id: int
    source: str
    destination: str
    size_bytes: int
    created_at_s: float
    route: Optional[Route] = None
    delivered_at_s: float = -1.0
    dropped: bool = False
    drop_reason: str = ""


class Router:
    """Routes packets through the satellite network topology.

    Supports three weight modes for shortest-path computation:

    - ``delay``: minimise end-to-end propagation delay.
    - ``hops``: minimise hop count.
    - ``capacity``: maximise bottleneck capacity (uses inverse capacity as weight).
    """

    def __init__(
        self,
        topology: TopologyManager,
        weight_mode: str = "delay",
        queue_limit: int = 500,
    ) -> None:
        self.topology = topology
        self.weight_mode = weight_mode
        self.queue_limit = queue_limit

        # Per-edge queue occupancy (edge_key -> current_queue_size_bytes)
        self.edge_queues: Dict[Tuple[str, str], int] = {}

        # Route cache (source, dest) -> Route  -- invalidated each topology update
        self._route_cache: Dict[Tuple[str, str], Route] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear route cache (call after topology update)."""
        self._route_cache.clear()

    def compute_route(self, source: str, destination: str) -> Route:
        """Find the best route between *source* and *destination* nodes."""
        cache_key = (source, destination)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        graph = self.topology.graph

        if source not in graph or destination not in graph:
            return Route(source, destination, [], 0.0, 0, 0.0, valid=False)

        try:
            weight_key = self._weight_key()
            path = nx.shortest_path(graph, source, destination, weight=weight_key)
        except nx.NetworkXNoPath:
            return Route(source, destination, [], 0.0, 0, 0.0, valid=False)

        total_delay = 0.0
        bottleneck_cap = float("inf")

        for i in range(len(path) - 1):
            edge = graph.edges[path[i], path[i + 1]]
            total_delay += edge.get("delay_ms", 0.0)
            cap = edge.get("capacity_mbps", float("inf"))
            bottleneck_cap = min(bottleneck_cap, cap)

        route = Route(
            source=source,
            destination=destination,
            path=path,
            total_delay_ms=total_delay,
            hop_count=len(path) - 1,
            bottleneck_capacity_mbps=bottleneck_cap,
        )

        self._route_cache[cache_key] = route
        return route

    def forward_packet(
        self,
        packet: Packet,
        current_time_s: float,
        congestion_factor: float = 0.0,
    ) -> Packet:
        """Attempt to route and deliver a packet.

        Parameters
        ----------
        packet : Packet
            The packet to forward.
        current_time_s : float
            Current simulation time [s].
        congestion_factor : float
            Load factor in [0, 1).  Higher values increase queuing delay and
            packet drop probability.

        Returns
        -------
        Packet
            The same packet, updated with delivery time or drop status.
        """
        route = self.compute_route(packet.source, packet.destination)
        packet.route = route

        if not route.valid:
            packet.dropped = True
            packet.drop_reason = "no_route"
            return packet

        # Compute queuing delay based on congestion (M/M/1 approximation)
        if congestion_factor >= 1.0:
            packet.dropped = True
            packet.drop_reason = "congestion_overload"
            return packet

        service_rate = route.bottleneck_capacity_mbps * 1e6 / (packet.size_bytes * 8)
        if service_rate <= 0:
            packet.dropped = True
            packet.drop_reason = "zero_capacity"
            return packet

        rho = min(congestion_factor, 0.99)

        # M/M/1 mean queuing delay per hop
        if rho > 0:
            mean_queue_delay_ms = (1.0 / service_rate) * (rho / (1.0 - rho)) * 1e3
        else:
            mean_queue_delay_ms = 0.0

        total_queue_delay_ms = mean_queue_delay_ms * route.hop_count

        # Random packet drop under high congestion (tail-drop model)
        if congestion_factor > 0.5:
            drop_prob = (congestion_factor - 0.5) * 0.4  # 0% at 0.5 -> 20% at 1.0
            if np.random.random() < drop_prob:
                packet.dropped = True
                packet.drop_reason = "queue_overflow"
                return packet

        # Total latency
        total_latency_ms = route.total_delay_ms + total_queue_delay_ms
        # Add processing delay (0.1 ms per hop)
        total_latency_ms += route.hop_count * 0.1

        packet.delivered_at_s = current_time_s + total_latency_ms / 1000.0
        return packet

    def compute_all_pairs_delay(self) -> Dict[Tuple[str, str], float]:
        """Compute delay for all connected ground-station pairs."""
        gs_nodes = [
            n for n, d in self.topology.graph.nodes(data=True)
            if d.get("type") == "ground_station"
        ]
        delays = {}
        for i, src in enumerate(gs_nodes):
            for dst in gs_nodes[i + 1:]:
                route = self.compute_route(src, dst)
                if route.valid:
                    delays[(src, dst)] = route.total_delay_ms
        return delays

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _weight_key(self) -> str:
        if self.weight_mode == "delay":
            return "delay_ms"
        elif self.weight_mode == "hops":
            return None  # unweighted
        elif self.weight_mode == "capacity":
            return "_inv_capacity"
        return "delay_ms"
