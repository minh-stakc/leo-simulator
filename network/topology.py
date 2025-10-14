"""
Dynamic network topology manager.

Maintains a time-varying graph of satellite nodes, ground station nodes,
inter-satellite links (ISLs), and ground-to-satellite links (GSLs).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config import ISL, LINK_BUDGET, CONSTELLATION
from constellation.satellite import Satellite
from constellation.ground_station import GroundStation
from network.link_budget import LinkBudgetCalculator


class TopologyManager:
    """Builds and updates the network topology graph each time step."""

    def __init__(
        self,
        satellites: List[Satellite],
        ground_stations: List[GroundStation],
        link_calc: Optional[LinkBudgetCalculator] = None,
    ) -> None:
        self.satellites = {s.sat_id: s for s in satellites}
        self.ground_stations = {gs.gs_id: gs for gs in ground_stations}
        self.link_calc = link_calc or LinkBudgetCalculator()
        self.graph = nx.Graph()

        self._num_planes = CONSTELLATION["num_planes"]
        self._sats_per_plane = CONSTELLATION["sats_per_plane"]

    # ------------------------------------------------------------------
    # Node naming conventions
    # ------------------------------------------------------------------

    @staticmethod
    def sat_node(sat_id: int) -> str:
        return f"SAT_{sat_id}"

    @staticmethod
    def gs_node(gs_id: int) -> str:
        return f"GS_{gs_id}"

    # ------------------------------------------------------------------
    # Topology update
    # ------------------------------------------------------------------

    def update(self, time_s: float) -> nx.Graph:
        """Rebuild the full topology for the given simulation time.

        Returns the updated NetworkX graph.
        """
        self.graph.clear()
        self._add_satellite_nodes()
        self._add_ground_station_nodes()
        self._add_intra_plane_isls(time_s)
        self._add_inter_plane_isls(time_s)
        self._add_ground_links(time_s)
        return self.graph

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def _add_satellite_nodes(self) -> None:
        for sat in self.satellites.values():
            self.graph.add_node(
                self.sat_node(sat.sat_id),
                type="satellite",
                sat_id=sat.sat_id,
                plane=sat.plane_id,
                lat=sat.lat_deg,
                lon=sat.lon_deg,
                alt=sat.alt_km,
            )

    def _add_ground_station_nodes(self) -> None:
        for gs in self.ground_stations.values():
            self.graph.add_node(
                self.gs_node(gs.gs_id),
                type="ground_station",
                gs_id=gs.gs_id,
                name=gs.name,
                lat=gs.lat_deg,
                lon=gs.lon_deg,
            )

    # ------------------------------------------------------------------
    # Inter-satellite links
    # ------------------------------------------------------------------

    def _add_intra_plane_isls(self, time_s: float) -> None:
        """Connect each satellite to its immediate neighbours in the same plane."""
        if not ISL["intra_plane"]:
            return

        for plane_id in range(self._num_planes):
            plane_sats = [
                s for s in self.satellites.values() if s.plane_id == plane_id
            ]
            plane_sats.sort(key=lambda s: s.plane_index)

            for i in range(len(plane_sats)):
                s1 = plane_sats[i]
                s2 = plane_sats[(i + 1) % len(plane_sats)]
                dist = s1.distance_to(s2.position_eci)
                if dist <= ISL["max_range_km"]:
                    link = self.link_calc.inter_satellite(dist)
                    self.graph.add_edge(
                        self.sat_node(s1.sat_id),
                        self.sat_node(s2.sat_id),
                        type="isl_intra",
                        distance_km=dist,
                        delay_ms=link.propagation_delay_ms,
                        capacity_mbps=link.capacity_mbps,
                        snr_db=link.snr_db,
                    )

    def _add_inter_plane_isls(self, time_s: float) -> None:
        """Connect each satellite to the nearest satellite in each adjacent plane."""
        if not ISL["inter_plane"]:
            return

        for sat in self.satellites.values():
            for adj_plane_offset in [-1, 1]:
                adj_plane = (sat.plane_id + adj_plane_offset) % self._num_planes
                adj_sats = [
                    s for s in self.satellites.values() if s.plane_id == adj_plane
                ]
                if not adj_sats:
                    continue

                nearest = min(adj_sats, key=lambda s: sat.distance_to(s.position_eci))
                dist = sat.distance_to(nearest.position_eci)

                if dist > ISL["max_range_km"]:
                    continue

                n1 = self.sat_node(sat.sat_id)
                n2 = self.sat_node(nearest.sat_id)

                if self.graph.has_edge(n1, n2):
                    continue

                link = self.link_calc.inter_satellite(dist)
                self.graph.add_edge(
                    n1, n2,
                    type="isl_inter",
                    distance_km=dist,
                    delay_ms=link.propagation_delay_ms,
                    capacity_mbps=link.capacity_mbps,
                    snr_db=link.snr_db,
                )

    # ------------------------------------------------------------------
    # Ground-to-satellite links
    # ------------------------------------------------------------------

    def _add_ground_links(self, time_s: float) -> None:
        """Add edges from ground stations to all visible satellites."""
        min_el = LINK_BUDGET["min_elevation_deg"]

        for gs in self.ground_stations.values():
            for sat in self.satellites.values():
                el = gs.elevation_angle(sat.position_eci, time_s)
                if el < min_el:
                    continue

                slant = gs.slant_range_km(sat.position_eci, time_s)
                link = self.link_calc.ground_to_sat(slant, el)

                self.graph.add_edge(
                    self.gs_node(gs.gs_id),
                    self.sat_node(sat.sat_id),
                    type="gsl",
                    distance_km=slant,
                    delay_ms=link.propagation_delay_ms,
                    capacity_mbps=link.capacity_mbps,
                    snr_db=link.snr_db,
                    elevation_deg=el,
                )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def connected_ground_stations(self) -> List[int]:
        """Return gs_ids that have at least one satellite link."""
        connected = []
        for gs in self.ground_stations.values():
            node = self.gs_node(gs.gs_id)
            if node in self.graph and self.graph.degree(node) > 0:
                connected.append(gs.gs_id)
        return connected

    def get_serving_satellites(self, gs_id: int) -> List[Tuple[int, float]]:
        """Return (sat_id, elevation_deg) for all satellites linked to a station."""
        node = self.gs_node(gs_id)
        result = []
        for neighbor in self.graph.neighbors(node):
            edge = self.graph.edges[node, neighbor]
            if edge.get("type") == "gsl":
                result.append((edge.get("sat_id", self.graph.nodes[neighbor].get("sat_id")),
                                edge.get("elevation_deg", 0.0)))
        # Sort by elevation descending
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def num_isl_edges(self) -> int:
        return sum(
            1
            for _, _, d in self.graph.edges(data=True)
            if d.get("type", "").startswith("isl")
        )

    def num_gsl_edges(self) -> int:
        return sum(
            1
            for _, _, d in self.graph.edges(data=True)
            if d.get("type") == "gsl"
        )
