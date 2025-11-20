"""
Discrete event simulation engine.

Orchestrates the time-stepped simulation loop: propagating orbits, updating
the network topology, executing handovers, routing traffic, and collecting
performance metrics.
"""

from __future__ import annotations

import logging
import time as wall_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import SIMULATION, CONSTELLATION, TRAFFIC
from constellation.satellite import Satellite, create_constellation
from constellation.ground_station import GroundStation, create_ground_stations
from constellation.visibility import VisibilityCalculator
from network.topology import TopologyManager
from network.handover import HandoverManager
from network.routing import Router, Packet
from network.link_budget import LinkBudgetCalculator
from simulation.traffic import TrafficGenerator
from simulation.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Snapshot of the simulation at a single time step."""

    time_s: float = 0.0
    step: int = 0
    num_visible_links: int = 0
    num_isl_links: int = 0
    active_handovers: int = 0
    packets_sent: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0


class SimulationEngine:
    """Main simulation engine for the LEO satellite network.

    Usage::

        engine = SimulationEngine()
        engine.setup()
        results = engine.run()
    """

    def __init__(
        self,
        duration_s: Optional[float] = None,
        time_step_s: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.duration = duration_s or SIMULATION["duration_s"]
        self.time_step = time_step_s or SIMULATION["time_step_s"]
        self.seed = seed or SIMULATION["random_seed"]

        # Components (initialised in setup())
        self.satellites: List[Satellite] = []
        self.ground_stations: List[GroundStation] = []
        self.visibility_calc: Optional[VisibilityCalculator] = None
        self.topology: Optional[TopologyManager] = None
        self.handover_mgr: Optional[HandoverManager] = None
        self.router: Optional[Router] = None
        self.traffic_gen: Optional[TrafficGenerator] = None
        self.metrics: Optional[MetricsCollector] = None
        self.link_calc: Optional[LinkBudgetCalculator] = None

        # State history
        self.states: List[SimulationState] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise all simulation components."""
        logger.info("Setting up simulation...")
        np.random.seed(self.seed)

        # Create constellation and ground stations
        self.satellites = create_constellation()
        self.ground_stations = create_ground_stations()
        logger.info(
            "Created %d satellites in %d planes, %d ground stations",
            len(self.satellites),
            CONSTELLATION["num_planes"],
            len(self.ground_stations),
        )

        # Link budget calculator
        self.link_calc = LinkBudgetCalculator()

        # Visibility calculator
        self.visibility_calc = VisibilityCalculator(
            self.satellites, self.ground_stations
        )

        # Network topology
        self.topology = TopologyManager(
            self.satellites, self.ground_stations, self.link_calc
        )

        # Handover manager
        self.handover_mgr = HandoverManager(
            self.satellites, self.ground_stations, self.visibility_calc, self.link_calc
        )

        # Router
        self.router = Router(self.topology, weight_mode="delay")

        # Traffic generator
        self.traffic_gen = TrafficGenerator(self.ground_stations)

        # Metrics collector
        self.metrics = MetricsCollector()

        logger.info("Setup complete.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, congestion_levels: Optional[List[float]] = None) -> Dict:
        """Execute the simulation.

        Parameters
        ----------
        congestion_levels : list of float, optional
            Load factors to sweep.  Defaults to ``config.TRAFFIC["congestion_levels"]``.

        Returns
        -------
        dict
            Summary results keyed by congestion level.
        """
        if congestion_levels is None:
            congestion_levels = TRAFFIC["congestion_levels"]

        all_results: Dict[float, Dict] = {}

        for cong in congestion_levels:
            logger.info("=== Running congestion level %.0f%% ===", cong * 100)
            result = self._run_single(cong)
            all_results[cong] = result

        return all_results

    def _run_single(self, congestion_factor: float) -> Dict:
        """Run one full simulation pass at a given congestion level."""
        self.metrics.reset()
        self.states.clear()

        # Reset handover state
        for gs in self.ground_stations:
            self.handover_mgr.serving[gs.gs_id] = None
        self.handover_mgr.events.clear()
        self.handover_mgr._recent_handovers.clear()

        num_steps = int(self.duration / self.time_step)
        t_start = wall_time.time()
        packet_id_counter = 0

        for step in range(num_steps):
            t = step * self.time_step
            state = SimulationState(time_s=t, step=step)

            # 1. Propagate all satellite positions
            for sat in self.satellites:
                sat.update(t)

            # 2. Update network topology
            self.topology.update(t)
            self.router.invalidate_cache()
            state.num_isl_links = self.topology.num_isl_edges()
            state.num_visible_links = self.topology.num_gsl_edges()

            # 3. Execute handovers
            ho_events = self.handover_mgr.update(t)
            state.active_handovers = len(ho_events)
            for event in ho_events:
                self.metrics.record_handover(event)

            # 4. Generate and route traffic
            packets = self.traffic_gen.generate(t, self.time_step)
            for pkt_src, pkt_dst, pkt_size in packets:
                # Map ground stations to their serving satellite for routing
                src_node = TopologyManager.gs_node(pkt_src)
                dst_node = TopologyManager.gs_node(pkt_dst)

                packet = Packet(
                    packet_id=packet_id_counter,
                    flow_id=pkt_src * 100 + pkt_dst,
                    source=src_node,
                    destination=dst_node,
                    size_bytes=pkt_size,
                    created_at_s=t,
                )
                packet_id_counter += 1

                packet = self.router.forward_packet(packet, t, congestion_factor)
                self.metrics.record_packet(packet)

                if packet.dropped:
                    state.packets_dropped += 1
                else:
                    state.packets_delivered += 1
                state.packets_sent += 1

            # 5. Record topology metrics
            self.metrics.record_topology_snapshot(
                t, state.num_isl_links, state.num_visible_links
            )

            self.states.append(state)

            if step % 50 == 0:
                logger.info(
                    "  t=%6.0fs  ISLs=%3d  GSLs=%3d  HOs=%d  pkts=%d/%d",
                    t,
                    state.num_isl_links,
                    state.num_visible_links,
                    state.active_handovers,
                    state.packets_delivered,
                    state.packets_sent,
                )

        elapsed = wall_time.time() - t_start
        logger.info("Simulation completed in %.1f seconds (wall time).", elapsed)

        summary = self.metrics.summarise()
        summary["congestion_factor"] = congestion_factor
        summary["total_handovers"] = self.handover_mgr.total_handovers()
        summary["handovers_per_station"] = self.handover_mgr.handovers_per_station()
        summary["wall_time_s"] = elapsed

        return summary

    # ------------------------------------------------------------------
    # Quick single-step preview (for visualisation)
    # ------------------------------------------------------------------

    def snapshot(self, time_s: float) -> SimulationState:
        """Compute a single time-step snapshot without traffic."""
        for sat in self.satellites:
            sat.update(time_s)
        self.topology.update(time_s)

        state = SimulationState(time_s=time_s)
        state.num_isl_links = self.topology.num_isl_edges()
        state.num_visible_links = self.topology.num_gsl_edges()
        return state
