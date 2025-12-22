#!/usr/bin/env python3
"""
LEO Satellite Network Simulator -- main entry point.

Runs the full simulation pipeline:
  1. Build constellation and ground stations
  2. Simulate orbital propagation, handovers, and traffic routing
  3. Sweep across congestion levels to measure latency, loss, throughput
  4. Generate visualisation outputs

Usage:
    python main.py [--fast]

    --fast  Run a shorter simulation (900 s instead of 5400 s) for quick testing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SIMULATION, TRAFFIC, CONSTELLATION
from simulation.engine import SimulationEngine
from visualization.plots import Plotter


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_summary(results: dict) -> None:
    """Pretty-print simulation results to the console."""
    print("\n" + "=" * 72)
    print("  LEO SATELLITE NETWORK SIMULATION RESULTS")
    print("=" * 72)
    print(
        f"  Constellation: {CONSTELLATION['name']}  "
        f"({CONSTELLATION['num_planes']} planes x "
        f"{CONSTELLATION['sats_per_plane']} sats = "
        f"{CONSTELLATION['num_planes'] * CONSTELLATION['sats_per_plane']} total)"
    )
    print(
        f"  Altitude: {CONSTELLATION['altitude_km']} km   "
        f"Inclination: {CONSTELLATION['inclination_deg']} deg"
    )
    print("-" * 72)

    for cong, data in sorted(results.items()):
        print(f"\n  Congestion Level: {cong:.0%}")
        print(f"    Packets sent / delivered / dropped: "
              f"{data['total_packets']} / {data['delivered']} / {data['dropped']}")
        print(f"    Packet loss ratio:   {data['packet_loss_ratio']:.4f}")
        print(f"    Mean latency:        {data['mean_latency_ms']:.2f} ms")
        print(f"    Median latency:      {data['median_latency_ms']:.2f} ms")
        print(f"    P95 latency:         {data['p95_latency_ms']:.2f} ms")
        print(f"    P99 latency:         {data['p99_latency_ms']:.2f} ms")
        print(f"    Throughput:          {data['throughput_mbps']:.2f} Mbps")
        print(f"    Mean jitter:         {data['mean_jitter_ms']:.2f} ms")
        print(f"    Total handovers:     {data['total_handovers']}")
        if data.get("topology_stats"):
            ts = data["topology_stats"]
            print(f"    Avg ISL links:       {ts.get('mean_isl', 0):.1f}")
            print(f"    Avg GSL links:       {ts.get('mean_gsl', 0):.1f}")
        if data.get("drop_reasons"):
            print(f"    Drop reasons:        {data['drop_reasons']}")
        print(f"    Wall-clock time:     {data.get('wall_time_s', 0):.1f} s")

    print("\n" + "=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="LEO Satellite Network Simulator")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a shorter 15-minute simulation for quick testing",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Custom simulation duration in seconds",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="Custom time step in seconds",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("main")

    # Determine simulation parameters
    duration = args.duration
    if duration is None:
        duration = 900.0 if args.fast else SIMULATION["duration_s"]
    time_step = args.step or SIMULATION["time_step_s"]

    logger.info("Starting LEO Satellite Network Simulator")
    logger.info("Duration: %.0f s   Time step: %.1f s", duration, time_step)

    # Create and set up the simulation engine
    engine = SimulationEngine(duration_s=duration, time_step_s=time_step)
    engine.setup()

    # Run simulation across congestion levels
    results = engine.run()

    # Attach latency samples for CDF plotting
    for cong, data in results.items():
        data["latency_samples"] = engine.metrics.get_latency_timeseries()

    # Print results
    print_summary(results)

    # Generate visualisations
    if not args.no_plots:
        logger.info("Generating plots...")
        plotter = Plotter(output_dir=args.output_dir)

        try:
            path = plotter.plot_ground_tracks(
                engine.satellites, engine.ground_stations, duration_s=duration
            )
            logger.info("Ground tracks: %s", path)
        except Exception as e:
            logger.warning("Failed to generate ground tracks plot: %s", e)

        try:
            path = plotter.plot_coverage_snapshot(
                engine.satellites, engine.ground_stations, time_s=0.0
            )
            logger.info("Coverage snapshot: %s", path)
        except Exception as e:
            logger.warning("Failed to generate coverage plot: %s", e)

        try:
            path = plotter.plot_metrics_vs_congestion(results)
            logger.info("Metrics vs congestion: %s", path)
        except Exception as e:
            logger.warning("Failed to generate metrics plot: %s", e)

        try:
            topo_data = engine.metrics.get_topology_timeseries()
            if len(topo_data["time"]) > 0:
                path = plotter.plot_topology_timeseries(topo_data)
                logger.info("Topology timeseries: %s", path)
        except Exception as e:
            logger.warning("Failed to generate topology plot: %s", e)

        try:
            path = plotter.plot_handover_timeline(
                engine.handover_mgr.events, engine.ground_stations
            )
            logger.info("Handover timeline: %s", path)
        except Exception as e:
            logger.warning("Failed to generate handover plot: %s", e)

        try:
            path = plotter.plot_latency_cdf(results)
            logger.info("Latency CDF: %s", path)
        except Exception as e:
            logger.warning("Failed to generate latency CDF plot: %s", e)

    logger.info("Simulation complete. Output in '%s/'", args.output_dir)


if __name__ == "__main__":
    main()
