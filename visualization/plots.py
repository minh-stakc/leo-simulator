"""
Visualization utilities.

Generates publication-quality plots of orbital ground tracks, constellation
coverage, network topology snapshots, and time-series performance metrics.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import VISUALIZATION, EARTH_RADIUS_KM, SIMULATION
from constellation.satellite import Satellite
from constellation.ground_station import GroundStation


class Plotter:
    """High-level plotting interface for simulation results."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir or VISUALIZATION["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.dpi = VISUALIZATION["dpi"]
        self.figsize = VISUALIZATION["figsize"]

    # ------------------------------------------------------------------
    # Ground tracks
    # ------------------------------------------------------------------

    def plot_ground_tracks(
        self,
        satellites: List[Satellite],
        ground_stations: List[GroundStation],
        duration_s: Optional[float] = None,
        time_step_s: float = 30.0,
        filename: str = "ground_tracks.png",
    ) -> str:
        """Plot satellite ground tracks on an equirectangular projection.

        Returns the path to the saved figure.
        """
        duration = duration_s or SIMULATION["duration_s"]
        times = np.arange(0, duration, time_step_s)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title("Satellite Ground Tracks")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Plot coastline-like grid
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        # Color by orbital plane
        planes = set(s.plane_id for s in satellites)
        cmap = plt.cm.get_cmap("tab10", len(planes))

        for sat in satellites:
            lats, lons = [], []
            for t in times:
                sat.update(t)
                lats.append(sat.lat_deg)
                lons.append(sat.lon_deg)

            # Split track at longitude wrapping
            lats, lons = np.array(lats), np.array(lons)
            segments = self._split_track(lats, lons)
            color = cmap(sat.plane_id % len(planes))
            for seg_lat, seg_lon in segments:
                ax.plot(seg_lon, seg_lat, color=color, linewidth=0.5, alpha=0.6)

        # Plot ground stations
        for gs in ground_stations:
            ax.plot(
                gs.lon_deg,
                gs.lat_deg,
                "r^",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=5,
            )
            ax.annotate(
                gs.name,
                (gs.lon_deg, gs.lat_deg),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

        # Legend
        plane_patches = [
            mpatches.Patch(color=cmap(p), label=f"Plane {p}") for p in sorted(planes)
        ]
        ax.legend(handles=plane_patches, loc="lower left", fontsize=6, ncol=3)

        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Coverage map
    # ------------------------------------------------------------------

    def plot_coverage_snapshot(
        self,
        satellites: List[Satellite],
        ground_stations: List[GroundStation],
        time_s: float = 0.0,
        filename: str = "coverage_snapshot.png",
    ) -> str:
        """Plot satellite footprints at a single time instant."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(f"Coverage Snapshot at t={time_s:.0f}s")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for sat in satellites:
            sat.update(time_s)
            radius_deg = sat.coverage_half_angle_deg
            circle = plt.Circle(
                (sat.lon_deg, sat.lat_deg),
                radius_deg,
                color="blue",
                alpha=0.08,
                linewidth=0,
            )
            ax.add_patch(circle)
            ax.plot(sat.lon_deg, sat.lat_deg, "b.", markersize=3)

        for gs in ground_stations:
            ax.plot(gs.lon_deg, gs.lat_deg, "r^", markersize=10, zorder=5)
            ax.annotate(
                gs.name,
                (gs.lon_deg, gs.lat_deg),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Network metrics
    # ------------------------------------------------------------------

    def plot_latency_cdf(
        self,
        results: Dict[float, Dict],
        filename: str = "latency_cdf.png",
    ) -> str:
        """Plot CDF of end-to-end latency for each congestion level."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        for cong, data in sorted(results.items()):
            latencies = data.get("latency_samples", None)
            if latencies is None or len(latencies) == 0:
                continue
            sorted_lat = np.sort(latencies)
            cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
            ax.plot(sorted_lat, cdf, label=f"Load={cong:.0%}")

        ax.set_xlabel("End-to-End Latency [ms]")
        ax.set_ylabel("CDF")
        ax.set_title("Latency CDF under Varying Congestion")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_metrics_vs_congestion(
        self,
        results: Dict[float, Dict],
        filename: str = "metrics_vs_congestion.png",
    ) -> str:
        """Bar/line charts showing latency, loss, and throughput vs. congestion."""
        congs = sorted(results.keys())
        mean_lat = [results[c].get("mean_latency_ms", 0) for c in congs]
        p95_lat = [results[c].get("p95_latency_ms", 0) for c in congs]
        loss = [results[c].get("packet_loss_ratio", 0) for c in congs]
        throughput = [results[c].get("throughput_mbps", 0) for c in congs]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=self.dpi)

        # Latency
        x = np.arange(len(congs))
        width = 0.35
        axes[0].bar(x - width / 2, mean_lat, width, label="Mean", color="steelblue")
        axes[0].bar(x + width / 2, p95_lat, width, label="P95", color="coral")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{c:.0%}" for c in congs])
        axes[0].set_xlabel("Congestion Level")
        axes[0].set_ylabel("Latency [ms]")
        axes[0].set_title("Latency vs Congestion")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        # Packet loss
        axes[1].bar(x, [l * 100 for l in loss], color="crimson", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{c:.0%}" for c in congs])
        axes[1].set_xlabel("Congestion Level")
        axes[1].set_ylabel("Packet Loss [%]")
        axes[1].set_title("Packet Loss vs Congestion")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Throughput
        axes[2].plot(
            [f"{c:.0%}" for c in congs], throughput, "go-", linewidth=2, markersize=8
        )
        axes[2].set_xlabel("Congestion Level")
        axes[2].set_ylabel("Throughput [Mbps]")
        axes[2].set_title("Throughput vs Congestion")
        axes[2].grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_topology_timeseries(
        self,
        topo_data: Dict[str, np.ndarray],
        filename: str = "topology_timeseries.png",
    ) -> str:
        """Plot ISL and GSL link counts over time."""
        fig, ax = plt.subplots(figsize=(12, 5), dpi=self.dpi)

        t = topo_data["time"] / 60.0  # convert to minutes
        ax.plot(t, topo_data["isl"], label="ISL links", color="steelblue")
        ax.plot(t, topo_data["gsl"], label="GSL links", color="coral")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Number of Links")
        ax.set_title("Network Topology Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_handover_timeline(
        self,
        handover_events: List,
        ground_stations: List[GroundStation],
        filename: str = "handover_timeline.png",
    ) -> str:
        """Plot handover events on a timeline per ground station."""
        fig, ax = plt.subplots(figsize=(14, 5), dpi=self.dpi)

        gs_names = {gs.gs_id: gs.name for gs in ground_stations}
        gs_ids = sorted(gs_names.keys())

        for event in handover_events:
            if event.from_sat_id is None:
                continue
            y = gs_ids.index(event.gs_id)
            ax.plot(event.time_s / 60.0, y, "rv", markersize=6, alpha=0.7)

        ax.set_yticks(range(len(gs_ids)))
        ax.set_yticklabels([gs_names[gid] for gid in gs_ids])
        ax.set_xlabel("Time [min]")
        ax.set_title("Handover Events")
        ax.grid(True, alpha=0.3, axis="x")

        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_track(
        lats: np.ndarray, lons: np.ndarray, threshold: float = 180.0
    ) -> List:
        """Split a ground track at longitude wrap-around points."""
        segments = []
        seg_lat, seg_lon = [lats[0]], [lons[0]]

        for i in range(1, len(lons)):
            if abs(lons[i] - lons[i - 1]) > threshold:
                if len(seg_lat) > 1:
                    segments.append((np.array(seg_lat), np.array(seg_lon)))
                seg_lat, seg_lon = [lats[i]], [lons[i]]
            else:
                seg_lat.append(lats[i])
                seg_lon.append(lons[i])

        if len(seg_lat) > 1:
            segments.append((np.array(seg_lat), np.array(seg_lon)))

        return segments
