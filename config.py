"""
Simulation configuration parameters for the LEO Satellite Network Simulator.

All physical constants use SI units unless otherwise noted.
Angles are in degrees for configuration, converted to radians internally.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0  # Mean Earth radius [km]
EARTH_MU = 3.986004418e14  # Standard gravitational parameter [m^3/s^2]
EARTH_J2 = 1.08263e-3  # J2 zonal harmonic coefficient
EARTH_ROTATION_RATE = 7.2921159e-5  # Earth rotation rate [rad/s]
SPEED_OF_LIGHT = 299792.458  # Speed of light [km/s]
BOLTZMANN_DB = -228.6  # Boltzmann constant [dBW/K/Hz]

# ---------------------------------------------------------------------------
# Constellation parameters  (Walker-Star default: Starlink-like shell)
# ---------------------------------------------------------------------------
CONSTELLATION = {
    "name": "LEO-Sim-Alpha",
    "num_planes": 6,  # Number of orbital planes
    "sats_per_plane": 12,  # Satellites per plane
    "altitude_km": 550.0,  # Orbital altitude above Earth surface [km]
    "inclination_deg": 53.0,  # Orbital inclination [deg]
    "walker_phase_offset": 1,  # Phase offset factor (F in Walker notation)
    "pattern": "star",  # 'star' or 'delta'
}

NUM_SATELLITES = CONSTELLATION["num_planes"] * CONSTELLATION["sats_per_plane"]

# ---------------------------------------------------------------------------
# Ground station locations  (lat, lon in degrees, altitude in km ASL)
# ---------------------------------------------------------------------------
GROUND_STATIONS = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "alt_km": 0.01},
    {"name": "London", "lat": 51.5074, "lon": -0.1278, "alt_km": 0.011},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "alt_km": 0.04},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "alt_km": 0.058},
    {"name": "Sao Paulo", "lat": -23.5505, "lon": -46.6333, "alt_km": 0.76},
]

# ---------------------------------------------------------------------------
# Antenna / link-budget parameters
# ---------------------------------------------------------------------------
LINK_BUDGET = {
    "frequency_ghz": 12.0,  # Ku-band downlink frequency [GHz]
    "sat_tx_power_dbw": 10.0,  # Satellite transmit power [dBW]
    "sat_antenna_gain_dbi": 32.0,  # Satellite antenna gain [dBi]
    "gs_antenna_gain_dbi": 34.0,  # Ground station antenna gain [dBi]
    "gs_system_noise_temp_k": 290.0,  # System noise temperature [K]
    "atmospheric_loss_db": 0.5,  # Clear-sky atmospheric loss [dB]
    "pointing_loss_db": 0.3,  # Antenna pointing loss [dB]
    "rain_margin_db": 3.0,  # Rain fade margin [dB]
    "bandwidth_mhz": 250.0,  # Channel bandwidth [MHz]
    "min_elevation_deg": 25.0,  # Minimum elevation for link [deg]
    "isl_data_rate_gbps": 5.0,  # Inter-satellite link capacity [Gbps]
}

# ---------------------------------------------------------------------------
# Inter-satellite link (ISL) parameters
# ---------------------------------------------------------------------------
ISL = {
    "max_range_km": 5000.0,  # Maximum ISL range [km]
    "intra_plane": True,  # Connect to neighbors in same plane
    "inter_plane": True,  # Connect to nearest sats in adjacent planes
}

# ---------------------------------------------------------------------------
# Handover parameters
# ---------------------------------------------------------------------------
HANDOVER = {
    "min_elevation_deg": 25.0,  # Minimum elevation to consider [deg]
    "hysteresis_deg": 3.0,  # Hysteresis margin [deg]
    "min_remaining_visibility_s": 30.0,  # Min remaining visibility [s]
    "handover_delay_ms": 50.0,  # Handover execution latency [ms]
    "max_handovers_per_min": 4,  # Rate limit on handovers
    "strategy": "best_elevation",  # 'best_elevation', 'longest_visibility', 'best_snr'
}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIMULATION = {
    "duration_s": 5400.0,  # Total simulation time [s] (90 min)
    "time_step_s": 10.0,  # Position update interval [s]
    "start_epoch": "2026-03-20T00:00:00",  # Simulation start (ISO 8601)
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# Traffic generation parameters
# ---------------------------------------------------------------------------
TRAFFIC = {
    "model": "poisson",  # 'poisson', 'cbr', 'bursty'
    "mean_arrival_rate_hz": 50.0,  # Packets per second (Poisson)
    "packet_size_bytes": 1400,  # Payload size [bytes]
    "cbr_rate_mbps": 10.0,  # Constant bit-rate [Mbps]
    "burst_size": 20,  # Packets per burst
    "burst_interval_s": 0.5,  # Time between bursts [s]
    "num_flows": 10,  # Number of concurrent traffic flows
    "congestion_levels": [0.0, 0.25, 0.5, 0.75, 0.95],  # Load factors to test
}

# ---------------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------------
VISUALIZATION = {
    "dpi": 150,
    "figsize": (14, 8),
    "ground_track_duration_min": 90,
    "output_dir": "output",
}
