# LEO Satellite Network Simulator

A Python-based simulator for Low Earth Orbit (LEO) satellite constellations, modeling
orbital mechanics, ground station connectivity, satellite handovers, and network
performance metrics.

## Features

- **Orbital Mechanics**: Keplerian orbit propagation with J2 perturbation correction
  and optional SGP4 propagation via TLE data.
- **Constellation Modeling**: Walker-Delta and Walker-Star constellation patterns with
  configurable altitude, inclination, number of planes, and satellites per plane.
- **Ground Station Connectivity**: Visibility window calculation using elevation angle
  masks and line-of-sight geometry.
- **Satellite Handover**: Predictive handover logic that selects the best satellite
  based on elevation, remaining visibility time, and link quality.
- **Network Topology**: Dynamic graph-based topology that updates as satellites move,
  with inter-satellite links (ISLs) and ground-to-satellite links.
- **Link Budget**: Free-space path loss, atmospheric attenuation, antenna gain, and
  SNR/capacity estimation.
- **Traffic Simulation**: Configurable traffic generators (Poisson, constant bit-rate,
  bursty) with congestion modeling.
- **Performance Metrics**: Latency, packet loss ratio, throughput, jitter, and handover
  frequency measurement.
- **Visualization**: Orbit ground tracks, coverage maps, network topology snapshots,
  and time-series metric plots.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py
```

This runs the default simulation: a Walker-Star constellation at 550 km altitude with
72 satellites across 6 orbital planes, evaluated against 5 ground stations over a
90-minute window.

## Configuration

Edit `config.py` to adjust simulation parameters:

- Constellation geometry (altitude, inclination, satellite count)
- Ground station locations
- Simulation duration and time step
- Traffic model parameters
- Link budget parameters

## Project Structure

```
leo_simulator/
  config.py              - Simulation parameters
  main.py                - Entry point
  constellation/
    orbit.py             - Orbital mechanics
    satellite.py         - Satellite model
    ground_station.py    - Ground station model
    visibility.py        - Visibility calculations
  network/
    topology.py          - Dynamic network topology
    handover.py          - Handover logic
    routing.py           - Packet routing
    link_budget.py       - Link budget calculations
  simulation/
    engine.py            - Discrete event simulation engine
    traffic.py           - Traffic generation models
    metrics.py           - Performance measurement
  visualization/
    plots.py             - Plotting utilities
```

## License

MIT
