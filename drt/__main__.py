"""Benchmark: on-demand dynamic routing vs. a fixed bus line with the same fleet.

    python -m drt --buses 8 --rate 60 --hours 2 --forecast auto
"""
from __future__ import annotations

import argparse
import os
import time

from .models import Config
from .simulation import FixedLineBaseline, Simulation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buses", type=int, default=8)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--rate", type=float, default=60, help="average trip requests per hour (peaks ~2x)")
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--city", default=os.getenv("DRT_CITY") or None,
                    help='OpenStreetMap place, e.g. "Union Square, San Francisco, USA" '
                         '(default: DRT_CITY from .env). Use --city grid for the synthetic grid.')
    ap.add_argument("--radius", type=float, default=float(os.getenv("DRT_RADIUS_M", 2000)),
                    help="metres around the city point")
    ap.add_argument("--forecast", default="auto", choices=["auto", "watsonx", "naive", "off"],
                    help="demand forecast for positioning idle buses (auto = watsonx if configured)")
    a = ap.parse_args()
    city = None if (a.city or "").lower() in ("", "grid") else a.city

    cfg = Config(city=city, radius_m=a.radius, n_buses=a.buses, capacity=a.capacity,
                 demand_per_hour=a.rate, seed=a.seed, forecast_source=a.forecast)
    t = time.perf_counter()
    sim = Simulation(cfg)
    print(f"City: {sim.net.name} - {len(sim.net.G)} intersections, "
          f"{len(sim.stations.nodes)} stations, {len(sim.landmarks)} named places "
          f"(loaded in {time.perf_counter() - t:.1f}s)")
    t = time.perf_counter()
    sim.run(a.hours * 3600)
    sim.drain()
    m = sim.metrics()
    elapsed = time.perf_counter() - t

    base = FixedLineBaseline(sim)
    done = [r for r in sim.requests.values() if r.status == "done"]
    b_trips = [base.trip_time(r.origin, r.dest) for r in done]
    b_trip = sum(x[0] for x in b_trips) / len(b_trips) if b_trips else 0
    b_walk = sum(x[1] for x in b_trips) / len(b_trips) if b_trips else 0
    faster = sum(1 for r, (bt, _) in zip(done, b_trips)
                 if r.t_dropoff + r.walk_from_m / cfg.walk_speed - r.t_request < bt)

    mins = lambda s: f"{s / 60:5.1f} min"
    print(f"\nSimulated {a.hours:g} h, {a.buses} buses x {a.capacity} seats, "
          f"{a.rate:g} requests/h  (computed in {elapsed:.1f}s)\n")
    print(f"Requests: {m['requests']}   status: {m['status']}   served: {m['served_pct']:.1f}%")
    print(f"Re-optimisation moved {m['reassignments']} customers between buses/stations\n")
    print(f"{'':28}{'On-demand (DRT)':>18}{'Fixed line':>14}")
    print(f"{'Avg door-to-door trip':28}{mins(m['avg_trip_s']):>18}{mins(b_trip):>14}")
    print(f"{'Avg walking':28}{m['avg_walk_m']:>15.0f} m{b_walk:>12.0f} m")
    print(f"{'Avg wait at station':28}{mins(m['avg_wait_s']):>18}{mins(base.headway / 2):>14}")
    print(f"{'Avg in-bus time':28}{mins(m['avg_ride_s']):>18}")
    print(f"{'Avg detour vs. direct car':28}{m['avg_detour']:>17.2f}x")
    print(f"{'Avg passengers on board':28}{m['avg_occupancy']:>18.2f}")
    print(f"{'Fleet km':28}{m['fleet_km']:>18.0f}")
    if done:
        print(f"\nDRT was faster than the fixed line for {100 * faster / len(done):.0f}% of trips.")

    mae = m["forecast_mae"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"\nForecast: {m['forecast_source']}   error per zone per 15 min "
          f"(MAE): watsonx {fmt(mae['watsonx'])}, seasonal-naive {fmt(mae['naive'])}")
    if m["forecast_error"]:
        print(f"  last watsonx error: {m['forecast_error']}")
    print(f"Zone coverage (bus within {cfg.coverage_s / 60:.0f} min): {m['coverage_pct']:.0f}% of checks")
    print(f"Driver breaks taken: {m['breaks_taken']}   bus states at end: {m['buses']}")


if __name__ == "__main__":
    main()
