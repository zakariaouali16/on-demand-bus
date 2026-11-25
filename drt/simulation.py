"""Time-stepped city simulation: demand, bus movement, pickups and drop-offs."""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from .demand import DemandModel, Zones
from .dispatcher import Dispatcher
from .fleet import FleetManager
from .forecast import Forecaster
from .models import Bus, Config, Request, Stop
from .network import Node, RoadNetwork
from .stations import StationNetwork

# Named places for the synthetic grid city (grid column, row). Real cities get
# their landmarks from OpenStreetMap instead (hospitals, stations, museums...).
GRID_LANDMARKS: dict[str, tuple[int, int]] = {
    "Central Station": (6, 6),
    "University": (2, 9),
    "Hospital": (9, 9),
    "City Mall": (9, 2),
    "Airport": (11, 0),
    "Old Town": (1, 1),
    "Stadium": (10, 6),
    "Business Park": (4, 3),
}


def build_network(cfg: Config) -> RoadNetwork:
    if cfg.city:
        return RoadNetwork.from_osm(cfg.city, cfg.radius_m)
    net = RoadNetwork.grid(cfg.grid_cols, cfg.grid_rows, cfg.block_m)
    net.landmarks = {name: x * cfg.grid_rows + y for name, (x, y) in GRID_LANDMARKS.items()}
    return net


class Simulation:
    def __init__(self, cfg: Config | None = None, auto_demand: bool = True,
                 blocking_forecast: bool = True):
        self.cfg = cfg = cfg or Config()
        self.rng = random.Random(cfg.seed)
        self.net = build_network(cfg)
        self.landmarks = self.net.landmarks
        self.stations = StationNetwork(self.net, cfg.station_spacing_m)
        self.requests: dict[int, Request] = {}
        self.dispatcher = Dispatcher(self.net, self.stations, self.requests, cfg)
        start = self.rng.sample(self.stations.nodes, cfg.n_buses)
        self.buses = [Bus(i, cfg.capacity, n,
                          wheelchair_capacity=cfg.wheelchair_capacity)  # Made with IBM Bob AI
                      for i, n in enumerate(start)]
        self.now = 0.0
        self.auto_demand = auto_demand
        self._next_id = 0
        self._last_reopt = 0.0
        self._last_retry = 0.0
        self.reassignments = 0
        self.events: list[str] = []
        # zones, time-of-day demand, watsonx forecasting, fleet rules
        self.zones = Zones(self.net, self.stations, cfg.n_zones, seed=cfg.seed)
        self.demand = DemandModel(self.zones, self.net, self.landmarks, cfg.demand_per_hour, cfg.seed)
        self.forecaster = Forecaster(self.demand.history(), cfg.forecast_source,
                                     cfg.forecast_horizon_slots, blocking=blocking_forecast)
        self.fleet = FleetManager(self)

    def available(self) -> list[Bus]:
        """Buses that may take new customers (not heading to / on a break)."""
        return [b for b in self.buses if b.state == "service"]

    # ---------------------------------------------------------------- demand
    def landmark_xy(self, name: str) -> tuple[float, float]:
        return self.net.xy(self.landmarks[name])

    def _generate_demand(self, dt: float) -> None:
        self.demand.per_hour = self.cfg.demand_per_hour
        for o, d, pax in self.demand.sample_trips(self.now, dt):
            self.submit(o, d, pax)

    def submit(self, origin, dest, passengers: int = 1, label: str = "",
               wheelchair: bool = False,          # Made with IBM Bob AI
               earliest: float | None = None,     # Made with IBM Bob AI
               latest: float | None = None) -> Request:  # Made with IBM Bob AI
        r = Request(self._next_id, tuple(origin), tuple(dest), self.now,
                    passengers, wheelchair=wheelchair, label=label,
                    earliest=earliest, latest=latest)
        self._next_id += 1
        self.requests[r.id] = r
        self.dispatcher.prepare(r)
        self.forecaster.record(self.zones.zone_of(r.origin))
        if self.dispatcher.try_assign(r, self.now, self.available()):
            self._log(f"#{r.id} -> bus {r.bus_id}")
        else:
            self._log(f"#{r.id} waiting for a free bus")
        self._update_etas()
        return r

    # ------------------------------------------------------------------ step
    def step(self, dt: float) -> None:
        t0 = self.now
        for bus in self.buses:
            self._advance_bus(bus, t0, dt)
            if bus.state != "break":
                bus.worked_s += dt
        self.now = t0 + dt
        if self.forecaster.tick(self.now):
            self._log(f"forecast updated ({self.forecaster.forecast_source}): "
                      f"{self.forecaster.next_window().sum():.0f} requests expected in next 30 min")

        if self.auto_demand:
            self._generate_demand(dt)

        if self.now - self._last_retry >= 30:
            self._last_retry = self.now
            for r in list(self.requests.values()):
                if r.status != "pending":
                    continue
                if self.now - r.t_request > self.cfg.pending_timeout_s:
                    r.status = "rejected"
                    self._log(f"#{r.id} rejected (no capacity)")
                elif self.dispatcher.try_assign(r, self.now, self.available()):
                    self._log(f"#{r.id} -> bus {r.bus_id} (retry)")

        if self.now - self._last_reopt >= self.cfg.reopt_interval_s:
            self._last_reopt = self.now
            moved = self.dispatcher.reoptimize(self.available(), self.now)
            if moved:
                self.reassignments += moved
                self._log(f"re-optimised: {moved} customer(s) re-routed")
            self.fleet.update()
        self._update_etas()

    def _log(self, msg: str) -> None:
        self.events.append(f"[{fmt_clock(self.now)}] {msg}")
        del self.events[:-50]

    # -------------------------------------------------------------- movement
    def _depart(self, bus: Bus, target: Node) -> None:
        nxt = self.net.next_hop(bus.node, target)
        bus.next_node = nxt
        bus.edge_total = bus.edge_remaining = self.net.edge_time(bus.node, nxt)
        bus.arrived = False

    def _advance_bus(self, bus: Bus, t: float, dt: float) -> None:
        t_end, dwell = t + dt, self.cfg.dwell_s
        for _ in range(10_000):
            if t >= t_end - 1e-9:
                return
            if bus.next_node is not None:                      # driving
                d = min(t_end - t, bus.edge_remaining)
                bus.edge_remaining -= d
                t += d
                bus.drive_s += d
                bus.pax_s += d * bus.load
                if bus.edge_remaining <= 1e-9:
                    bus.km += self.net.edge_length(bus.node, bus.next_node) / 1000
                    bus.node, bus.next_node = bus.next_node, None
                    if bus.schedule and bus.schedule[0].node == bus.node:
                        bus.arrived, bus.dwell_until = True, t + dwell
                continue
            if t < bus.dwell_until:                            # doors open
                t = min(t_end, bus.dwell_until)
                continue
            if bus.schedule:
                bus.reposition = None
                if bus.schedule[0].node != bus.node:
                    self._depart(bus, bus.schedule[0].node)
                    continue
                if not bus.arrived:
                    bus.arrived, bus.dwell_until = True, t + dwell
                    continue
                while bus.schedule and bus.schedule[0].node == bus.node:
                    st = bus.schedule[0]
                    if st.kind == "P" and st.ready > t + 1e-9:
                        break                                  # customer still walking
                    bus.schedule.pop(0)
                    self._serve(bus, st, t)
                if bus.schedule and bus.schedule[0].node == bus.node:
                    bus.dwell_until = bus.schedule[0].ready
                    continue
                bus.arrived = False
                continue
            if self.fleet.on_idle(bus, t):
                return                                         # on break
            if bus.state == "to_break":                        # empty: head to depot for break
                bus.reposition, bus.task = self.fleet.depot, "break"
            if bus.reposition is not None and bus.reposition != bus.node:
                self._depart(bus, bus.reposition)
                continue
            bus.reposition = None
            return                                             # idle

    def _serve(self, bus: Bus, st: Stop, t: float) -> None:
        r = self.requests[st.req_id]
        if st.kind == "P":
            r.status, r.t_pickup = "onboard", t
            bus.onboard[r.id] = r.passengers
        else:
            r.status, r.t_dropoff = "done", t
            bus.onboard.pop(r.id, None)
            if r.wheelchair:                                      # Made with IBM Bob AI
                bus.wheelchair_load = max(0, bus.wheelchair_load - 1)

    def _update_etas(self) -> None:
        for b in self.buses:
            res = self.dispatcher.plan(b, b.schedule, self.now)
            if res is None:
                continue
            for st, t in zip(b.schedule, res.times):
                r = self.requests[st.req_id]
                if st.kind == "P":
                    r.eta_pickup = t
                else:
                    r.eta_dropoff = t

    # ----------------------------------------------------------- reporting
    def run(self, seconds: float, dt: float = 5.0) -> None:
        end = self.now + seconds
        while self.now < end:
            self.step(dt)

    def drain(self, dt: float = 5.0, limit_s: float = 7200) -> None:
        """Stop new demand and finish all accepted trips."""
        self.auto_demand = False
        self.forecaster.frozen = True  # no new demand: stop forecasting / scoring
        end = self.now + limit_s
        while self.now < end and any(r.status in ("pending", "assigned", "onboard")
                                     for r in self.requests.values()):
            self.step(dt)

    def metrics(self) -> dict:
        cfg = self.cfg
        reqs = list(self.requests.values())
        done = [r for r in reqs if r.status == "done"]
        by_status = Counter(r.status for r in reqs)

        def avg(xs):
            xs = list(xs)
            return sum(xs) / len(xs) if xs else 0.0

        walk = lambda r: (r.walk_to_m + r.walk_from_m)
        trip = lambda r: r.t_dropoff + r.walk_from_m / cfg.walk_speed - r.t_request
        drive = sum(b.drive_s for b in self.buses)
        decided = by_status["done"] + by_status["rejected"] + by_status["onboard"] + by_status["assigned"]
        return {
            "clock": fmt_clock(self.now),
            "requests": len(reqs),
            "status": dict(by_status),
            "served_pct": 100 * (decided - by_status["rejected"]) / decided if decided else 0,
            "avg_walk_m": avg(walk(r) for r in done),
            "avg_wait_s": avg(r.t_pickup - (r.t_request + r.walk_to_m / cfg.walk_speed) for r in done),
            "avg_ride_s": avg(r.t_dropoff - r.t_pickup for r in done),
            "avg_detour": avg((r.t_dropoff - r.t_pickup) / r.direct_time for r in done),
            "avg_trip_s": avg(trip(r) for r in done),
            "buses": dict(Counter(b.state for b in self.buses)),
            "breaks_taken": sum(b.breaks_taken for b in self.buses),
            "coverage_pct": self.fleet.coverage_pct(),
            "forecast_source": self.forecaster.forecast_source,
            "forecast_mae": self.forecaster.mae(),
            "forecast_error": self.forecaster.last_error,
            "fleet_km": sum(b.km for b in self.buses),
            "avg_occupancy": sum(b.pax_s for b in self.buses) / drive if drive else 0,
            "reassignments": self.reassignments,
        }


def fmt_clock(t: float) -> str:
    t = int(t) + 7 * 3600  # simulation starts at 07:00
    return f"{t // 3600 % 24:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


class FixedLineBaseline:
    """A conventional bus line with the same fleet, for comparison.

    The line snakes through the city in both directions; riders walk to the
    nearest line stop, wait half a headway, ride, and walk to the destination.
    Capacity is ignored, which flatters the baseline.
    """

    def __init__(self, sim: Simulation):
        net, cfg = sim.net, sim.cfg
        self.net, self.cfg = net, cfg
        # Snake across the city on 3 horizontal bands, a stop every ~400 m.
        W, H = net.width_m, net.height_m
        self.stops: list[Node] = []
        for k, fy in enumerate((1 / 6, 1 / 2, 5 / 6)):
            xs = np.arange(0.05 * W, 0.95 * W + 1, 400.0)
            for x in (xs if k % 2 == 0 else xs[::-1]):
                s = sim.stations.nearest((float(x), fy * H))
                if not self.stops or self.stops[-1] != s:
                    self.stops.append(s)
        # cumulative time along the line (driving + dwell at each stop)
        self.cum = [0.0]
        for a, b in zip(self.stops, self.stops[1:]):
            self.cum.append(self.cum[-1] + net.travel_time(a, b) + cfg.dwell_s)
        one_way = self.cum[-1]
        buses_per_dir = max(cfg.n_buses // 2, 1)
        self.headway = 2 * one_way / buses_per_dir  # bus goes out and back

    def trip_time(self, origin, dest) -> tuple[float, float]:
        """Returns (door-to-door seconds, walking metres)."""
        ws = self.cfg.walk_speed
        walk = StationNetwork.walk_m
        xy = self.net.xy
        oa = [walk(origin, xy(s)) for s in self.stops]
        db = [walk(dest, xy(s)) for s in self.stops]
        best_walk = walk(origin, dest)  # just walk
        best = best_walk / ws
        for i, wo in enumerate(oa):
            for j, wd in enumerate(db):
                if i == j:
                    continue
                total = (wo + wd) / ws + self.headway / 2 + abs(self.cum[j] - self.cum[i])
                if total < best:
                    best, best_walk = total, wo + wd
        return best, best_walk
