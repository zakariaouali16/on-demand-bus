"""Dynamic dispatcher: decides which bus serves a customer, at which stations,
and in what order the bus visits its stops.

Algorithm
---------
1. Cheapest insertion: for each new request, try every bus, every candidate
   pickup station x dropoff station within walking range, and every position
   in the bus's stop list. Keep the option that adds the least cost
   (bus time + passenger time + walking), subject to capacity, max wait and
   max detour. Because the stations are chosen *together with* the bus route,
   the bus is pulled to the street that passes closest to everyone on board,
   instead of to each customer's door or a fixed line.
2. Continuous re-optimisation: every `reopt_interval_s`, all customers not yet
   picked up are removed and re-inserted (possibly on another bus). The new plan
   is kept only if it is cheaper; buses re-route on the fly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import INF, Bus, Config, Request, Stop
from .network import RoadNetwork
from .stations import StationNetwork

SOFT_PENALTY = 1000.0  # cost per second of violating an existing promise


@dataclass
class PlanResult:
    cost: float
    times: list[float]
    penalty: float


@dataclass
class Insertion:
    delta: float
    bus: Bus
    schedule: list[Stop]
    pickup: Stop
    dropoff: Stop
    walk_to_m: float
    walk_from_m: float


class Dispatcher:
    def __init__(self, net: RoadNetwork, stations: StationNetwork,
                 requests: dict[int, Request], cfg: Config):
        self.net, self.stations, self.requests, self.cfg = net, stations, requests, cfg

    # ------------------------------------------------------------------ plan
    def plan(self, bus: Bus, schedule: list[Stop], now: float,
             strict: int | None = None) -> PlanResult | None:
        """Simulate the bus driving `schedule`. Returns None if infeasible.

        Capacity is always hard. Time promises are hard for `strict` (the request
        being inserted) and soft (heavily penalised) for already-accepted ones,
        so small real-world delays never make a bus "unplannable".
        """
        reqs, to, cfg = self.requests, self.net.to, self.cfg
        node, t = bus.anchor(now)
        t0 = t
        load = bus.load
        wload = bus.wheelchair_load  # Made with IBM Bob AI
        pick = {rid: reqs[rid].t_pickup for rid in bus.onboard}
        dwelling_here = bus.arrived and bus.next_node is None
        pax_time = penalty = 0.0
        times: list[float] = []
        first = True
        for st in schedule:
            if st.node != node:
                t += to[st.node][node] + cfg.dwell_s
                node = st.node
            elif first and not dwelling_here:
                t += cfg.dwell_s
            first = False
            r = reqs[st.req_id]
            if st.kind == "P":
                if t < st.ready:
                    t = st.ready
                if t > st.deadline:
                    if st.req_id == strict:
                        return None
                    penalty += t - st.deadline
                load += r.passengers
                if load > bus.capacity:
                    return None
                if r.wheelchair:                     # Made with IBM Bob AI
                    wload += 1
                    if wload > bus.wheelchair_capacity:
                        return None
                pick[st.req_id] = t
            else:
                ride = t - pick[st.req_id]
                if ride > r.max_ride:
                    if st.req_id == strict:
                        return None
                    penalty += ride - r.max_ride
                load -= r.passengers
                if reqs[st.req_id].wheelchair:       # Made with IBM Bob AI
                    wload -= 1
                pax_time += t - r.t_request
            times.append(t)
        cost = (t - t0) + cfg.w_passenger * pax_time + SOFT_PENALTY * penalty
        return PlanResult(cost, times, penalty)

    # ------------------------------------------------------------ candidates
    def prepare(self, r: Request) -> None:
        cfg = self.cfg
        r.origin_cands = self.stations.candidates(r.origin, cfg.max_walk_m, k=3)
        r.dest_cands = self.stations.candidates(r.dest, cfg.max_walk_m, k=3)
        # direct car time: nearest street corner -> station nearest the destination
        a = self.net.nearest_node(r.origin)
        r.direct_time = max(self.net.travel_time(a, r.dest_cands[0][0]), 60.0)
        r.max_ride = max(r.direct_time * cfg.detour_factor,
                         r.direct_time + cfg.min_ride_slack_s)

    def pickup_stops(self, r: Request) -> list[tuple[Stop, float]]:
        cfg = self.cfg
        out = []
        for node, walk in r.origin_cands:
            walk_ready = r.t_request + walk / cfg.walk_speed
            # Time-window: passenger cannot board before `earliest` (if given).
            # Made with IBM Bob AI
            ready = max(walk_ready, r.earliest) if r.earliest is not None else walk_ready
            # Time-window: `latest` overrides the config max_wait deadline (if tighter).
            deadline = ready + cfg.max_wait_s
            if r.latest is not None:
                deadline = min(deadline, r.latest)
            out.append((Stop("P", r.id, node, ready, deadline), walk))
        return out

    def dropoff_stops(self, r: Request) -> list[tuple[Stop, float]]:
        return [(Stop("D", r.id, node), walk) for node, walk in r.dest_cands]

    # ------------------------------------------------------------- insertion
    def best_insertion(self, r: Request, now: float, buses: list[Bus],
                       p_cands: list[tuple[Stop, float]],
                       d_cands: list[tuple[Stop, float]]) -> Insertion | None:
        to, cfg = self.net.to, self.cfg
        best: Insertion | None = None
        for bus in buses:
            # Wheelchair requests need a bus with at least one free wheelchair space.
            # Made with IBM Bob AI
            if r.wheelchair and bus.wheelchair_load >= bus.wheelchair_capacity:
                continue
            anchor, t_anchor = bus.anchor(now)
            # Prune buses that can't reach any pickup station in time even directly.
            if all(t_anchor + to[p.node][anchor] > p.deadline for p, _ in p_cands):
                continue
            base = self.plan(bus, bus.schedule, now)
            if base is None:
                continue
            s, n = bus.schedule, len(bus.schedule)
            # State of the bus after serving s[:i]: node, time, load.
            nodes, times, loads = [anchor], [t_anchor], [bus.load]
            for st, t in zip(s, base.times):
                nodes.append(st.node)
                times.append(t)
                loads.append(loads[-1] + (self.requests[st.req_id].passengers
                                          * (1 if st.kind == "P" else -1)))
            pax = r.passengers
            for P, pwalk in p_cands:
                for D, dwalk in d_cands:
                    if D.node == P.node:
                        continue
                    walk_cost = cfg.w_walk * (pwalk + dwalk) / cfg.walk_speed
                    for i in range(n + 1):
                        # Times only grow along the route: once the pickup deadline
                        # is unreachable at position i, it is for every later i too.
                        if times[i] + to[P.node][nodes[i]] > P.deadline:
                            break
                        if loads[i] + pax > bus.capacity:
                            continue
                        head = s[:i] + [P]
                        for j in range(i, n + 1):
                            if j > i and loads[j] + pax > bus.capacity:
                                break  # bus would be over capacity while carrying r
                            cand = head + s[i:j] + [D] + s[j:]
                            res = self.plan(bus, cand, now, strict=r.id)
                            # Reject if it breaks a promise made to someone already booked.
                            if res is None or res.penalty > base.penalty + 1.0:
                                continue
                            delta = res.cost - base.cost + walk_cost
                            if best is None or delta < best.delta:
                                best = Insertion(delta, bus, cand, P, D, pwalk, dwalk)
        return best

    def assign(self, r: Request, ins: Insertion) -> None:
        ins.bus.schedule = ins.schedule
        if r.wheelchair:                 # Made with IBM Bob AI
            ins.bus.wheelchair_load += 1
        r.status = "assigned"
        r.bus_id = ins.bus.id
        r.pickup_node, r.dropoff_node = ins.pickup.node, ins.dropoff.node
        r.walk_to_m, r.walk_from_m = ins.walk_to_m, ins.walk_from_m

    def try_assign(self, r: Request, now: float, buses: list[Bus]) -> bool:
        ins = self.best_insertion(r, now, buses, self.pickup_stops(r), self.dropoff_stops(r))
        if ins is None:
            return False
        self.assign(r, ins)
        return True

    # -------------------------------------------------------- re-optimise
    def total_cost(self, buses: list[Bus], now: float) -> float:
        total = 0.0
        for b in buses:
            res = self.plan(b, b.schedule, now)
            total += INF if res is None else res.cost
        return total

    def reoptimize(self, buses: list[Bus], now: float) -> int:
        """Re-insert every not-yet-boarded customer. Returns #requests moved."""
        snapshot = {b.id: list(b.schedule) for b in buses}
        old_cost = self.total_cost(buses, now)

        movable: list[Stop] = []
        for b in buses:
            for k, st in enumerate(b.schedule):
                # Keep each bus's very next stop locked: it is already heading there.
                if st.kind == "P" and k > 0:
                    movable.append(st)
        if not movable:
            return 0
        ids = {st.req_id for st in movable}
        for b in buses:
            b.schedule = [s for s in b.schedule if s.req_id not in ids]

        movable.sort(key=lambda st: st.deadline)
        chosen: dict[int, Insertion] = {}
        for P in movable:
            r = self.requests[P.req_id]
            # The pickup station is locked (the customer is already walking there);
            # bus, visiting order and dropoff station may all change.
            ins = self.best_insertion(r, now, buses, [(P, r.walk_to_m)], self.dropoff_stops(r))
            if ins is None:
                break
            ins.bus.schedule = ins.schedule
            chosen[r.id] = ins
        else:
            new_cost = self.total_cost(buses, now)
            if new_cost < old_cost - 1.0:
                moved = 0
                for rid, ins in chosen.items():
                    r = self.requests[rid]
                    if r.bus_id != ins.bus.id or r.dropoff_node != ins.dropoff.node:
                        moved += 1
                    r.bus_id = ins.bus.id
                    r.dropoff_node, r.walk_from_m = ins.dropoff.node, ins.walk_from_m
                return moved

        for b in buses:
            b.schedule = snapshot[b.id]
        return 0
