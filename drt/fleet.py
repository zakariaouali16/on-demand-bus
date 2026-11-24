"""Fleet manager: what buses do when they are not carrying anyone.

Runs every minute and decides, in priority order:
1. Breaks    - a driver who has worked `break_after_s` stops taking new
               customers, finishes the ones already booked, drives to the depot
               and rests `break_s`. At most `max_on_break` buses are off at once
               (a break can be postponed up to `break_grace_s` to respect that).
               A driver close to their limit takes the break early when the
               forecast says it is quiet.
2. Coverage  - every zone must have a bus within `coverage_s` of its hub. Idle
               buses are sent to uncovered zones first (busiest forecast first).
3. Forecast  - remaining idle buses go to zones where watsonx predicts the most
               requests in the next 30 minutes, relative to buses already there.
4. Depot     - idle buses that are not needed (quiet forecast everywhere) wait
               at the depot, still available for bookings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .models import Bus

if TYPE_CHECKING:
    from .simulation import Simulation


class FleetManager:
    def __init__(self, sim: "Simulation"):
        self.sim = sim
        cfg, net, st = sim.cfg, sim.net, sim.stations
        if cfg.depot and cfg.depot in sim.landmarks:
            self.depot = st.nearest(net.xy(sim.landmarks[cfg.depot]))
        else:
            # A depot sits at the edge of the service area, like real bus garages.
            c = st.xy_arr.mean(0)
            self.depot = st.nodes[int(np.hypot(*(st.xy_arr - c).T).argmax())]
        self.zones = sim.zones
        self.covered = [True] * self.zones.k
        self.coverage_checks = 0
        self.coverage_ok = 0
        # Stagger drivers' shifts so breaks don't all fall at the same time.
        n = len(sim.buses)
        for i, b in enumerate(sim.buses):
            b.worked_s = cfg.break_after_s * i / n

    # ------------------------------------------------------------- helpers
    def _position(self, b: Bus) -> int:
        """Where the bus will be soon: end of its route, its target, or here."""
        if b.schedule:
            return b.schedule[-1].node
        if b.reposition is not None:
            return b.reposition
        return b.anchor(self.sim.now)[0]

    def _tt(self, a: int, hub: int) -> float:
        return self.sim.net.travel_time(a, hub)

    # ----------------------------------------------------------------- main
    def update(self) -> None:
        self._manage_breaks()
        self._position_idle()

    def _manage_breaks(self) -> None:
        """Forecast-aware breaks: rest before the rush, not in it."""
        sim, cfg = self.sim, self.sim.cfg
        off = sum(1 for b in sim.buses if b.state != "service")
        fc = sim.forecaster
        now30 = float(fc.next_window(0, 2).sum())      # next 30 min
        later30 = float(fc.next_window(2, 4).sum())    # 30-60 min from now
        avg30 = cfg.demand_per_hour / 2                # a normal half hour
        level = now30 / avg30 if avg30 else 0.0
        rising = later30 > 1.15 * now30
        quiet = level < 0.8
        peak_now = level > 1.4 and not rising
        # most overworked first
        for b in sorted(sim.buses, key=lambda b: -b.worked_s):
            if b.state != "service":
                continue
            due = b.worked_s >= cfg.break_after_s
            forced = b.worked_s >= cfg.break_after_s + cfg.break_grace_s
            early = (not due and not b.schedule and b.worked_s >= 0.5 * cfg.break_after_s
                     and (quiet or rising))
            if forced or (off < cfg.max_on_break and ((due and not peak_now) or early)):
                b.state, b.task = "to_break", "break"
                off += 1
                why = ("taken early: quiet now" if quiet else "taken early: rush hour coming") \
                    if early else ("postponed limit reached" if forced and peak_now else "due")
                sim._log(f"bus {b.id}: driver break ({b.worked_s / 3600:.1f} h worked, {why})")

    def _position_idle(self) -> None:
        sim, cfg, Z = self.sim, self.sim.cfg, self.zones
        forecast = sim.forecaster.next_window()            # requests per zone, next 30 min
        in_service = [b for b in sim.buses if b.state == "service"]
        idle = [b for b in in_service if not b.schedule]
        busy = [b for b in in_service if b.schedule]

        # Buses on their way to a break head to the depot once empty.
        for b in sim.buses:
            if b.state == "to_break" and not b.schedule:
                b.reposition, b.task = self.depot, "break"

        def zone(p: int) -> int:
            return Z.station_zone.get(p, Z.zone_of(sim.net.xy(p)))

        def coverers(z: int, positions: dict) -> list:
            return [k for k, p in positions.items() if self._tt(p, Z.hubs[z]) <= cfg.coverage_s]

        # Where every in-service bus is / will be free.
        pos = {b.id: self._position(b) for b in in_service}
        idle_ids = {b.id for b in idle}
        by_id = {b.id: b for b in idle}

        # --- 1. coverage -------------------------------------------------
        self.covered = [bool(coverers(z, pos)) for z in range(Z.k)]
        self.coverage_checks += Z.k
        self.coverage_ok += sum(self.covered)
        moved: set[int] = set()

        def sole_coverer(bid: int) -> bool:
            return any(coverers(z, pos) == [bid] for z in range(Z.k))

        for z in sorted(range(Z.k), key=lambda z: -forecast[z]):
            if coverers(z, pos):
                continue
            options = [i for i in idle_ids - moved if not sole_coverer(i)] or list(idle_ids - moved)
            if not options:
                break
            bid = min(options, key=lambda i: self._tt(pos[i], Z.hubs[z]))
            self._send(by_id[bid], Z.hubs[z], "coverage")
            pos[bid] = Z.hubs[z]
            moved.add(bid)

        # --- 2. forecast: each zone gets buses in proportion to demand ---
        total = float(forecast.sum())
        n = len(in_service)
        want = forecast / total * n if total > 0 else np.zeros(Z.k)
        have = np.zeros(Z.k)
        for p in pos.values():
            have[zone(p)] += 1
        quiet = total < cfg.min_zone_demand * Z.k
        # buses in the most over-supplied zones move first
        for bid in sorted(idle_ids - moved, key=lambda i: -(have[zone(pos[i])] - want[zone(pos[i])])):
            b, here = by_id[bid], pos[bid]
            cur = zone(here)
            surplus = have[cur] - want[cur]
            best, gain = None, 0.0
            if surplus >= 0.5:
                for z in range(Z.k):
                    deficit = want[z] - have[z]
                    if (z != cur and deficit >= 0.7 and forecast[z] >= cfg.min_zone_demand
                            and self._tt(here, Z.hubs[z]) <= 600 and deficit > gain):
                        best, gain = z, deficit
            if best is not None:
                have[cur] -= 1
                have[best] += 1
                pos[bid] = Z.hubs[best]
                self._send(b, Z.hubs[best], "demand")
            elif quiet and surplus >= 0.5 and not sole_coverer(bid):
                # --- 3. quiet everywhere and not needed here: wait at depot
                have[cur] -= 1
                pos[bid] = self.depot
                self._send(b, self.depot, "depot")
            elif b.reposition is None:
                b.task = "standby"  # stay where it is; it is useful here

    def _send(self, b: Bus, node: int, task: str) -> None:
        if b.reposition != node and b.node != node:
            b.reposition = node
        b.task = task

    # ----------------------------------------------------------- movement
    def on_idle(self, b: Bus, t: float) -> bool:
        """Called by the simulation when a bus has nothing to do at time t.
        Returns True if the bus should stay put until the end of the step."""
        cfg = self.sim.cfg
        if b.state == "break":
            if t >= b.break_until:
                b.state, b.task, b.worked_s = "service", "", 0.0
                self.sim._log(f"bus {b.id}: back in service")
                return False
            return True
        if b.state == "to_break" and b.node == self.depot:
            b.state, b.break_until = "break", t + cfg.break_s
            b.breaks_taken += 1
            b.reposition = None
            self.sim._log(f"bus {b.id}: driver on break until {_clock(b.break_until)}")
            return True
        return False

    def coverage_pct(self) -> float:
        return 100 * self.coverage_ok / self.coverage_checks if self.coverage_checks else 100.0


def _clock(t: float) -> str:
    from .simulation import fmt_clock
    return fmt_clock(t)[:5]
