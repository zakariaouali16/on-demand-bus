import unittest

from drt.models import Config
from drt.simulation import GRID_LANDMARKS, Simulation
from drt.watsonx import parse_with_rules


def quiet_sim(**kw) -> Simulation:
    return Simulation(Config(forecast_source="naive", **kw), auto_demand=False)


class DispatcherTests(unittest.TestCase):
    def test_pickup_before_dropoff_and_served(self):
        sim = quiet_sim(n_buses=2)
        r = sim.submit((300, 300), (2700, 2400))
        self.assertEqual(r.status, "assigned")
        bus = sim.buses[r.bus_id]
        kinds = [s.kind for s in bus.schedule if s.req_id == r.id]
        self.assertEqual(kinds, ["P", "D"])
        sim.drain()
        self.assertEqual(r.status, "done")
        self.assertLessEqual(r.t_dropoff - r.t_pickup, r.max_ride + 1)

    def test_capacity_respected(self):
        sim = quiet_sim(n_buses=1, capacity=4)
        a = sim.submit((300, 300), (3000, 3000), passengers=3)
        b = sim.submit((300, 600), (3000, 2700), passengers=3)
        self.assertEqual(a.status, "assigned")
        # b can only ride after a has left the bus -> never more than 4 on board
        max_load = 0
        for _ in range(2000):
            sim.step(5)
            max_load = max(max_load, sim.buses[0].load)
        self.assertLessEqual(max_load, 4)
        self.assertIn(b.status, ("done", "rejected", "pending", "assigned", "onboard"))

    def test_shared_ride_uses_nearby_stations(self):
        sim = quiet_sim(n_buses=1)
        a = sim.submit((600, 600), (3000, 3000))
        b = sim.submit((700, 650), (2950, 3050))  # almost the same trip
        self.assertEqual(a.bus_id, b.bus_id)
        sim.drain()
        self.assertEqual((a.status, b.status), ("done", "done"))
        # both on board simultaneously at some point -> pooled
        self.assertLess(b.t_pickup, a.t_dropoff)

    def test_walk_within_limit(self):
        sim = quiet_sim()
        r = sim.submit((450, 450), (2850, 1950))
        self.assertLessEqual(r.walk_to_m, sim.cfg.max_walk_m)
        self.assertLessEqual(r.walk_from_m, sim.cfg.max_walk_m)

    def test_reoptimize_never_increases_cost(self):
        sim = Simulation(Config(demand_per_hour=300, seed=3, forecast_source="naive"))
        for _ in range(60):
            sim.step(5)
            before = sim.dispatcher.total_cost(sim.buses, sim.now)
            sim.dispatcher.reoptimize(sim.buses, sim.now)
            after = sim.dispatcher.total_cost(sim.buses, sim.now)
            self.assertLessEqual(after, before + 1e-6)

    # Made with IBM Bob AI
    def test_time_window_honored(self):
        """A pre-booked trip with earliest/latest constraints is served within the window."""
        sim = quiet_sim(n_buses=2)
        # Schedule a trip that cannot start for 300 s (earliest) and must start by 900 s.
        r = sim.submit((300, 300), (2700, 2400), earliest=300.0, latest=900.0)
        self.assertEqual(r.status, "assigned")
        sim.drain()
        self.assertEqual(r.status, "done")
        # Actual pickup must respect the earliest constraint.
        self.assertGreaterEqual(r.t_pickup, 300.0 - 1e-6)

    # Made with IBM Bob AI
    def test_time_window_too_tight_rejects(self):
        """A latest time in the past means no bus can honour it → rejected."""
        sim = quiet_sim(n_buses=2)
        # latest is in the past relative to t_request=0 → every deadline is 0.
        r = sim.submit((300, 300), (2700, 2400), earliest=0.0, latest=0.0)
        # Should either be rejected immediately or stay unassigned and time out.
        sim.run(sim.cfg.pending_timeout_s + 10)
        self.assertIn(r.status, ("rejected", "pending"))

    # Made with IBM Bob AI
    def test_wheelchair_capacity_respected(self):
        """Wheelchair spaces are a sub-capacity: more wheelchairs than slots → second bus."""
        # 1 bus, 1 wheelchair space → two simultaneous wheelchair riders should split buses.
        sim = quiet_sim(n_buses=2, wheelchair_capacity=1)
        a = sim.submit((300, 300), (2700, 2400), wheelchair=True)
        b = sim.submit((300, 600), (2700, 2100), wheelchair=True)
        self.assertEqual(a.status, "assigned")
        self.assertEqual(b.status, "assigned")
        # With only 1 space each, both wheelchair riders must be on different buses.
        self.assertNotEqual(a.bus_id, b.bus_id)

    # Made with IBM Bob AI
    def test_wheelchair_zero_capacity_cannot_serve(self):
        """A fleet with 0 wheelchair spaces rejects wheelchair requests."""
        sim = quiet_sim(n_buses=2, wheelchair_capacity=0)
        r = sim.submit((300, 300), (2700, 2400), wheelchair=True)
        sim.run(sim.cfg.pending_timeout_s + 10)
        self.assertIn(r.status, ("rejected", "pending"))

    def test_full_simulation_serves_everyone_at_moderate_demand(self):
        sim = Simulation(Config(demand_per_hour=120, seed=5, forecast_source="naive"))
        sim.run(1800)
        sim.drain()
        m = sim.metrics()
        self.assertGreater(m["served_pct"], 95)
        self.assertEqual(m["status"].get("onboard", 0), 0)


class FleetTests(unittest.TestCase):
    def test_breaks_at_depot_and_limited(self):
        sim = Simulation(Config(forecast_source="naive", n_buses=4, demand_per_hour=60,
                                break_after_s=1800, break_s=600, max_on_break=1,
                                break_grace_s=1e9, seed=2))
        max_off = 0
        on_break_at = set()
        for _ in range(int(3 * 3600 / 5)):
            sim.step(5)
            off = [b for b in sim.buses if b.state == "break"]
            max_off = max(max_off, len(off))
            for b in off:
                on_break_at.add(b.node)
                self.assertFalse(b.onboard, "nobody on board during a break")
        self.assertGreater(sum(b.breaks_taken for b in sim.buses), 0)
        self.assertLessEqual(max_off, 1)
        self.assertEqual(on_break_at, {sim.fleet.depot})

    def test_overdue_driver_is_forced_to_rest(self):
        sim = quiet_sim(n_buses=3, max_on_break=0, break_after_s=600, break_grace_s=600)
        sim.run(3600)
        self.assertGreater(sum(b.breaks_taken for b in sim.buses), 0)
        self.assertTrue(all(b.worked_s <= 1200 + 300 for b in sim.buses))

    def test_no_new_customers_for_bus_due_a_break(self):
        sim = quiet_sim(n_buses=2)
        sim.buses[0].state = "to_break"
        r = sim.submit((300, 300), (2700, 2400))
        self.assertEqual(r.bus_id, 1)

    def test_coverage_spreads_idle_buses(self):
        sim = quiet_sim(n_buses=8)
        sim.fleet.update()
        sim.run(900)
        sim.fleet.update()
        self.assertGreaterEqual(sum(sim.fleet.covered), sim.zones.k - 1)

    def test_quiet_forecast_sends_buses_to_depot(self):
        sim = quiet_sim(n_buses=4)
        sim.forecaster.forecast[:] = 0  # nothing expected anywhere
        sim.fleet._position_idle()
        tasks = {b.task for b in sim.buses}
        self.assertIn("depot", tasks)


class ForecastTests(unittest.TestCase):
    def test_seasonal_naive_repeats_daily_pattern(self):
        import numpy as np
        from drt.forecast import seasonal_naive
        day = np.arange(96, dtype=float)
        hist = np.tile(day, (2, 4))           # 2 zones, 4 identical days
        fc = seasonal_naive(hist, 2)
        np.testing.assert_allclose(fc[0], [0, 1])

    def test_watsonx_failure_falls_back(self):
        from drt.forecast import Forecaster

        class Broken:
            enabled = True
            project_id = "x"
            url = "http://127.0.0.1:9"  # nothing listens here

            def _iam_token(self):
                return "token"

        sim = quiet_sim()
        f = Forecaster(sim.demand.history(), "watsonx", client=Broken())
        self.assertIn("seasonal-naive", f.forecast_source)
        self.assertTrue(f.last_error)
        self.assertEqual(f.next_window().shape, (sim.zones.k,))


class BookingParserTests(unittest.TestCase):
    places = list(GRID_LANDMARKS)

    def test_from_to(self):
        out = parse_with_rules("Two of us need to go from the University to the Airport", self.places)
        self.assertEqual(out, {"origin": "University", "destination": "Airport", "passengers": 2})

    def test_to_from_order(self):
        out = parse_with_rules("I need a ride to the hospital from old town", self.places)
        self.assertEqual((out["origin"], out["destination"]), ("Old Town", "Hospital"))

    def test_missing_place(self):
        with self.assertRaises(ValueError):
            parse_with_rules("take me to the airport", self.places)


if __name__ == "__main__":
    unittest.main()
