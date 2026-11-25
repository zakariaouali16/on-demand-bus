from __future__ import annotations

from dataclasses import dataclass, field

from .network import Node

INF = float("inf")


@dataclass
class Config:
    # City: an OpenStreetMap place (e.g. "Union Square, San Francisco, USA"),
    # or None for a synthetic grid city (used by the tests).
    city: str | None = None
    radius_m: float = 2000.0          # area served around the city point
    grid_cols: int = 12
    grid_rows: int = 12
    block_m: float = 300.0
    station_spacing_m: float = 350.0  # distance between virtual stations
    # Fleet
    n_buses: int = 8
    capacity: int = 12
    dwell_s: float = 20.0             # time per stop (doors, boarding)
    # Service promises
    walk_speed: float = 1.3           # m/s
    max_walk_m: float = 600.0         # how far a customer may walk to a station
    max_wait_s: float = 600.0         # max wait at the station
    detour_factor: float = 1.8        # max ride time = direct time * factor ...
    min_ride_slack_s: float = 300.0   # ... or direct + slack, whichever is larger
    pending_timeout_s: float = 300.0  # unassigned request gives up after this
    # Optimisation
    reopt_interval_s: float = 60.0
    w_passenger: float = 1.0          # weight of passenger time vs. bus time
    w_walk: float = 1.5               # walking feels longer than riding
    # Demand (average over the day; peaks at 08:00 and 17:30 are ~2x this)
    demand_per_hour: float = 150.0
    seed: int = 1
    # Forecasting
    n_zones: int = 8
    forecast_source: str = "auto"     # auto | watsonx | naive | off
    forecast_horizon_slots: int = 4   # 4 x 15 min = next hour
    # Fleet management
    coverage_s: float = 300.0         # every zone must have a bus within 5 min
    min_zone_demand: float = 0.5      # forecast requests/30 min worth sending a bus for
    depot: str | None = None          # landmark name; None = station nearest the edge
    break_after_s: float = 3 * 3600   # driver works at most 3 h ...
    break_s: float = 30 * 60          # ... then takes a 30 min break at the depot
    break_grace_s: float = 30 * 60    # may postpone a break this long if many are off
    max_on_break: int = 2             # at most this many buses off at the same time
    # Accessibility — Made with IBM Bob AI
    wheelchair_capacity: int = 2      # wheelchair spaces per bus (≤ capacity)


@dataclass(frozen=True)
class Stop:
    kind: str            # "P" pickup or "D" dropoff
    req_id: int
    node: Node
    ready: float = 0.0   # customer can't board before this (walking time)
    deadline: float = INF


@dataclass
class Request:
    id: int
    origin: tuple[float, float]
    dest: tuple[float, float]
    t_request: float
    passengers: int = 1
    wheelchair: bool = False   # True → occupies one wheelchair space — Made with IBM Bob AI
    label: str = ""
    # Optional time-window for scheduled/pre-booked trips (seconds since sim start).
    # earliest: passenger cannot board before this time (overrides walking-time ready).
    # latest:   if the bus cannot pick up by this time, the request is rejected.
    # Made with IBM Bob AI
    earliest: float | None = None
    latest: float | None = None
    status: str = "pending"   # pending | assigned | onboard | done | rejected
    bus_id: int | None = None
    pickup_node: Node | None = None
    dropoff_node: Node | None = None
    walk_to_m: float = 0.0
    walk_from_m: float = 0.0
    direct_time: float = 0.0
    max_ride: float = INF
    t_pickup: float | None = None
    t_dropoff: float | None = None
    eta_pickup: float | None = None
    eta_dropoff: float | None = None
    origin_cands: list = field(default_factory=list, repr=False)
    dest_cands: list = field(default_factory=list, repr=False)


@dataclass
class Bus:
    id: int
    capacity: int
    node: Node                         # last intersection reached
    next_node: Node | None = None      # intersection currently driving to
    edge_remaining: float = 0.0        # seconds left on current street
    edge_total: float = 0.0
    schedule: list[Stop] = field(default_factory=list)
    onboard: dict[int, int] = field(default_factory=dict)  # req_id -> pax
    dwell_until: float = 0.0
    arrived: bool = False              # already dwelling at self.node
    reposition: Node | None = None     # idle bus heading somewhere useful
    wheelchair_capacity: int = 2       # wheelchair spaces available on this bus — Made with IBM Bob AI
    wheelchair_load: int = 0           # wheelchair spaces currently in use — Made with IBM Bob AI
    state: str = "service"             # service | to_break | break
    task: str = ""                     # why an idle bus is moving: coverage | demand | depot
    worked_s: float = 0.0              # time since last break
    break_until: float = 0.0
    breaks_taken: int = 0
    drive_s: float = 0.0
    km: float = 0.0
    pax_s: float = 0.0                 # passenger-seconds carried

    @property
    def load(self) -> int:
        return sum(self.onboard.values())

    def anchor(self, now: float) -> tuple[Node, float]:
        """Where/when the bus is free to follow a new plan (can't U-turn mid-street)."""
        if self.next_node is not None:
            return self.next_node, now + self.edge_remaining
        return self.node, max(now, self.dwell_until)
