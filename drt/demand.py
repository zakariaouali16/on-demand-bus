"""Zones and a time-of-day demand model.

The city is split into zones (k-means on the stations). Each zone has a type:
- work        (downtown core): people arrive in the morning, leave in the evening
- residential (outer zones):   people leave in the morning, come back in the evening
- mixed:                       steady demand with a lunchtime bump

The same model generates 6 days of history (for the forecaster to learn from)
and the live requests during the simulation, so forecasts can be checked.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np

from .network import Node, RoadNetwork
from .stations import StationNetwork

SLOT_S = 900                                   # 15-minute time slots
SIM_START = datetime(2026, 11, 23, 7, 0)       # simulation clock 0 = Monday 07:00


def _g(h: float, mu: float, sd: float) -> float:
    """Gaussian bump on a 24h clock."""
    d = min(abs(h - mu), 24 - abs(h - mu))
    return math.exp(-0.5 * (d / sd) ** 2)


def city_profile(h: float) -> float:
    """Relative city-wide demand by hour of day (mean over 24h = 1)."""
    return (0.12 + 1.3 * _g(h, 8.0, 1.1) + 1.4 * _g(h, 17.5, 1.3)
            + 0.6 * _g(h, 12.5, 1.4) + 0.7 * (1 if 7 <= h <= 21 else 0)) / _PROFILE_MEAN


_PROFILE_MEAN = 1.0
_PROFILE_MEAN = sum(city_profile(i / 4) for i in range(96)) / 96


def _origin_weight(kind: str, h: float) -> float:
    m, e, lunch = _g(h, 8.0, 1.8), _g(h, 17.5, 1.8), _g(h, 12.5, 1.5)
    return {"residential": 1 + 2.0 * m - 0.5 * e,
            "work": 0.6 - 0.3 * m + 2.2 * e,
            "mixed": 1 + 0.8 * lunch}[kind]


def _dest_weight(kind: str, h: float) -> float:
    m, e, lunch = _g(h, 8.0, 1.8), _g(h, 17.5, 1.8), _g(h, 12.5, 1.5)
    return {"residential": 0.8 - 0.4 * m + 2.0 * e,
            "work": 0.8 + 2.2 * m - 0.4 * e,
            "mixed": 1 + 0.8 * lunch}[kind]


class Zones:
    """k-means clustering of stations into service zones."""

    def __init__(self, net: RoadNetwork, stations: StationNetwork, k: int = 8, seed: int = 0):
        pts = stations.xy_arr
        k = min(k, len(pts))
        rng = np.random.default_rng(seed)
        cent = pts[rng.choice(len(pts), k, replace=False)]
        for _ in range(50):
            lab = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(1)
            new = np.array([pts[lab == z].mean(0) if (lab == z).any() else cent[z] for z in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        self.k = k
        self.centroids = cent
        self.station_zone: dict[Node, int] = {s: int(z) for s, z in zip(stations.nodes, lab)}
        # hub = station closest to the zone centre (where buses wait)
        self.hubs: list[Node] = []
        for z in range(k):
            members = [i for i in range(len(pts)) if lab[i] == z] or list(range(len(pts)))
            d = ((pts[members] - cent[z]) ** 2).sum(1)
            self.hubs.append(stations.nodes[members[int(d.argmin())]])
        # every street node belongs to its nearest zone centre (for sampling addresses)
        node_lab = ((net.xy_arr[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(1)
        self.nodes_in_zone: list[list[Node]] = [list(map(int, np.flatnonzero(node_lab == z)))
                                                for z in range(k)]
        # central zones are business districts, outer ones residential
        center = net.xy_arr.mean(0)
        dist = np.hypot(*(cent - center).T)
        order = np.argsort(dist)
        self.kind = ["mixed"] * k
        n_work = max(1, k // 4)
        for rank, z in enumerate(order):
            self.kind[z] = "work" if rank < n_work else ("mixed" if rank < k // 2 else "residential")
        self.size = np.array([max(len(n), 1) for n in self.nodes_in_zone], float)

    def zone_of(self, pt: tuple[float, float]) -> int:
        return int(((self.centroids - pt) ** 2).sum(1).argmin())


class DemandModel:
    def __init__(self, zones: Zones, net: RoadNetwork, landmarks: dict[str, Node],
                 per_hour: float, seed: int = 1):
        self.zones, self.net = zones, net
        self.per_hour = per_hour
        self.rng = random.Random(seed)
        self.landmarks_in_zone: list[list[Node]] = [[] for _ in range(zones.k)]
        for n in landmarks.values():
            self.landmarks_in_zone[zones.zone_of(net.xy(n))].append(n)

    @staticmethod
    def hour(t: float) -> float:
        """Hour of day for simulation time t (seconds since SIM_START)."""
        return (SIM_START.hour + SIM_START.minute / 60 + t / 3600) % 24

    def zone_rates(self, t: float) -> np.ndarray:
        """Expected requests per hour starting in each zone at time t."""
        h = self.hour(t)
        w = np.array([_origin_weight(self.zones.kind[z], h) for z in range(self.zones.k)])
        w = w * self.zones.size
        return self.per_hour * city_profile(h) * w / w.sum()

    def _point_in(self, z: int) -> tuple[float, float]:
        rng = self.rng
        lms = self.landmarks_in_zone[z]
        node = rng.choice(lms) if lms and rng.random() < 0.3 else rng.choice(self.zones.nodes_in_zone[z])
        x, y = self.net.xy(node)
        return (x + rng.gauss(0, 40), y + rng.gauss(0, 40))

    def sample_trips(self, t: float, dt: float) -> list[tuple[tuple, tuple, int]]:
        rates = self.zone_rates(t)
        h = self.hour(t)
        dest_w = [_dest_weight(self.zones.kind[z], h) * self.zones.size[z] for z in range(self.zones.k)]
        trips = []
        for z, lam in enumerate(rates * dt / 3600):
            for _ in range(_poisson(self.rng, lam)):
                o = self._point_in(z)
                for _ in range(20):
                    d = self._point_in(self.rng.choices(range(self.zones.k), dest_w)[0])
                    if math.hypot(o[0] - d[0], o[1] - d[1]) > 1000:
                        break
                pax = 1 if self.rng.random() < 0.8 else self.rng.randint(2, 3)
                trips.append((o, d, pax))
        return trips

    def history(self, days: int = 6, seed: int = 99) -> np.ndarray:
        """Synthetic past request counts per zone per 15-min slot, ending at SIM_START.

        Shape (zones, days*96). Includes day-to-day variation and noise.
        """
        rng = np.random.default_rng(seed)
        n = days * 96
        out = np.zeros((self.zones.k, n))
        for i in range(n):
            t = (i - n) * SLOT_S + SLOT_S / 2
            day_factor = 1 + 0.08 * math.sin(i / 96 * 2.1)  # some days busier
            lam = self.zone_rates(t) * SLOT_S / 3600 * day_factor
            out[:, i] = rng.poisson(lam)
        return out


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    k, p, L = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def slot_time(index_from_start: int) -> datetime:
    return SIM_START + timedelta(seconds=index_from_start * SLOT_S)
