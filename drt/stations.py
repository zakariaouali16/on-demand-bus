"""Virtual stations (meeting points) the bus can stop at.

The city is divided into a mesh of stations about `spacing_m` apart, placed on
real intersections (busier intersections preferred). A customer is never forced
to the single nearest one: we offer the dispatcher every station within walking
distance, so it can pick the one that lies closest to the bus's trajectory.
"""
from __future__ import annotations

import numpy as np

from .network import Node, RoadNetwork

WALK_DETOUR = 1.25  # walking distance along streets ~ 1.25 x straight line


class StationNetwork:
    def __init__(self, net: RoadNetwork, spacing_m: float = 350.0):
        self.net = net
        G = net.G
        # Prefer real intersections (many connecting streets) over mid-block nodes.
        order = sorted(G.nodes, key=lambda n: (-(G.in_degree(n) + G.out_degree(n)), n))
        chosen: list[Node] = []
        chosen_xy: list[tuple[float, float]] = []
        cell = spacing_m
        buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for n in order:
            x, y = net.xy(n)
            cx, cy = int(x // cell), int(y // cell)
            near = (p for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for p in buckets.get((cx + dx, cy + dy), ()))
            if any((x - px) ** 2 + (y - py) ** 2 < spacing_m ** 2 for px, py in near):
                continue
            chosen.append(n)
            chosen_xy.append((x, y))
            buckets.setdefault((cx, cy), []).append((x, y))
        # Landmarks are always stations, so named places are directly reachable.
        for n in net.landmarks.values():
            if n not in chosen:
                chosen.append(n)
                chosen_xy.append(net.xy(n))
        self.nodes: list[Node] = chosen
        self.xy_arr = np.array(chosen_xy)
        net.set_targets(self.nodes)

    @staticmethod
    def walk_m(pt: tuple[float, float], xy: tuple[float, float]) -> float:
        return WALK_DETOUR * float(np.hypot(pt[0] - xy[0], pt[1] - xy[1]))

    def nearest(self, pt: tuple[float, float]) -> Node:
        d = ((self.xy_arr - pt) ** 2).sum(axis=1)
        return self.nodes[int(d.argmin())]

    def candidates(self, pt: tuple[float, float], max_walk_m: float,
                   k: int = 3) -> list[tuple[Node, float]]:
        """Stations within walking range, closest first (at least one)."""
        walk = WALK_DETOUR * np.hypot(self.xy_arr[:, 0] - pt[0], self.xy_arr[:, 1] - pt[1])
        order = np.argsort(walk)[:k]
        within = [(self.nodes[i], float(walk[i])) for i in order if walk[i] <= max_walk_m]
        return within or [(self.nodes[int(order[0])], float(walk[order[0]]))]
