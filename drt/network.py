"""Road network: real OpenStreetMap streets (one-ways, speed limits) or a
synthetic grid for tests.

Routing works like this: every trip ends at a *station*, so we precompute, for
every station, the travel time from every intersection to it, and the next
turn to take toward it (one reverse Dijkstra per station). The dispatcher can
then look up any travel time in O(1) while evaluating thousands of routes.
"""
from __future__ import annotations

import hashlib
import math
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np

Node = int
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# OSM road classes drawn thicker on the map
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary",
               "motorway_link", "trunk_link", "primary_link", "secondary_link"}

# OSM features used as named places for natural-language booking
LANDMARK_TAGS = {
    "amenity": ["hospital", "university", "townhall"],
    "railway": ["station"],
    "shop": ["mall", "department_store"],
    "leisure": ["stadium", "park"],
    "tourism": ["museum"],
}
# Order in which kinds take turns when picking landmarks
LANDMARK_KINDS = ["railway=station", "amenity=hospital", "amenity=university",
                  "leisure=park", "amenity=townhall", "shop=mall", "leisure=stadium",
                  "tourism=museum", "shop=department_store"]


class RoadNetwork:
    """Directed street graph. Nodes are ints 0..N-1 with x/y in metres."""

    def __init__(self, G: nx.DiGraph, name: str = "", landmarks: dict[str, Node] | None = None):
        self.G = G
        self.name = name
        self.landmarks: dict[str, Node] = landmarks or {}
        self.xy_arr = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in range(len(G))])
        self.width_m = float(self.xy_arr[:, 0].max())
        self.height_m = float(self.xy_arr[:, 1].max())
        self._targets: dict[Node, int] = {}
        self._T: list[list[float]] = []      # _T[target_idx][node] = seconds node -> target
        self._hop: list[list[int]] = []      # _hop[target_idx][node] = next node toward target

    # ------------------------------------------------------------ building
    @classmethod
    def grid(cls, cols: int = 12, rows: int = 12, block_m: float = 300.0,
             seed: int = 7, removed_frac: float = 0.06) -> "RoadNetwork":
        rng = random.Random(seed)
        idx = lambda x, y: x * rows + y
        U = nx.Graph()
        for x in range(cols):
            for y in range(rows):
                U.add_node(idx(x, y), x=x * block_m, y=y * block_m)
        for x in range(cols):
            for y in range(rows):
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx >= cols or y + dy >= rows:
                        continue
                    major = (dx == 1 and y % 4 == 0) or (dy == 1 and x % 4 == 0)
                    U.add_edge(idx(x, y), idx(x + dx, y + dy), length=block_m,
                               time=block_m / ((50 if major else 30) / 3.6), major=major)
        local = [e for e in U.edges if not U.edges[e]["major"]]
        rng.shuffle(local)
        for e in local[: int(len(local) * removed_frac)]:
            data = dict(U.edges[e])
            U.remove_edge(*e)
            if not nx.is_connected(U):
                U.add_edge(*e, **data)
        return cls(U.to_directed(), "Grid city")

    @classmethod
    def from_osm(cls, place: str, radius_m: float = 2000.0) -> "RoadNetwork":
        """Download (once, then cached in ./data) the drivable streets around `place`."""
        key = hashlib.sha1(f"{place}|{radius_m}|v2".encode()).hexdigest()[:10]
        cache = DATA_DIR / f"osm_{key}.pkl"
        if cache.exists():
            with cache.open("rb") as f:
                G, landmarks = pickle.load(f)
            return cls(G, place, landmarks)

        import osmnx as ox  # heavy import, only needed on first download

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(DATA_DIR / "osm_http_cache")
        center = ox.geocode(place)
        M = ox.graph_from_point(center, dist=radius_m, network_type="drive", simplify=True)
        M = ox.truncate.largest_component(M, strongly=True)
        M = ox.add_edge_speeds(M)
        M = ox.add_edge_travel_times(M)
        M = ox.project_graph(M)  # metres (UTM)

        nodes = list(M.nodes)
        minx = min(M.nodes[n]["x"] for n in nodes)
        miny = min(M.nodes[n]["y"] for n in nodes)
        ids = {n: i for i, n in enumerate(nodes)}
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(ids[n], x=M.nodes[n]["x"] - minx, y=M.nodes[n]["y"] - miny)
        for u, v, d in M.edges(data=True):
            if u == v:
                continue
            hw = d.get("highway")
            hw = hw[0] if isinstance(hw, list) else hw
            a, b = ids[u], ids[v]
            t = float(d["travel_time"])
            if G.has_edge(a, b) and G.edges[a, b]["time"] <= t:
                continue  # keep the fastest of parallel edges
            G.add_edge(a, b, time=t, length=float(d["length"]), major=hw in MAJOR_ROADS)

        landmarks = cls._fetch_landmarks(ox, center, radius_m, M, ids)
        DATA_DIR.mkdir(exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump((G, landmarks), f)
        return cls(G, place, landmarks)

    @staticmethod
    def _fetch_landmarks(ox, center, radius_m, M, ids, max_n: int = 14,
                         min_gap_m: float = 450.0) -> dict[str, Node]:
        try:
            feats = ox.features_from_point(center, tags=LANDMARK_TAGS, dist=radius_m)
        except Exception:  # noqa: BLE001 - landmarks are optional
            return {}
        feats = feats[feats["name"].notna()].to_crs(M.graph["crs"])
        # Group by kind, biggest first (big = well known), then take turns
        # between kinds so we get a mix: a station, a hospital, a park, ...
        groups: dict[str, list] = {k: [] for k in LANDMARK_KINDS}
        for _, row in feats.iterrows():
            for kind in LANDMARK_KINDS:
                col, val = kind.split("=")
                if col in row and row[col] == val:
                    groups[kind].append(row)
                    break
        for rows in groups.values():
            rows.sort(key=lambda r: -r.geometry.area)
        picked: dict[str, Node] = {}
        pts: list[tuple[float, float]] = []
        while len(picked) < max_n and any(groups.values()):
            for kind in LANDMARK_KINDS:
                rows = groups[kind]
                while rows:
                    row = rows.pop(0)
                    name = str(row["name"]).strip()
                    c = row.geometry.centroid
                    if (not name or name in picked or len(name) > 40
                            or any(math.hypot(c.x - px, c.y - py) < min_gap_m for px, py in pts)):
                        continue
                    picked[name] = ids[ox.distance.nearest_nodes(M, c.x, c.y)]
                    pts.append((c.x, c.y))
                    break
                if len(picked) >= max_n:
                    break
        return picked

    # ------------------------------------------------------------- routing
    def set_targets(self, targets: list[Node]) -> None:
        """Precompute travel time + next hop from every node to each target."""
        R = self.G.reverse(copy=False)
        n = len(self.G)
        self._targets = {t: i for i, t in enumerate(targets)}
        self._T, self._hop = [], []
        for t in targets:
            pred, dist = nx.dijkstra_predecessor_and_distance(R, t, weight="time")
            row = [math.inf] * n
            hop = [-1] * n
            for node, d in dist.items():
                row[node] = d
            for node, p in pred.items():
                if p:
                    hop[node] = p[0]  # in the reversed graph, predecessor = next hop forward
            self._T.append(row)
            self._hop.append(hop)
        # to[target][node] = seconds node -> target (fast lookup for the dispatcher)
        self.to: dict[Node, list[float]] = {t: self._T[i] for t, i in self._targets.items()}

    def travel_time(self, a: Node, b: Node) -> float:
        """Seconds to drive from any node `a` to target node `b` (a station)."""
        return self._T[self._targets[b]][a]

    def time_table(self, b: Node) -> list[float]:
        return self._T[self._targets[b]]

    def next_hop(self, a: Node, b: Node) -> Node:
        return self._hop[self._targets[b]][a]

    def path(self, a: Node, b: Node) -> list[Node]:
        hop = self._hop[self._targets[b]]
        out = [a]
        while out[-1] != b:
            out.append(hop[out[-1]])
        return out

    # ------------------------------------------------------------ geometry
    def xy(self, node: Node) -> tuple[float, float]:
        x, y = self.xy_arr[node]
        return (float(x), float(y))

    def nearest_node(self, pt: tuple[float, float]) -> Node:
        d = ((self.xy_arr - pt) ** 2).sum(axis=1)
        return int(d.argmin())

    def edge_time(self, a: Node, b: Node) -> float:
        return self.G.edges[a, b]["time"]

    def edge_length(self, a: Node, b: Node) -> float:
        return self.G.edges[a, b]["length"]
