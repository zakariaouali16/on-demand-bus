"""Live demo server: runs the simulation in real time and serves a map UI.

    python -m uvicorn drt.server:app --reload
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .models import Config
from .simulation import Simulation
from .watsonx import WatsonxClient, parse_booking

STATIC = Path(__file__).parent / "static"

sim = Simulation(Config(city=os.getenv("DRT_CITY") or None,
                         radius_m=float(os.getenv("DRT_RADIUS_M", 2000)),
                         demand_per_hour=float(os.getenv("DRT_DEMAND_PER_HOUR", 60))),
                 blocking_forecast=False)  # watsonx calls run in the background
# Spare the watsonx quota: at most one forecast call every 2 min of real time.
sim.forecaster.min_interval_s = float(os.getenv("WATSONX_FORECAST_MIN_INTERVAL_S", 120))
lock = threading.Lock()
control = {"speed": 10.0, "running": True}
wx = WatsonxClient()


async def _ticker():
    tick = 0.2
    while True:
        await asyncio.sleep(tick)
        if control["running"]:
            with lock:
                remaining = tick * control["speed"]
                while remaining > 0:  # keep sim steps small for accurate movement
                    dt = min(remaining, 5.0)
                    sim.step(dt)
                    remaining -= dt


@contextlib.asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(_ticker())
    yield
    task.cancel()


app = FastAPI(title="On-demand bus dispatcher", lifespan=lifespan)


def _xy(node):
    return list(sim.net.xy(node))


def _bus_xy(b):
    if b.next_node is None:
        return _xy(b.node)
    (x0, y0), (x1, y1) = sim.net.xy(b.node), sim.net.xy(b.next_node)
    f = 1 - b.edge_remaining / b.edge_total if b.edge_total else 1
    return [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]


def _bus_route(b):
    """Polyline of the bus's planned path through all its stops."""
    pts = [_bus_xy(b)]
    node, _ = b.anchor(sim.now)
    pts.append(_xy(node))
    targets = [s.node for s in b.schedule] or ([b.reposition] if b.reposition else [])
    for t in targets:
        for n in sim.net.path(node, t)[1:]:
            pts.append(_xy(n))
        node = t
    return pts


def _req(r):
    return {
        "id": r.id, "status": r.status, "bus": r.bus_id, "pax": r.passengers, "label": r.label,
        "origin": list(r.origin), "dest": list(r.dest),
        "pickup": _xy(r.pickup_node) if r.pickup_node else None,
        "dropoff": _xy(r.dropoff_node) if r.dropoff_node else None,
        "eta_pickup": r.eta_pickup - sim.now if r.eta_pickup and r.status == "assigned" else None,
        "eta_dropoff": r.eta_dropoff - sim.now if r.eta_dropoff and r.status in ("assigned", "onboard") else None,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/network")
def network():
    net = sim.net
    return {
        "width": net.width_m, "height": net.height_m,
        "name": net.name,
        "edges": [[_xy(a), _xy(b), d["major"]] for a, b, d in net.G.edges(data=True)
                  if a < b or not net.G.has_edge(b, a)],  # draw two-way streets once
        "stations": [_xy(n) for n in sim.stations.nodes],
        "landmarks": [{"name": k, "xy": _xy(v)} for k, v in sim.landmarks.items()],
        "watsonx": wx.enabled,
        "depot": _xy(sim.fleet.depot),
        "zones": [{"hub": _xy(h), "kind": sim.zones.kind[z]} for z, h in enumerate(sim.zones.hubs)],
    }


@app.get("/api/state")
def state():
    with lock:
        return {
            "now": sim.now,
            "speed": control["speed"], "running": control["running"],
            "auto_demand": sim.auto_demand, "rate": sim.cfg.demand_per_hour,
            "buses": [{
                "id": b.id, "xy": _bus_xy(b), "load": b.load, "capacity": b.capacity,
                "state": b.state, "task": "" if b.schedule else b.task,
                "worked_h": round(b.worked_s / 3600, 2),
                "break_left": max(b.break_until - sim.now, 0) if b.state == "break" else None,
                "route": _bus_route(b),
                "stops": [{"kind": s.kind, "req": s.req_id, "xy": _xy(s.node)} for s in b.schedule],
            } for b in sim.buses],
            "requests": [_req(r) for r in sim.requests.values()
                         if r.status in ("pending", "assigned", "onboard")
                         or (r.status == "done" and sim.now - (r.t_dropoff or 0) < 60)],
            "metrics": sim.metrics(),
            "zones": [{"forecast": round(float(f), 1), "covered": c}
                      for f, c in zip(sim.forecaster.next_window(), sim.fleet.covered)],
            "events": sim.events[-12:][::-1],
        }


class TripIn(BaseModel):
    origin: tuple[float, float]
    dest: tuple[float, float]
    passengers: int = Field(1, ge=1, le=10)
    label: str = ""


@app.post("/api/requests")
def create_request(trip: TripIn):
    with lock:
        r = sim.submit(trip.origin, trip.dest, trip.passengers, trip.label or "manual")
        return _req(r)


class BookingIn(BaseModel):
    text: str


@app.post("/api/book")
def book(b: BookingIn):
    try:
        parsed = parse_booking(b.text, list(sim.landmarks), wx)
    except Exception as e:  # noqa: BLE001 - surface parser/LLM errors to the user
        raise HTTPException(400, str(e))
    with lock:
        r = sim.submit(sim.landmark_xy(parsed["origin"]), sim.landmark_xy(parsed["destination"]),
                       parsed["passengers"], f'{parsed["origin"]} → {parsed["destination"]}')
        return {"parsed": parsed, "request": _req(r)}


class ControlIn(BaseModel):
    speed: float | None = Field(None, gt=0, le=60)
    running: bool | None = None
    auto_demand: bool | None = None
    rate: float | None = Field(None, ge=0, le=1000)


@app.post("/api/control")
def set_control(c: ControlIn):
    with lock:
        if c.speed is not None:
            control["speed"] = c.speed
        if c.running is not None:
            control["running"] = c.running
        if c.auto_demand is not None:
            sim.auto_demand = c.auto_demand
        if c.rate is not None:
            sim.cfg.demand_per_hour = c.rate
    return {"ok": True}
