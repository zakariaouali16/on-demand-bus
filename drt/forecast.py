"""Demand forecasting per zone with IBM watsonx.ai (Granite TinyTimeMixer).

Every 15-minute slot, the forecaster sends the last 512 slots of request counts
per zone to watsonx `time_series/forecast` and gets back the expected requests
per zone for the coming slots. The fleet manager uses this to place idle buses
where customers are *about to* appear.

If watsonx is not configured or fails, a seasonal-naive forecast is used
instead (same slot on previous days, scaled by how busy today is so far).
Both are scored against what actually happened, so you can see how well each
predicts.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from .demand import SLOT_S, slot_time
from .watsonx import API_VERSION, WatsonxClient

TTM_MODEL = "ibm/granite-ttm-512-96-r2"
TTM_CONTEXT = 512


def seasonal_naive(history: np.ndarray, steps: int, days: int = 3) -> np.ndarray:
    """history (zones, T) -> forecast (zones, steps)."""
    zones, T = history.shape
    out = np.zeros((zones, steps))
    for s in range(steps):
        idx = [T + s - 96 * d for d in range(1, days + 1) if T + s - 96 * d >= 0]
        out[:, s] = history[:, idx].mean(1) if idx else history[:, -4:].mean(1)
    # scale by how today compares to previous days over the last hour
    recent = history[:, -4:].sum()
    past = np.mean([history[:, T - 96 * d - 4: T - 96 * d].sum()
                    for d in range(1, days + 1) if T - 96 * d - 4 >= 0] or [recent])
    if past > 0:
        out *= np.clip(recent / past, 0.5, 2.0)
    return out


class Forecaster:
    def __init__(self, history: np.ndarray, source: str = "auto", horizon: int = 2,
                 blocking: bool = True, client: WatsonxClient | None = None,
                 min_interval_s: float = 0.0):
        """source: 'auto' (watsonx if configured), 'watsonx', 'naive' or 'off'."""
        self.history = history.astype(float)          # completed slots, (zones, T)
        self.current = np.zeros(history.shape[0])     # counts in the running slot
        self.slot_index = 0                           # slots completed since sim start
        self.horizon = horizon
        self.blocking = blocking
        self.client = client or WatsonxClient()
        self.source = source
        if source == "auto":
            self.source = "watsonx" if self.client.enabled else "naive"
        self.forecast = seasonal_naive(self.history, horizon)  # (zones, horizon)
        self.forecast_source = "seasonal-naive"
        self.last_error = ""
        self.latency_s = 0.0
        # accuracy tracking: forecast made for the slot that is now running
        self._pending = {"watsonx": None, "naive": self.forecast[:, 0].copy()}
        self.abs_err = {"watsonx": [], "naive": []}
        self._lock = threading.Lock()
        self._busy = False
        self.frozen = False  # set when demand generation stops (end of a benchmark)
        self.min_interval_s = min_interval_s  # real seconds between watsonx calls
        self._last_call = 0.0
        self._paused_until = 0.0
        if self.source == "off":   # no forecast: fleet only keeps coverage
            self.forecast[:] = 0
            self.forecast_source = "off"
        else:
            self._refresh()

    # ---------------------------------------------------------------- inputs
    def record(self, zone: int) -> None:
        self.current[zone] += 1

    def tick(self, now: float) -> bool:
        """Call every step. Returns True when a new slot started (and forecast refreshed)."""
        if self.frozen:
            return False
        target = int(now // SLOT_S)
        changed = False
        while self.slot_index < target:
            actual = self.current
            for k, pred in self._pending.items():
                if pred is not None:
                    self.abs_err[k].append(float(np.abs(pred - actual).mean()))
            self.history = np.concatenate([self.history, actual[:, None]], axis=1)
            self.current = np.zeros_like(self.current)
            self.slot_index += 1
            changed = True
        if changed and self.source != "off":
            self._refresh()
        return changed

    # --------------------------------------------------------------- outputs
    def next_window(self, start: int = 0, end: int = 2) -> np.ndarray:
        """Expected requests per zone in slots [start, end) from now (default: next 30 min)."""
        with self._lock:
            return self.forecast[:, start:end].sum(1)

    def mae(self) -> dict[str, float | None]:
        return {k: (float(np.mean(v)) if v else None) for k, v in self.abs_err.items()}

    # -------------------------------------------------------------- refresh
    def _refresh(self) -> None:
        naive = seasonal_naive(self.history, self.horizon)
        with self._lock:
            self._pending["naive"] = naive[:, 0].copy()
            if self.source != "watsonx":
                self.forecast, self.forecast_source = naive, "seasonal-naive"
                return
            # Spare the watsonx quota: limit call rate, back off after quota errors.
            wall = time.time()
            if wall < self._paused_until or wall - self._last_call < self.min_interval_s:
                self._pending["watsonx"] = None
                if wall < self._paused_until:
                    self.forecast, self.forecast_source = naive, "seasonal-naive (watsonx paused)"
                elif self.forecast.shape[1] > 1:
                    # reuse the last watsonx forecast, shifted by one slot
                    self.forecast = np.concatenate([self.forecast[:, 1:], naive[:, -1:]], axis=1)
                return
            self._last_call = wall
        if self.blocking:
            self._call_watsonx(self.history.copy(), self.slot_index, naive)
        elif not self._busy:
            self._busy = True
            threading.Thread(target=self._call_watsonx,
                             args=(self.history.copy(), self.slot_index, naive), daemon=True).start()

    def _call_watsonx(self, history: np.ndarray, slot_index: int, naive: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            fc = self._watsonx_forecast(history, slot_index)
            with self._lock:
                self.forecast, self.forecast_source = fc, f"watsonx {TTM_MODEL}"
                if slot_index == self.slot_index:  # still the same slot -> score it
                    self._pending["watsonx"] = fc[:, 0].copy()
                self.last_error = ""
        except Exception as e:  # noqa: BLE001 - fall back, keep the sim running
            with self._lock:
                self.forecast, self.forecast_source = naive, "seasonal-naive (watsonx failed)"
                self._pending["watsonx"] = None
                self.last_error = _describe(e)
                if "quota" in self.last_error:
                    self._paused_until = time.time() + 3600
                    self.last_error = ("watsonx quota reached (Lite plan monthly limit) - "
                                       "using seasonal-naive for the next hour")
        finally:
            self.latency_s = time.perf_counter() - t0
            self._busy = False

    def _watsonx_forecast(self, history: np.ndarray, slot_index: int) -> np.ndarray:
        zones = history.shape[0]
        T = history.shape[1]
        ctx = history[:, -TTM_CONTEXT:]
        first = slot_index - ctx.shape[1]  # slot index (relative to sim start) of ctx[:, 0]
        stamps = [slot_time(first + i).strftime("%Y-%m-%dT%H:%M:%S") for i in range(ctx.shape[1])]
        data = {"date": [], "zone": [], "requests": []}
        for z in range(zones):
            data["date"] += stamps
            data["zone"] += [f"z{z}"] * len(stamps)
            data["requests"] += [float(v) for v in ctx[z]]
        body = {
            "model_id": TTM_MODEL,
            "project_id": self.client.project_id,
            "data": data,
            "schema": {"timestamp_column": "date", "id_columns": ["zone"],
                       "freq": f"{SLOT_S // 60}min", "target_columns": ["requests"]},
            "parameters": {"prediction_length": self.horizon},
        }
        req = urllib.request.Request(
            f"{self.client.url}/ml/v1/time_series/forecast?version={API_VERSION}",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.client._iam_token()}",
                     "Content-Type": "application/json", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            res = json.load(resp)["results"][0]
        out = np.zeros((zones, self.horizon))
        seen = [0] * zones
        for zid, v in zip(res["zone"], res["requests"]):
            z = int(zid[1:])
            if seen[z] < self.horizon:
                out[z, seen[z]] = max(float(v), 0.0)
                seen[z] += 1
        assert T >= TTM_CONTEXT, "not enough history for TTM"
        return out


def _describe(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            return f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception:  # noqa: BLE001
            return f"HTTP {e.code}"
    return f"{type(e).__name__}: {e}"
