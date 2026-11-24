"""IBM watsonx.ai integration: turn a free-text booking into a structured trip.

  "Two of us need to get from the university to the airport"
      -> {"origin": "University", "destination": "Airport", "passengers": 2}

Uses the watsonx.ai REST chat API directly (no SDK needed). Configure with:
  WATSONX_APIKEY      IBM Cloud API key
  WATSONX_PROJECT_ID  watsonx.ai project id
  WATSONX_URL         region endpoint, default https://us-south.ml.cloud.ibm.com
  WATSONX_MODEL_ID    default ibm/granite-4-h-small
If not configured, a simple rule-based parser is used so the demo still works.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
API_VERSION = "2024-10-08"

WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_local_config() -> None:
    """Fill missing env vars from project-root files (never overrides real env vars):
    - .env         KEY=VALUE lines (WATSONX_PROJECT_ID, WATSONX_URL, ...)
    - apikey.json  the key file downloaded from IBM Cloud
    """
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key_file = PROJECT_ROOT / "apikey.json"
    if not os.getenv("WATSONX_APIKEY") and key_file.exists():
        try:
            os.environ["WATSONX_APIKEY"] = json.loads(key_file.read_text(encoding="utf-8"))["apikey"]
        except (ValueError, KeyError):
            pass


class WatsonxClient:
    def __init__(self):
        _load_local_config()
        self.apikey = os.getenv("WATSONX_APIKEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
        self.model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")
        self._token: str | None = None
        self._token_exp = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.apikey and self.project_id)

    def _iam_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        data = urllib.parse.urlencode({
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.apikey,
        }).encode()
        req = urllib.request.Request(IAM_URL, data=data, headers={
            "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.load(resp)
        self._token = body["access_token"]
        self._token_exp = time.time() + body.get("expires_in", 3600)
        return self._token

    def chat(self, messages: list[dict], max_tokens: int = 200) -> str:
        payload = {
            "model_id": self.model_id,
            "project_id": self.project_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.url}/ml/v1/text/chat?version={API_VERSION}",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self._iam_token()}",
                     "Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
        return body["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"model did not return JSON: {text!r}")
    return json.loads(m.group(0))


GENERIC_WORDS = {"city", "park", "town", "the"}


def _mentioned(place: str, text: str) -> bool:
    """True if the place's name, or a distinctive word of it, appears in text."""
    low = text.lower()
    if place.lower() in low:
        return True
    return any(len(w) > 3 and w not in GENERIC_WORDS and re.search(rf"\b{w}", low)
               for w in place.lower().split())


def parse_with_rules(text: str, places: list[str]) -> dict:
    low = text.lower()
    found = sorted(((low.find(p.lower()), p) for p in places if p.lower() in low))
    # also match a distinctive single word, e.g. "mall", "airport"
    if len(found) < 2:
        for p in places:
            for w in p.lower().split():
                if len(w) > 3 and w not in ("city", "park", "town") and re.search(rf"\b{w}\b", low):
                    pos = low.find(w)
                    if all(q != p for _, q in found):
                        found.append((pos, p))
        found.sort()
    if len(found) < 2:
        raise ValueError("Could not find both an origin and a destination. "
                         f"Known places: {', '.join(places)}")
    origin, dest = found[0][1], found[1][1]
    # "to X from Y" -> swap
    m_from = re.search(r"\bfrom\b", low)
    if m_from and abs(m_from.end() - found[1][0]) < abs(m_from.end() - found[0][0]):
        origin, dest = dest, origin
    pax = 1
    m = re.search(r"\b(\d+|one|two|three|four|five|six)\s+(?:of us|people|persons|passengers|adults|riders)", low)
    if m:
        g = m.group(1)
        pax = int(g) if g.isdigit() else WORD_NUMBERS[g]
    return {"origin": origin, "destination": dest, "passengers": pax}


def parse_booking(text: str, places: list[str], client: WatsonxClient | None = None) -> dict:
    """Returns {"origin", "destination", "passengers", "source"}."""
    client = client or WatsonxClient()
    if client.enabled:
        system = (
            "You are a booking assistant for an on-demand city bus. Extract the trip "
            "from the user's message. Origin and destination MUST be chosen from this "
            f"list exactly as written: {json.dumps(places)}. "
            'Reply with JSON only: {"origin": str, "destination": str, "passengers": int}. '
            "If the number of passengers is not mentioned, use 1. "
            "Never guess a place the user did not mention. "
            'If the origin or destination is not mentioned, reply {"error": "<short question to ask the user>"}.'
        )
        out = _extract_json(client.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]))
        if "error" in out:
            raise ValueError(out["error"])
        if out.get("origin") not in places or out.get("destination") not in places:
            raise ValueError(f"Unknown place in {out}")
        # LLMs sometimes invent a starting point: only accept places the user named.
        if not _mentioned(out["origin"], text):
            raise ValueError(f"Where are you starting from? Known places: {', '.join(places)}")
        if not _mentioned(out["destination"], text):
            raise ValueError(f"Where would you like to go? Known places: {', '.join(places)}")
        out["passengers"] = max(1, int(out.get("passengers", 1)))
        out["source"] = f"watsonx ({client.model_id})"
        return out
    out = parse_with_rules(text, places)
    out["source"] = "rule-based fallback (set WATSONX_APIKEY + WATSONX_PROJECT_ID to use watsonx)"
    return out
