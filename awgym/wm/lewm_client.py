"""LeWM HTTP client — the world model's full API surface.

Write paths (/observe /train /save) carry the X-WM-Token header; reads work
without it. TLS uses the shared internal CA when one is configured.

Reachability: the WM service publishes 127.0.0.1:8197 inside the distro, and
the in-container service name resolves to it on the shared network. Override
the base URL and the CA via ARC_GYM_LEWM_BASE and ARC_GYM_LEWM_CA.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from ..config import LEWM_CA_ENV as CA_ENV
from ..config import internal_ca, lewm_base, lewm_token


class LeWMClient:
    def __init__(self, base: str | None = None, token: str | None = None,
                 ca: str | None = None, timeout: float = 120.0):
        self.base = (base or lewm_base()).rstrip("/")
        self.token = token if token is not None else lewm_token()
        self._ca = ca or os.environ.get(CA_ENV) or internal_ca()
        self._timeout = timeout

    def _headers(self, write: bool = False) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if write and self.token:
            h["X-WM-Token"] = self.token
        return h

    def _post(self, path: str, payload: dict, write: bool = False) -> dict:
        with httpx.Client(verify=self._ca, timeout=self._timeout) as c:
            r = c.post(f"{self.base}{path}", json=payload,
                       headers=self._headers(write))
            r.raise_for_status()
            return r.json()

    def _get(self, path: str) -> dict:
        with httpx.Client(verify=self._ca, timeout=self._timeout) as c:
            r = c.get(f"{self.base}{path}")
            r.raise_for_status()
            return r.json()

    # -- reads -----------------------------------------------------------
    def health(self) -> dict:
        return self._get("/health")

    def dataset(self) -> dict:
        return self._get("/dataset")

    def encode(self, grid: list, cond: Any = None) -> list:
        return self._post("/encode", {"grid": grid, "cond": cond})["z"]

    def predict(self, z: list, action: int, ctx: Any = None) -> list:
        return self._post("/predict", {"z": z, "action": action, "ctx": ctx})["z_hat"]

    def surprise(self, grid: list, action: int, next_grid: list,
                 cond: Any = None, next_cond: Any = None, ctx: Any = None) -> Optional[float]:
        out = self._post("/surprise", {
            "grid": grid, "action": action, "next_grid": next_grid,
            "cond": cond, "next_cond": next_cond, "ctx": ctx})
        return out.get("surprise")

    def decode(self, z: list) -> list:
        return self._post("/decode", {"z": z})["grid"]

    def probe(self, grid: list, cond: Any = None) -> dict:
        return self._post("/probe", {"grid": grid, "cond": cond})

    def value(self, grid: list) -> Optional[float]:
        return self._post("/value", {"grid": grid}).get("value")

    # -- writes (X-WM-Token) --------------------------------------------
    def observe(self, grid: list, action: int, next_grid: list,
                game: str | None = None, cond: Any = None,
                next_cond: Any = None, source: str = "awgym") -> dict:
        return self._post("/observe", {
            "grid": grid, "action": action, "next_grid": next_grid,
            "game": game, "cond": cond, "next_cond": next_cond,
            "source": source}, write=True)

    def train(self, steps: int = 100) -> dict:
        return self._post("/train", {"steps": steps}, write=True)

    def save(self, path: str) -> bool:
        return bool(self._post("/save", {"path": path}, write=True).get("ok"))
