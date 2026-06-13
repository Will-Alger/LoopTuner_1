"""Nightscout REST API client.

Fetches the historical data LoopTuner needs:

* ``entries``      -- CGM glucose readings (sgv, mg/dL)
* ``treatments``   -- boluses, carbs, temp basals
* ``profile``      -- current basal / ISF / carb-ratio schedules
* ``devicestatus`` -- (optional) the loop's own IOB/COB, for sanity checks

Reads are paginated by date so arbitrarily long histories can be pulled, and
every response is cached on disk so re-runs are instant and offline-friendly.
Authentication supports both the legacy ``api-secret`` (SHA-1 hashed) header
and token-based access.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from .config import NightscoutConfig


class NightscoutClient:
    def __init__(self, cfg: NightscoutConfig):
        self.cfg = cfg
        self.base = cfg.url.rstrip("/")
        self.session = requests.Session()
        if cfg.api_secret:
            digest = hashlib.sha1(cfg.api_secret.encode()).hexdigest()
            self.session.headers["api-secret"] = digest
        os.makedirs(cfg.cache_dir, exist_ok=True)

    # -- low level -----------------------------------------------------------
    def _params(self, extra: dict) -> dict:
        params = dict(extra)
        if self.cfg.token:
            params["token"] = self.cfg.token
        return params

    def _get(self, endpoint: str, params: dict, retries: int = 4) -> list:
        url = f"{self.base}/api/v1/{endpoint}.json"
        delay = 2.0
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(
                    url, params=self._params(params), timeout=60
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as err:
                last_err = err
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"Nightscout request failed: {url} ({last_err})")

    def _cache_path(self, name: str) -> str:
        tag = f"{name}_{self.cfg.days}d"
        return os.path.join(self.cfg.cache_dir, f"{tag}.json")

    def _cached(self, name: str, fetch, use_cache: bool) -> list:
        path = self._cache_path(name)
        if use_cache and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        data = fetch()
        with open(path, "w") as fh:
            json.dump(data, fh)
        return data

    # -- paginated date-range fetch -----------------------------------------
    def _fetch_range(self, endpoint: str, date_field: str, page: int = 20000) -> list:
        """Pull every record in the configured window, paging backwards."""
        since = datetime.now(timezone.utc) - timedelta(days=self.cfg.days)
        since_ms = int(since.timestamp() * 1000)
        since_iso = since.isoformat()

        out: list = []
        # Walk backwards in time using the high end as a moving cursor.
        cursor_iso = datetime.now(timezone.utc).isoformat()
        while True:
            params = {
                "count": page,
                f"find[{date_field}][$gte]": since_iso,
                f"find[{date_field}][$lt]": cursor_iso,
            }
            # entries use epoch-ms 'date'; treatments use ISO 'created_at'.
            if date_field == "date":
                params = {
                    "count": page,
                    "find[date][$gte]": since_ms,
                    "find[date][$lt]": _iso_to_ms(cursor_iso),
                }
            batch = self._get(endpoint, params)
            if not batch:
                break
            out.extend(batch)
            # Advance cursor to the oldest record we just saw.
            oldest = batch[-1].get(date_field)
            if oldest is None:
                break
            cursor_iso = _to_iso(oldest)
            if len(batch) < page:
                break
        return out

    # -- public --------------------------------------------------------------
    def entries(self, use_cache: bool = True) -> list:
        return self._cached(
            "entries", lambda: self._fetch_range("entries", "date"), use_cache
        )

    def treatments(self, use_cache: bool = True) -> list:
        return self._cached(
            "treatments",
            lambda: self._fetch_range("treatments", "created_at"),
            use_cache,
        )

    def devicestatus(self, use_cache: bool = True) -> list:
        return self._cached(
            "devicestatus",
            lambda: self._fetch_range("devicestatus", "created_at"),
            use_cache,
        )

    def profile(self, use_cache: bool = True) -> list:
        return self._cached(
            "profile", lambda: self._get("profile", {}), use_cache
        )


def _iso_to_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def _to_iso(value) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
    return str(value).replace("Z", "+00:00")
