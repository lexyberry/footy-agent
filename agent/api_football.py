"""Thin API-Football v3 client with on-disk caching and rate-limit handling."""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

BASE = "https://v3.football.api-sports.io"


class ApiFootball:
    def __init__(self, key: str | None = None, cache_dir: str = "data/cache"):
        self.key = key or os.environ.get("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY is not set")
        self.s = requests.Session()
        self.s.headers["x-apisports-key"] = self.key
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self._last = 0.0
        self.min_interval = 0.3   # ≤200 calls/min, under the per-minute plan limit

    def get(self, path: str, ttl: int = 3600, **params) -> list:
        """GET with cache. ttl=0 disables cache; ttl<0 caches forever (finished data)."""
        key = hashlib.md5(f"{path}{sorted(params.items())}".encode()).hexdigest()
        cf = self.cache / f"{key}.json"
        if ttl != 0 and cf.exists() and (ttl < 0 or time.time() - cf.stat().st_mtime < ttl):
            return json.loads(cf.read_text())

        out, page = [], 1
        while True:
            q = dict(params)
            if page > 1:
                q["page"] = page
            for attempt in range(5):
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                r = self.s.get(f"{BASE}/{path}", params=q, timeout=30)
                self._last = time.time()
                self.calls += 1
                if r.status_code == 429:
                    time.sleep(2 ** attempt * 5)
                    continue
                r.raise_for_status()
                body = r.json()
                errs = body.get("errors")
                if isinstance(errs, dict) and "rateLimit" in errs and attempt < 4:
                    time.sleep(61)  # per-minute window; wait it out and retry
                    continue
                break
            if errs and errs != []:
                raise RuntimeError(f"API-Football error on {path}: {errs}")
            out.extend(body.get("response", []))
            paging = body.get("paging") or {}
            if paging.get("current", 1) >= paging.get("total", 1):
                break
            page += 1
            time.sleep(0.25)

        if ttl != 0:
            cf.write_text(json.dumps(out))
        return out

    # convenience wrappers -------------------------------------------------
    def fixtures(self, league: int, season: int, ttl=3600):
        return self.get("fixtures", ttl=ttl, league=league, season=season)

    def fixture(self, fixture_id: int):
        r = self.get("fixtures", ttl=300, id=fixture_id)
        return r[0] if r else None

    def odds(self, fixture_id: int, bookmaker: int | None = None):
        p = {"fixture": fixture_id}
        if bookmaker:
            p["bookmaker"] = bookmaker
        return self.get("odds", ttl=600, **p)

    def bookmakers(self):
        return self.get("odds/bookmakers", ttl=-1)

    def injuries(self, fixture_id: int):
        return self.get("injuries", ttl=1800, fixture=fixture_id)

    def lineups(self, fixture_id: int):
        return self.get("fixtures/lineups", ttl=600, fixture=fixture_id)

    def standings(self, league: int, season: int):
        return self.get("standings", ttl=6 * 3600, league=league, season=season)
