"""TMDB API client (series, seasons, episodes, crew, guest stars).

Uses TMDB API v3 (https://api.themoviledb.org/3) with either an API key
(``?api_key=``) or a v4 read-access token (``Authorization: Bearer``). Requests are
rate limited with a token bucket and cached in SQLite with a TTL.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import ProviderError, RateLimitError
from episodematch.core.models import (
    EpisodeInfo,
    GuestStar,
    SeasonGuide,
    SeriesInfo,
)

logger = logging.getLogger(__name__)

TMDB_API_BASE = "https://api.themoviedb.org/3"
_DEFAULT_TTL = 7 * 24 * 3600.0
_SEARCH_TTL = 30 * 24 * 3600.0


class TokenBucket:
    """Simple thread-safe token bucket (default: TMDB allows ~50 req/s)."""

    def __init__(self, rate_per_second: float = 40.0, capacity: float = 40.0) -> None:
        self._rate = rate_per_second
        self._capacity = capacity
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self._rate
            time.sleep(min(wait, 0.5))


class TmdbClient:
    """Thin, cached TMDB v3 client returning EpisodeMatch models."""

    def __init__(
        self,
        api_key: str,
        cache: CacheDatabase | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "TMDB API key is missing",
                provider="tmdb",
                hint="Set EPISODEMATCH_TMDB_API_KEY or add it in the settings view (free at themoviedb.org).",
            )
        self._api_key = api_key
        self._cache = cache
        self._client = client
        self._owns_client = client is None
        self._bucket = TokenBucket()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(follow_redirects=True, timeout=30.0)
        return self._client

    # -- low-level ----------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None, *, cache_ttl: float = _DEFAULT_TTL) -> dict:
        cache_key = f"tmdb:{path}:{sorted((params or {}).items())}"
        if self._cache is not None:
            cached = self._cache.get_json(cache_key)
            if cached is not None:
                return cached
        request_params = dict(params or {})
        if not self._api_key.startswith("ey"):  # v4 JWT tokens go into the header
            request_params["api_key"] = self._api_key
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key.startswith("ey") else None
        self._bucket.acquire()
        try:
            response = self.client.get(f"{TMDB_API_BASE}{path}", params=request_params, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"TMDB request failed: {exc}", provider="tmdb") from exc
        if response.status_code == 429:
            raise RateLimitError("TMDB rate limit hit", provider="tmdb", status_code=429)
        if response.status_code != 200:
            raise ProviderError(
                f"TMDB returned HTTP {response.status_code} for {path}: {response.text[:200]}",
                provider="tmdb",
                status_code=response.status_code,
            )
        payload = response.json()
        if self._cache is not None:
            self._cache.put_json(cache_key, payload, ttl_seconds=cache_ttl)
        return payload

    # -- high level ------------------------------------------------------------------

    def search_series(self, query: str, *, year: int | None = None) -> list[SeriesInfo]:
        params: dict[str, Any] = {"query": query, "include_adult": "false"}
        if year:
            params["first_air_date_year"] = str(year)
        payload = self._get("/search/tv", params, cache_ttl=_SEARCH_TTL)
        results: list[SeriesInfo] = []
        for item in payload.get("results") or []:
            first_air = str(item.get("first_air_date") or "")
            year_value = int(first_air[:4]) if first_air[:4].isdigit() else None
            results.append(
                SeriesInfo(
                    name=str(item.get("name") or item.get("original_name") or ""),
                    tmdb_id=item.get("id"),
                    first_air_year=year_value,
                    overview=item.get("overview") or None,
                )
            )
        return results

    def series_details(self, tmdb_id: int) -> SeriesInfo:
        payload = self._get(f"/tv/{tmdb_id}")
        external = {}
        try:
            external = self._get(f"/tv/{tmdb_id}/external_ids")
        except ProviderError as exc:
            logger.debug("External ids unavailable for TMDB %s: %s", tmdb_id, exc)
        first_air = str(payload.get("first_air_date") or "")
        return SeriesInfo(
            name=str(payload.get("name") or ""),
            tmdb_id=tmdb_id,
            imdb_id=external.get("imdb_id"),
            first_air_year=int(first_air[:4]) if first_air[:4].isdigit() else None,
            overview=payload.get("overview") or None,
        )

    def find_series(self, query: str, *, year: int | None = None) -> SeriesInfo:
        """Search and return the best match (raises ProviderError when nothing found)."""
        results = self.search_series(query, year=year)
        if not results:
            raise ProviderError(f"No TMDB series found for {query!r}", provider="tmdb")
        if year is not None:
            exact = [info for info in results if info.first_air_year == year]
            if exact:
                return exact[0]
        return results[0]

    def season_guide(self, series: SeriesInfo, season: int) -> SeasonGuide:
        """Fetch the complete episode guide for one season."""
        if series.tmdb_id is None:
            raise ProviderError(f"Series {series.name!r} has no TMDB id", provider="tmdb")
        payload = self._get(f"/tv/{series.tmdb_id}/season/{season}")
        # Carry the series' IMDb id from the official /external_ids endpoint:
        # the IMDb enrichment then goes straight to /title/{imdb_id}/episodes/
        # instead of scraping IMDb's /find endpoint, which is protected by bot
        # mitigations (HTTP 202 soft blocks) and fails on real scans.
        if series.imdb_id is None:
            try:
                external = self._get(f"/tv/{series.tmdb_id}/external_ids")
                imdb_id = external.get("imdb_id")
                if imdb_id:
                    series.imdb_id = imdb_id
            except ProviderError as exc:
                logger.debug("External ids unavailable for TMDB %s: %s", series.tmdb_id, exc)
        episodes: list[EpisodeInfo] = []
        for raw in payload.get("episodes") or []:
            crew = raw.get("crew") or []
            directors: list[str] = []
            writers: list[str] = []
            for member in crew:
                job = str(member.get("job") or "").lower()
                name = str(member.get("name") or "").strip()
                if not name:
                    continue
                if job == "director" and name not in directors:
                    directors.append(name)
                if job in ("writer", "screenplay", "story", "teleplay") and name not in writers:
                    writers.append(name)
            guests = [
                GuestStar(name=str(person.get("name") or ""), character=person.get("character") or None)
                for person in (raw.get("guest_stars") or [])
                if person.get("name")
            ]
            runtime_list = raw.get("runtime")
            if isinstance(runtime_list, list):
                runtime = runtime_list[0] if runtime_list else None
            else:
                runtime = runtime_list
            episodes.append(
                EpisodeInfo(
                    season=int(raw.get("season_number") or season),
                    episode=int(raw.get("episode_number") or 0),
                    title=str(raw.get("name") or ""),
                    plot=raw.get("overview") or None,
                    air_date=raw.get("air_date") or None,
                    runtime_seconds=int(runtime) * 60 if runtime else None,
                    directors=directors,
                    writers=writers,
                    guest_stars=guests,
                    tmdb_id=raw.get("id"),
                )
            )
        guide = SeasonGuide(series=series, season=season, episodes=episodes)
        # PILLAR4_PLAN batch 3.5: series-level cast names for the visual pillar's
        # hint-only evidence (main-cast cards confirm the series, no score).
        # The season-DETAILS payload has no cast — it lives on the dedicated
        # season credits endpoint.
        try:
            credits_payload = self._get(f"/tv/{series.tmdb_id}/season/{season}/credits")
            guide.cast = [
                str(member.get("name"))
                for member in credits_payload.get("cast") or []
                if member.get("name")
            ]
        except ProviderError as exc:
            logger.debug("Season credits unavailable for TMDB %s: %s", series.tmdb_id, exc)
        return guide
