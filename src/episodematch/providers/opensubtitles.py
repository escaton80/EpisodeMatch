"""OpenSubtitles REST client: reference SRTs for dialog matching (pillar 2).

Uses the modern REST API at ``api.opensubtitles.com`` (API key required, free
tier available). Includes the classic OpenSubtitles *movie hash* implementation,
which lets the provider fall back to hash-based search even without a TMDB id.
"""

from __future__ import annotations

import logging
import re
import struct
import time
from pathlib import Path
from typing import Any

import httpx

from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import ProviderError, RateLimitError
from episodematch.core.models import SubtitleLine, Transcript
from episodematch.extractors.subtitle_extractor import parse_srt

logger = logging.getLogger(__name__)

OSDT_API_BASE = "https://api.opensubtitles.com/api/v1"
_HASH_CHUNK = 64 * 1024

#: Reset windows <= this many seconds are treated as short-lived throttling
#: (wait + retry) instead of a hard daily quota (give up).
_THROTTLE_WINDOW_SECONDS = 10.0

#: Detects the free-tier daily limit message ("... allowed 5 subtitles for 24h
#: ... renewed in 16 hours and 28 minutes (2026-11-08 23:59:59 UTC) ...") — the
#: parenthesized UTC datetime is the renewal moment.
_DAILY_QUOTA_RE = re.compile(
    r"allowed\s+\d+\s+subtitles.*?\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC\)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# OpenSubtitles movie hash (pure function, deterministic, testable)
# ---------------------------------------------------------------------------


def compute_opensubtitles_hash(path: Path | str) -> str:
    """Compute the OpenSubtitles 64-bit movie hash of a file.

    Algorithm: ``size + sum of unsigned 64-bit little-endian words`` over the first
    and last 64 KiB, all modulo 2^64. Files smaller than 128 KiB are handled by
    reading the whole file for both halves.
    """
    file_path = Path(path)
    file_size = file_path.stat().st_size
    hash_value = file_size & 0xFFFFFFFFFFFFFFFF
    chunk = _HASH_CHUNK
    with open(file_path, "rb") as handle:
        head = handle.read(chunk)
        if file_size > chunk * 2:
            handle.seek(-chunk, 2)  # os.SEEK_END
            tail = handle.read(chunk)
        else:
            handle.seek(0)
            tail = handle.read(chunk)
    for block in (head, tail):
        words = struct.unpack(f"<{len(block) // 8}Q", block[: len(block) // 8 * 8])
        for word in words:
            hash_value = (hash_value + word) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenSubtitlesClient:
    """Searches and downloads reference subtitles from OpenSubtitles."""

    SOURCE_NAME = "OpenSubtitles"

    def __init__(
        self,
        api_key: str,
        cache: CacheDatabase | None = None,
        *,
        client: httpx.Client | None = None,
        subtitle_dir: Path | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "OpenSubtitles API key is missing",
                provider="opensubtitles",
                hint="Set EPISODEMATCH_OSDT_API_KEY (free at opensubtitles.com).",
            )
        self._api_key = api_key
        self._cache = cache
        self._client = client
        self._owns_client = client is None
        self._subtitle_dir = Path(subtitle_dir) if subtitle_dir else Path.cwd() / "data" / "subtitles"
        self._subtitle_dir.mkdir(parents=True, exist_ok=True)
        # Set when the daily download quota is exhausted (epoch seconds): all
        # further reference downloads fail fast until then.
        self._quota_blocked_until: float = 0.0

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                follow_redirects=True,
                timeout=30.0,
                headers={"Api-Key": self._api_key, "User-Agent": "EpisodeMatch v0.1.0"},
            )
        return self._client

    # -- low level ------------------------------------------------------------------

    @staticmethod
    def _reset_window(response: httpx.Response) -> float:
        """Seconds until the rate-limit window resets (header or default)."""
        raw = response.headers.get("x-ratelimit-reset") or response.headers.get("retry-after") or "1"
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 1.0

    def _get(self, path: str, params: dict[str, Any], *, throttle_retries: int = 2) -> dict:
        return self._request("GET", path, params, throttle_retries=throttle_retries)

    def _post(self, path: str, params: dict[str, Any], *, throttle_retries: int = 2) -> dict:
        return self._request("POST", path, params, throttle_retries=throttle_retries)

    def _request(self, method: str, path: str, params: dict[str, Any], *, throttle_retries: int = 2) -> dict:
        """Request with automatic 429 handling.

        A reset window of a few seconds is OpenSubtitles' per-second throttling —
        waiting and retrying turns those into successes. Long windows are the
        real daily quota and raise :class:`RateLimitError` immediately.

        Note: ``/download`` must be POSTed (GET returns the website's HTML 404).
        """
        attempt = 0
        while True:
            try:
                if method == "POST":
                    response = self.client.post(f"{OSDT_API_BASE}{path}", params=params)
                else:
                    response = self.client.get(f"{OSDT_API_BASE}{path}", params=params)
            except httpx.HTTPError as exc:
                raise ProviderError(f"OpenSubtitles request failed: {exc}", provider="opensubtitles") from exc
            already_blocked = self._quota_blocked_until > time.time()
            if response.status_code == 406:
                # Free-tier daily download limit (5 downloads / 24h) — the body
                # carries the renewal datetime. Block further download attempts
                # for this client instance until then.
                import datetime as _dt

                daily = _DAILY_QUOTA_RE.search(response.text)
                if daily is not None:
                    try:
                        renewal = _dt.datetime.strptime(daily.group(1), "%Y-%m-%d %H:%M:%S").replace(
                            tzinfo=_dt.timezone.utc
                        ).timestamp()
                    except ValueError:
                        renewal = time.time() + 24 * 3600
                    self._quota_blocked_until = renewal
                    if not already_blocked:
                        logger.warning(
                            "OpenSubtitles daily download quota exhausted (free tier: 5 downloads / 24h) — "
                            "no more references until %s. Dialog matching falls back to plot overlap.",
                            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self._quota_blocked_until)),
                        )
                    raise RateLimitError(
                        "OpenSubtitles daily download quota exhausted",
                        provider="opensubtitles",
                        status_code=406,
                    )
            if response.status_code == 429:
                window = self._reset_window(response)
                if window <= _THROTTLE_WINDOW_SECONDS and attempt < throttle_retries:
                    attempt += 1
                    logger.info(
                        "OpenSubtitles throttled (429, window %.1fs) — retry %d/%d for %s",
                        window,
                        attempt,
                        throttle_retries,
                        path,
                    )
                    time.sleep(max(window, 1.0))
                    continue
                if window <= _THROTTLE_WINDOW_SECONDS:
                    raise RateLimitError(
                        f"OpenSubtitles throttled (429) and still limited after {attempt} retries — slow down",
                        provider="opensubtitles",
                        status_code=429,
                    )
                raise RateLimitError(
                    f"OpenSubtitles quota exhausted — window resets in {window:.0f}s (daily download limit)",
                    provider="opensubtitles",
                    status_code=429,
                )
            if response.status_code != 200:
                raise ProviderError(
                    f"OpenSubtitles returned HTTP {response.status_code}: {response.text[:200]}",
                    provider="opensubtitles",
                    status_code=response.status_code,
                )
            return response.json()

    # -- search + download ---------------------------------------------------------------

    def search_episode_subtitles(
        self,
        *,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        season: int | None = None,
        episode: int | None = None,
        languages: str = "en",
        file_hash: str | None = None,
        file_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search subtitles; returns raw result dicts (id, files, release info)."""
        params: dict[str, Any] = {"languages": languages, "type": "episode"}
        if tmdb_id is not None:
            params["tmdb_id"] = str(tmdb_id)
        if imdb_id is not None:
            params["imdb_id"] = imdb_id.replace("tt", "")
        if season is not None:
            params["season_number"] = str(season)
        if episode is not None:
            params["episode_number"] = str(episode)
        if file_hash:
            params["moviehash"] = file_hash
        if file_size:
            params["moviehash_match"] = "true"
            params["moviebytesize"] = str(file_size)
        payload = self._get("/subtitles", params)
        return list(payload.get("data") or [])

    @staticmethod
    def entry_language(result: dict[str, Any]) -> str:
        """Two-letter language of a search result ("" when the API omits it)."""
        attributes = result.get("attributes") or {}
        return str(attributes.get("language") or "").strip().lower()[:2]

    def download_subtitle(self, result: dict[str, Any]) -> Path:
        """Request a download link and store the SRT locally; returns the file path.

        The ``files`` array lives under ``attributes.files`` in real /subtitles
        responses (a top-level ``files`` key is accepted for robustness).
        """
        subtitle_id = str(result.get("id") or "")
        if not subtitle_id:
            raise ProviderError("Subtitle result without id", provider="opensubtitles")
        if self._cache is not None:
            cached = self._cache.get_osdb_subtitle(subtitle_id)
            if cached and cached.get("_srt_path"):
                cached_path = Path(cached["_srt_path"])
                if cached_path.is_file():
                    return cached_path
        attributes = result.get("attributes") or {}
        files = result.get("files") or attributes.get("files") or []
        if not files:
            raise ProviderError(
                f"Subtitle {subtitle_id} has no downloadable files",
                provider="opensubtitles",
            )
        file_id = files[0].get("file_id")
        # The download endpoint requires POST — GET returns the website's HTML 404.
        payload = self._post("/download", {"file_id": str(file_id)})
        link = payload.get("link")
        if not link:
            raise ProviderError(f"OpenSubtitles download link missing for file {file_id}", provider="opensubtitles")
        try:
            response = self.client.get(str(link))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Subtitle download failed: {exc}", provider="opensubtitles") from exc
        target = self._subtitle_dir / f"osdt-{subtitle_id}.srt"
        target.write_bytes(response.content)
        if self._cache is not None:
            self._cache.put_osdb_subtitle(subtitle_id, {"file_id": file_id, "release": payload.get("release")}, target)
        return target

    # -- high level -----------------------------------------------------------------------

    def reference_transcript(
        self,
        *,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        season: int,
        episode: int,
        languages: str | None = None,
    ) -> Transcript | None:
        """Fetch the best reference subtitle for one episode and parse it to a transcript.

        ``languages`` is an OpenSubtitles language spec (ISO 639-1, comma-separated
        for several). Every returned entry's own ``language`` attribute is verified
        against the request — the API filter is trusted only as a hint, not as
        proof, and mismatched entries are skipped. Returns ``None`` when nothing
        suitable is found (the matcher handles that).
        """
        requested = (languages or "en").lower()
        requested_set = {part.strip().lower()[:2] for part in requested.split(",") if part.strip()}
        if self._quota_blocked_until > time.time():
            logger.debug(
                "Skipping OpenSubtitles download for S%02dE%02d — daily quota exhausted until %s",
                season,
                episode,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self._quota_blocked_until)),
            )
            return None
        results = self.search_episode_subtitles(
            tmdb_id=tmdb_id,
            imdb_id=imdb_id,
            season=season,
            episode=episode,
            languages=requested,
        )
        if not results:
            logger.info("OpenSubtitles search returned 0 entries for S%02dE%02d (%s)", season, episode, requested)
            return None
        logger.debug("OpenSubtitles search: %d entries for S%02dE%02d (%s)", len(results), season, episode, requested)

        # Prefer results with a download signal; fall back to first.
        def rank(item: dict[str, Any]) -> tuple[int, int]:
            attributes = item.get("attributes") or {}
            try:
                downloads = int(attributes.get("download_count") or 0)
            except (TypeError, ValueError):
                downloads = 0
            has_files = 1 if (item.get("files") or attributes.get("files")) else 0
            return (has_files, downloads)

        for item in sorted(results, key=rank, reverse=True):
            item_attributes = item.get("attributes") or {}
            item_language = self.entry_language(item)
            if item_language and requested_set and item_language not in requested_set:
                logger.debug(
                    "Skipping subtitle %s in language %r (requested %r)",
                    item.get("id"),
                    item_language,
                    requested,
                )
                continue
            try:
                srt_path = self.download_subtitle(item)
            except RateLimitError:
                # Quota/throttle block active (already logged by _request) —
                # stop the candidate loop; further attempts would 406 too.
                break
            except ProviderError as exc:
                if exc.status_code == 402:
                    logger.warning(
                        "OpenSubtitles daily download quota exhausted (402) for S%02dE%02d — "
                        "dialog matching falls back to plot overlap.",
                        season,
                        episode,
                    )
                    break
                logger.info("Skipping OpenSubtitles candidate %s for S%02dE%02d: %s", item.get("id"), season, episode, exc)
                continue
            try:
                text = srt_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.debug("Cannot read downloaded SRT: %s", exc)
                continue
            lines: list[SubtitleLine] = parse_srt(text)
            if not lines:
                # A downloaded file with zero parsable lines is a broken/HTML
                # response — skip it loudly instead of failing the episode
                # silently, then let the loop try the next candidate.
                logger.warning(
                    "OpenSubtitles reference for S%02dE%02d (id %s, release %r) parsed to 0 lines — skipping this candidate",
                    season,
                    episode,
                    item.get("id"),
                    (item_attributes.get("release") or "?"),
                )
                continue
            logger.info(
                "OpenSubtitles reference downloaded: %d lines (S%02dE%02d, %s, release %r)",
                len(lines),
                season,
                episode,
                item_language or next(iter(requested_set), requested),
                (item_attributes.get("release") or "?"),
            )
            return Transcript(
                source="osdt-reference",
                language=item_language or next(iter(requested_set), requested),
                lines=lines,
            )
        logger.info(
            "No usable OpenSubtitles reference after %d candidates (S%02dE%02d, %s)",
            len(results),
            season,
            episode,
            requested,
        )
        return None
