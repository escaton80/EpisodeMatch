"""SubDL REST client: alternative reference-subtitle source (pillar 2).

SubDL (https://subdl.com) offers a much more generous free tier than
OpenSubtitles (2,000 API requests/day; anonymous downloads 300/day per IP),
which makes it the practical choice when the OpenSubtitles 5-downloads-per-24h
limit is exhausted. API key from the SubDL account panel.

Search:  GET https://api.subdl.com/api/v1/subtitles
             ?api_key=…&tmdb_id=…&type=tv&season_number=…&episode_number=…&languages=EN,DE
Download: https://dl.subdl.com/<relative url from the result>  → ZIP archive
          containing .srt files (extracted here with the stdlib zipfile).
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import httpx

from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import ProviderError, RateLimitError
from episodematch.core.models import SubtitleLine, Transcript
from episodematch.extractors.subtitle_extractor import parse_srt

logger = logging.getLogger(__name__)

SUBDL_API_BASE = "https://api3.subdl.com/api/v1"  # api.subdl.com does not resolve (stale docs)
SUBDL_DOWNLOAD_HOST = "https://dl.subdl.com"


def to_subdl_languages(languages: str | None) -> str:
    """Normalize an OpenSubtitles-style language spec ("de,en") to SubDL's
    uppercase 2-letter form ("EN,DE"). Unknown codes pass through uppercase."""
    parts: list[str] = []
    for code in (languages or "").split(","):
        cleaned = code.strip().upper()
        if cleaned and cleaned not in parts:
            parts.append(cleaned[:2] if len(cleaned) > 2 else cleaned)
    return ",".join(parts)


class SubDlClient:
    """Searches and downloads reference subtitles from SubDL.

    Implements the same ``reference_transcript`` contract as
    :class:`~episodematch.providers.opensubtitles.OpenSubtitlesClient`, so the
    pipeline can swap between sources via configuration.
    """

    SOURCE_NAME = "SubDL"

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
                "SubDL API key is missing",
                provider="subdl",
                hint="Set EPISODEMATCH_SUBDL_API_KEY or add it in the settings view (free at subdl.com).",
            )
        self._api_key = api_key
        self._cache = cache
        self._client = client
        self._owns_client = client is None
        self._subtitle_dir = Path(subtitle_dir) if subtitle_dir else Path.cwd() / "data" / "subtitles"
        self._subtitle_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                follow_redirects=True,
                timeout=30.0,
                headers={"User-Agent": "EpisodeMatch v0.1.0", "Accept": "application/json"},
            )
        return self._client

    # -- low level ------------------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any]) -> dict:
        try:
            response = self.client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise ProviderError(f"SubDL request failed: {exc}", provider="subdl") from exc
        if response.status_code in (402, 429):
            raise RateLimitError(
                f"SubDL limit hit (HTTP {response.status_code}): {response.text[:150]}",
                provider="subdl",
                status_code=response.status_code,
            )
        if response.status_code != 200:
            raise ProviderError(
                f"SubDL returned HTTP {response.status_code}: {response.text[:150]}",
                provider="subdl",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            # Audit B6: proxies/CDNs sometimes answer 200 with an HTML error page.
            raise ProviderError(
                f"SubDL returned a 200 response that is not JSON: {response.text[:150]!r}",
                provider="subdl",
            ) from exc
        if payload.get("status") is False:
            error = str(payload.get("error") or "unknown SubDL error")
            raise ProviderError(f"SubDL error: {error}", provider="subdl")
        return payload

    def search_episode_subtitles(
        self,
        *,
        tmdb_id: int | None = None,
        imdb_id: str | None = None,
        season: int | None = None,
        episode: int | None = None,
        languages: str = "en",
    ) -> list[dict[str, Any]]:
        """Search episode subtitles; returns the raw ``subtitles`` result dicts."""
        params: dict[str, Any] = {"api_key": self._api_key, "type": "tv", "languages": to_subdl_languages(languages)}
        if tmdb_id is not None:
            params["tmdb_id"] = str(tmdb_id)
        elif imdb_id is not None:
            params["imdb_id"] = imdb_id.replace("tt", "")
        else:
            raise ProviderError("SubDL search needs tmdb_id or imdb_id", provider="subdl")
        if season is not None:
            params["season_number"] = str(season)
        if episode is not None:
            params["episode_number"] = str(episode)
        payload = self._get(f"{SUBDL_API_BASE}/subtitles", params)
        return list(payload.get("subtitles") or [])

    # -- download ---------------------------------------------------------------------

    @staticmethod
    def extract_srt_members(archive_bytes: bytes) -> list[str]:
        """Extract the text of every .srt member from a downloaded ZIP archive."""
        texts: list[str] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            for member in zf.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".srt"):
                    continue
                texts.append(zf.read(member).decode("utf-8", errors="replace"))
        return texts

    def _download_candidate(self, result: dict[str, Any]) -> list[str] | None:
        """Download one result (ZIP or raw SRT) and return parsed-ready texts."""
        relative_url = str(result.get("url") or "")
        if not relative_url:
            return None
        url = relative_url if relative_url.startswith("http") else f"{SUBDL_DOWNLOAD_HOST}{relative_url}"
        if self._cache is not None:
            cached = self._cache.get_json(f"subdl:content:{url}")
            if cached is not None:
                return [cached]
        try:
            response = self.client.get(url)
        except httpx.HTTPError as exc:
            logger.debug("SubDL download failed for %s: %s", url, exc)
            return None
        if response.status_code != 200 or not response.content:
            logger.debug("SubDL download returned HTTP %s for %s", response.status_code, url)
            return None
        content = response.content
        texts: list[str]
        if content[:2] == b"PK":  # ZIP archive (the common case)
            try:
                texts = self.extract_srt_members(content)
            except (zipfile.BadZipFile, OSError) as exc:
                logger.debug("SubDL archive unreadable: %s", exc)
                return None
        else:
            texts = [content.decode("utf-8", errors="replace")]
        if self._cache is not None and texts:
            # Cache the primary text for re-runs (quota-friendly).
            self._cache.put_json(f"subdl:content:{url}", texts[0], ttl_seconds=14 * 24 * 3600)
        return texts or None

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
        """Fetch the best reference subtitle for one episode and parse it to a
        transcript — same contract as the OpenSubtitles client. Returns None
        when nothing usable is found (the matcher handles that)."""
        requested = (languages or "en").lower()
        results = self.search_episode_subtitles(
            tmdb_id=tmdb_id,
            imdb_id=imdb_id,
            season=season,
            episode=episode,
            languages=requested,
        )
        if not results:
            logger.info("SubDL search returned 0 entries for S%02dE%02d (%s)", season, episode, requested)
            return None
        logger.debug("SubDL search: %d entries for S%02dE%02d (%s)", len(results), season, episode, requested)

        requested_languages = {part.strip().upper() for part in requested.split(",") if part.strip()}

        def candidate_language(item: dict[str, Any]) -> str:
            return str(item.get("lang") or item.get("language") or "").strip().upper()

        def candidate_ok(item: dict[str, Any]) -> bool:
            """Audit E1: SubDL's search filter is loose and may return releases
            for other episodes/languages. Candidates that explicitly disagree
            are rejected; missing fields are tolerated (they are optional)."""
            candidate_episode = item.get("episode")
            if candidate_episode is not None and str(candidate_episode) != str(episode):
                return False
            candidate_season = item.get("season")
            if candidate_season is not None and str(candidate_season) != str(season):
                return False
            language = candidate_language(item)
            if language and requested_languages and language not in requested_languages:
                return False
            return True

        usable = [item for item in results if candidate_ok(item)]
        if not usable:
            logger.info(
                "SubDL: all %d candidates rejected by episode/language validation (S%02dE%02d, %s)",
                len(results),
                season,
                episode,
                requested,
            )
            return None

        # Prefer results with an explicit episode match and the most releases.
        def rank(item: dict[str, Any]) -> tuple[int, int]:
            episode_match = 1 if str(item.get("episode") or "") == str(episode) else 0
            has_url = 1 if item.get("url") else 0
            return (has_url, episode_match)

        for item in sorted(usable, key=rank, reverse=True):
            texts = self._download_candidate(item)
            if not texts:
                logger.debug("Skipping SubDL candidate %r (no content)", item.get("release_name"))
                continue
            best: list[SubtitleLine] = []
            for text in texts:
                parsed = parse_srt(text)
                if len(parsed) > len(best):
                    best = parsed
            if not best:
                logger.warning(
                    "SubDL reference for S%02dE%02d (release %r) parsed to 0 lines — skipping this candidate",
                    season,
                    episode,
                    item.get("release_name") or "?",
                )
                continue
            logger.info(
                "SubDL reference downloaded: %d lines (S%02dE%02d, release %r)",
                len(best),
                season,
                episode,
                item.get("release_name") or "?",
            )
            # Audit E1: label the transcript with the language the candidate
            # actually claims (fallback: the first requested language).
            actual_language = (
                str(item.get("lang") or item.get("language") or "").strip().lower()
                or requested.split(",")[0]
            )
            return Transcript(
                source="subdl",
                language=actual_language,
                lines=best,
            )
        logger.info(
            "No usable SubDL reference after %d candidates (S%02dE%02d, %s)",
            len(usable),
            season,
            episode,
            requested,
        )
        return None
