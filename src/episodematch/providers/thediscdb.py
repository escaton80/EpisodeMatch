"""TheDiscDB client and local importer (pillar 1: disc fingerprint -> episode map).

Data source
-----------

The community database lives at https://github.com/TheDiscDb/data (MIT licensed) and
is consumed by this client in two ways:

* **GitHub (raw)** — ``data/series/<Series (Year)>/<release>/discNN.json`` files are
  fetched from ``raw.githubusercontent.com``; directory listings come from the
  GitHub contents API and are cached in SQLite.
* **Local clone** — any checkout of the repository can be imported offline.

Disc JSON shape (real example)::

    {
      "Index": 1, "Slug": "blu-ray", "Name": "... Disc 1", "Format": "Blu-Ray",
      "ContentHash": "...", "GlobalDiscId": "...",
      "Titles": [
        {
          "Index": 0,                       <- MakeMKV title number (title_tNN.mkv)
          "Comment": "...", 
          "SourceFile": "00055.mpls",       <- exact MPLS name (MakeMKV SOURCE_ID!)
          "SegmentMap": "1012",
          "Duration": "0:42:36",
          "Size": 12345678,
          "DisplaySize": "11.7 GB",
          "Item": {"Title": "Episode title", "Type": "Episode",
                   "Season": "1", "Episode": "3", "Chapters": []},
          "Tracks": [...]
        }, ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import ProviderError
from episodematch.core.models import DiscFingerprint, DiscPlaylistClip

logger = logging.getLogger(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/TheDiscDb/data/main"
GITHUB_API_BASE = "https://api.github.com/repos/TheDiscDb/data/contents"

_SERIES_LISTING_TTL = 7 * 24 * 3600.0
_DISC_TTL = 14 * 24 * 3600.0

_DURATION_RE = re.compile(r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})$")
_MPLS_RE = re.compile(r"(\d{4,6}\.mpls)", re.IGNORECASE)


def parse_hms_duration(text: str) -> float | None:
    """Parse TheDiscDB duration strings ("0:42:36", "42:36") into seconds."""
    match = _DURATION_RE.match(text.strip())
    if not match:
        return None
    hours = int(match["h"]) if match["h"] else 0
    return hours * 3600 + int(match["m"]) * 60 + int(match["s"])


# ---------------------------------------------------------------------------
# Parsers (pure functions over JSON dicts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscTitleEntry:
    """One ripped title/playlist inside a TheDiscDB disc."""

    make_mkv_index: int
    mpls_name: str | None
    segment_map: str | None
    duration_seconds: float | None
    size_bytes: int | None
    episode_title: str | None
    item_type: str | None
    season: int | None
    episode: int | None
    comment: str | None = None

    @property
    def is_episode(self) -> bool:
        return (self.item_type or "").lower() == "episode" and self.episode is not None


@dataclass(frozen=True)
class ParsedDisc:
    """A fully parsed disc entry."""

    series_title: str
    release_title: str
    disc_name: str
    disc_format: str
    slug: str
    content_hash: str | None
    global_disc_id: str | None
    upc: str | None
    titles: tuple[DiscTitleEntry, ...]
    source: str = "unknown"

    def episodes(self) -> list[DiscTitleEntry]:
        return [title for title in self.titles if title.is_episode]


def parse_disc_json(payload: dict, *, series_title: str, release_title: str, source: str) -> ParsedDisc:
    """Parse one discNN.json document (tolerant of missing fields)."""
    titles: list[DiscTitleEntry] = []
    for index, raw in enumerate(payload.get("Titles") or []):
        item = raw.get("Item") or {}
        duration = parse_hms_duration(str(raw.get("Duration", ""))) if raw.get("Duration") else None
        source_file = raw.get("SourceFile")
        mpls_match = _MPLS_RE.search(str(source_file)) if source_file else None
        season = item.get("Season")
        episode = item.get("Episode")
        titles.append(
            DiscTitleEntry(
                make_mkv_index=_safe_int(raw.get("Index"), default=index),
                mpls_name=mpls_match.group(1).lower() if mpls_match else None,
                segment_map=str(raw["SegmentMap"]) if raw.get("SegmentMap") is not None else None,
                duration_seconds=duration,
                size_bytes=_safe_int(raw.get("Size"), default=None),
                episode_title=item.get("Title"),
                item_type=item.get("Type"),
                season=_safe_int(season, default=None),
                episode=_safe_int(episode, default=None),
                comment=str(raw["Comment"]) if raw.get("Comment") else None,
            )
        )
    return ParsedDisc(
        series_title=series_title,
        release_title=release_title,
        disc_name=str(payload.get("Name") or ""),
        disc_format=str(payload.get("Format") or ""),
        slug=str(payload.get("Slug") or ""),
        content_hash=payload.get("ContentHash"),
        global_disc_id=payload.get("GlobalDiscId"),
        upc=None,
        titles=tuple(titles),
        source=source,
    )


def parse_release_json(payload: dict) -> str | None:
    """Extract the UPC from a release.json document."""
    return payload.get("Upc") or payload.get("UPC")


def _safe_int(value: Any, *, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Matching (pure functions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaylistMatch:
    """One scored (disc title entry, episode) match for a ripped MKV file."""

    disc: ParsedDisc
    entry: DiscTitleEntry
    score: float  # 0..1
    reasons: list[str]

    @property
    def season(self) -> int | None:
        return self.entry.season

    @property
    def episode(self) -> int | None:
        return self.entry.episode


def _duration_close(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def match_fingerprint(
    fingerprint: DiscFingerprint,
    discs: Iterable[ParsedDisc],
    *,
    duration_tolerance: float = 2.0,
    size_tolerance: float = 0.02,
    index_weight: float = 0.25,
) -> list[PlaylistMatch]:
    """Score every disc title entry against a ripped file's fingerprint.

    Score components (capped at 1.0):

    * exact MPLS name match: +0.60 (near-conclusive on its own)
    * MakeMKV title index == disc entry index **corroborated by duration**:
      +``index_weight`` (0.25 normally; with a pinned disc URL 0.90). An index
      match WITHOUT a duration match is ignored — different MakeMKV versions
      enumerate playlists differently, so bare index collisions actively
      produced wrong assignments in live testing (Castle Rock S1D2).
    * duration within tolerance: +0.20 normally, but **+0.60 when exactly ONE
      entry matches the duration** (an individual episode runtime is then
      near-conclusive), +0.35 when two or three share it.
    * size within ±2 %: +0.10 — deliberately weak, because ripped MKVs are
      usually smaller than the on-disc playlist (stripped audio, container
      overhead)
    * segment map count match: +0.05
    """
    mpls = fingerprint.mpls_name.lower() if fingerprint.mpls_name else None
    all_entries = [(disc, entry) for disc in discs for entry in disc.titles]

    # How many entries share this file's duration? A unique duration is a much
    # stronger signal than one colliding among many equal-length episodes.
    duration_matches = sum(
        1
        for _disc, entry in all_entries
        if _duration_close(fingerprint.duration_seconds, entry.duration_seconds, duration_tolerance)
    )
    if duration_matches == 1:
        duration_weight = 0.60
    elif duration_matches in (2, 3):
        duration_weight = 0.35
    else:
        duration_weight = 0.20

    matches: list[PlaylistMatch] = []
    for disc, entry in all_entries:
        score = 0.0
        reasons: list[str] = []
        duration_close = _duration_close(
            fingerprint.duration_seconds, entry.duration_seconds, duration_tolerance
        )
        if mpls and entry.mpls_name and mpls == entry.mpls_name:
            score += 0.60
            reasons.append(f"MPLS {entry.mpls_name} matches disc entry")
        # Audit A2: a title-index match against an entry whose MPLS *contradicts*
        # the rip's known MPLS is a numbering coincidence — it must never add to
        # the score, otherwise it outranks the exact MPLS identity match.
        mpls_conflict = bool(mpls and entry.mpls_name and mpls != entry.mpls_name)
        if (
            fingerprint.make_mkv_index is not None
            and entry.make_mkv_index is not None
            and fingerprint.make_mkv_index == entry.make_mkv_index
            and not mpls_conflict
        ):
            if duration_close:
                score += index_weight
                reasons.append(
                    f"MakeMKV title index {fingerprint.make_mkv_index} matches disc entry index "
                    "(corroborated by duration)"
                )
            else:
                # Bare index collision: the contributor's playlist enumeration
                # differs from this rip's — a wrong-assignment hazard, ignored.
                logger.debug(
                    "Ignoring index match %d (entry %s): duration does not corroborate",
                    fingerprint.make_mkv_index,
                    entry.episode_title or entry.mpls_name,
                )
        if duration_close:
            delta = abs((fingerprint.duration_seconds or 0.0) - (entry.duration_seconds or 0.0))
            if duration_matches == 1:
                # Unique on the whole disc: an individual episode runtime is
                # near-conclusive — no closeness dilution needed.
                score += 0.70
                reasons.append(f"Duration {entry.duration_seconds:.0f}s ±{delta:.1f}s (unique on disc)")
            else:
                closeness = max(0.5, 1.0 - (delta / duration_tolerance) * 0.5) if duration_tolerance else 0.5
                score += duration_weight * closeness
                reasons.append(f"Duration {entry.duration_seconds:.0f}s ±{delta:.1f}s")
        if (
            fingerprint.size_bytes
            and entry.size_bytes
            and entry.size_bytes > 0
            and abs(fingerprint.size_bytes - entry.size_bytes) / entry.size_bytes <= size_tolerance
        ):
            score += 0.10
            reasons.append(f"Size {entry.size_bytes} bytes within {size_tolerance:.0%}")
        if fingerprint.segment_count is not None and entry.segment_map:
            map_count = len(entry.segment_map.split(","))
            if map_count == fingerprint.segment_count:
                score += 0.05
                reasons.append(f"Segment count {map_count}")
        if score > 0.0:
            matches.append(PlaylistMatch(disc=disc, entry=entry, score=min(1.0, score), reasons=reasons))
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Client (GitHub) + local importer
# ---------------------------------------------------------------------------


class TheDiscDbClient:
    """Fetches series/disc data from the TheDiscDb/data GitHub repository."""

    def __init__(
        self,
        cache: CacheDatabase | None = None,
        *,
        client: httpx.Client | None = None,
        github_token: str | None = None,
    ) -> None:
        self._cache = cache
        self._client = client
        self._owns_client = client is None
        self._headers = {"User-Agent": "EpisodeMatch/0.1", "Accept": "application/vnd.github+json"}
        if github_token:
            self._headers["Authorization"] = f"Bearer {github_token}"

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(follow_redirects=True, timeout=30.0, headers=self._headers)
        return self._client

    # -- series discovery ----------------------------------------------------------

    def list_series(self, *, refresh: bool = False) -> list[str]:
        """List series folder names (e.g. ``24 (2001)``), cached for a week."""
        if self._cache is not None and not refresh:
            cached = self._cache.get_json("thediscdb:series-listing")
            if cached is not None:
                return [str(item) for item in cached]
        try:
            response = self.client.get(f"{GITHUB_API_BASE}/data/series")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot list TheDiscDb series: {exc}", provider="thediscdb") from exc
        names = [str(item["name"]) for item in response.json() if item.get("type") == "dir"]
        if self._cache is not None:
            self._cache.put_json("thediscdb:series-listing", names, ttl_seconds=_SERIES_LISTING_TTL)
        return names

    def find_series(self, query: str) -> list[str]:
        """Case-insensitive substring search over series names."""
        needle = query.strip().lower()
        return [name for name in self.list_series() if needle in name.lower()]

    def list_releases(self, series_folder: str, *, refresh: bool = False) -> list[str]:
        """List release folder names of one series (e.g. ``season-1``, ``2022-4k``)."""
        cache_key = f"thediscdb:releases:{series_folder}"
        if self._cache is not None and not refresh:
            cached = self._cache.get_json(cache_key)
            if cached is not None:
                return [str(item) for item in cached]
        encoded = _url_encode_path(f"data/series/{series_folder}")
        try:
            response = self.client.get(f"{GITHUB_API_BASE}/{encoded}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Cannot list releases of {series_folder!r}: {exc}",
                provider="thediscdb",
            ) from exc
        names = [str(item["name"]) for item in response.json() if item.get("type") == "dir"]
        if self._cache is not None:
            self._cache.put_json(cache_key, names, ttl_seconds=_SERIES_LISTING_TTL)
        return names

    # -- disc loading -----------------------------------------------------------------

    def load_release_discs(self, series_folder: str, release_folder: str, *, refresh: bool = False) -> list[ParsedDisc]:
        """Load every discNN.json of one release folder."""
        listing_key = f"thediscdb:files:{series_folder}/{release_folder}"
        file_names: list[str] | None = None
        if self._cache is not None and not refresh:
            cached = self._cache.get_json(listing_key)
            if cached is not None:
                file_names = [str(item) for item in cached]
        if file_names is None:
            encoded = _url_encode_path(f"data/series/{series_folder}/{release_folder}")
            try:
                response = self.client.get(f"{GITHUB_API_BASE}/{encoded}")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Cannot list release {release_folder!r} of {series_folder!r}: {exc}",
                    provider="thediscdb",
                ) from exc
            file_names = [
                str(item["name"]) for item in response.json() if str(item.get("name", "")).startswith("disc") and str(item.get("name", "")).endswith(".json")
            ]
            if self._cache is not None:
                self._cache.put_json(listing_key, file_names, ttl_seconds=_SERIES_LISTING_TTL)

        discs: list[ParsedDisc] = []
        for file_name in sorted(file_names):
            payload = self._fetch_disc_json(series_folder, release_folder, file_name, refresh=refresh)
            if payload is None:
                continue
            release_title = release_folder
            discs.append(
                parse_disc_json(
                    payload,
                    series_title=series_folder,
                    release_title=release_title,
                    source=f"{GITHUB_RAW_BASE}/data/series/{series_folder}/{release_folder}/{file_name}",
                )
            )
        return discs

    def _fetch_disc_json(self, series_folder: str, release_folder: str, file_name: str, *, refresh: bool) -> dict | None:
        cache_key = f"thediscdb:disc:{series_folder}/{release_folder}/{file_name}"
        if self._cache is not None and not refresh:
            cached = self._cache.get_json(cache_key)
            if cached is not None:
                return cached
        url = f"{GITHUB_RAW_BASE}/data/series/{_url_encode_path(series_folder)}/{_url_encode_path(release_folder)}/{file_name}"
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Cannot fetch %s: %s", url, exc)
            return None
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"TheDiscDb file {file_name} is not valid JSON: {exc}", provider="thediscdb") from exc
        if self._cache is not None:
            self._cache.put_json(cache_key, payload, ttl_seconds=_DISC_TTL)
        return payload


class LocalDiscDbImporter:
    """Imports discs from a local clone of TheDiscDb/data (offline usage)."""

    def __init__(self, repo_root: Path | str) -> None:
        self._root = Path(repo_root)
        if not (self._root / "data" / "series").is_dir():
            raise ProviderError(
                f"{self._root} does not look like a TheDiscDb/data checkout (missing data/series)",
                hint="git clone https://github.com/TheDiscDb/data",
            )

    def list_series(self) -> list[str]:
        return sorted(p.name for p in (self._root / "data" / "series").iterdir() if p.is_dir())

    def find_series(self, query: str) -> list[str]:
        needle = query.strip().lower()
        return [name for name in self.list_series() if needle in name.lower()]

    def list_releases(self, series_folder: str) -> list[str]:
        series_dir = self._root / "data" / "series" / series_folder
        if not series_dir.is_dir():
            raise ProviderError(f"Series folder not found: {series_dir}", provider="thediscdb")
        return sorted(p.name for p in series_dir.iterdir() if p.is_dir())

    def load_release_discs(self, series_folder: str, release_folder: str) -> list[ParsedDisc]:
        release_dir = self._root / "data" / "series" / series_folder / release_folder
        if not release_dir.is_dir():
            raise ProviderError(f"Release folder not found: {release_dir}", provider="thediscdb")
        discs: list[ParsedDisc] = []
        for disc_file in sorted(release_dir.glob("disc*.json")):
            try:
                payload = json.loads(disc_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Cannot read %s: %s", disc_file, exc)
                continue
            discs.append(
                parse_disc_json(
                    payload,
                    series_title=series_folder,
                    release_title=release_folder,
                    source=str(disc_file),
                )
            )
        return discs

    def load_all_releases(self, series_folder: str) -> list[ParsedDisc]:
        series_dir = self._root / "data" / "series" / series_folder
        discs: list[ParsedDisc] = []
        for release_dir in sorted(p for p in series_dir.iterdir() if p.is_dir()):
            discs.extend(self.load_release_discs(series_folder, release_dir.name))
        return discs


def _url_encode_path(path: str) -> str:
    """Percent-encode each path segment but keep '/' separators."""
    from urllib.parse import quote

    return "/".join(quote(segment) for segment in path.split("/"))


# ---------------------------------------------------------------------------
# Direct disc/release URL support (e.g. pasted from thediscdb.com)
# ---------------------------------------------------------------------------

_DISC_URL_RE = re.compile(
    r"thediscdb\.com/series/(?P<series>[^/]+)/releases/(?P<release>[^/]+)(?:/discs/(?P<disc>[^/?#]+))?",
    re.IGNORECASE,
)


def normalize_slug(text: str) -> str:
    """Fold folder names and URL slugs onto a comparable form.

    ``"South Park (1997)"`` and ``"south-park-1997"`` both become
    ``"southpark1997"``; release folders are already slugs, so they match
    themselves unchanged.
    """
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def parse_disc_url(url: str) -> dict[str, str | None] | None:
    """Extract (series_slug, release_slug, disc_slug) from a thediscdb.com URL.

    Both levels are accepted: a full disc URL
    (``.../releases/seasons-6-10-blu-ray/discs/season-06-disc-01``) and a
    release URL (without ``/discs/...``, loads every disc of the release).
    Returns None for anything else.
    """
    match = _DISC_URL_RE.search(url or "")
    if match is None:
        return None
    return {
        "series": match.group("series"),
        "release": match.group("release"),
        "disc": match.group("disc"),
    }


def resolve_discs_from_url(
    client: TheDiscDbClient | LocalDiscDbImporter,
    url: str,
) -> tuple[list[ParsedDisc], dict[str, str]]:
    """Load exactly the disc(s) a thediscdb.com URL points at.

    Works with both clients (they share the find_series/list_releases/
    load_release_discs surface). URL slugs are matched against the repository
    folder names via :func:`normalize_slug`.
    """
    parsed = parse_disc_url(url)
    if parsed is None:
        raise ProviderError(
            f"Not a TheDiscDB disc/release URL: {url!r}",
            provider="thediscdb",
            hint="Expected https://thediscdb.com/series/<series>/releases/<release>[/discs/<disc>]",
        )

    series_slug = normalize_slug(str(parsed["series"]))
    series_folders = [name for name in client.find_series(parsed["series"] or "") if normalize_slug(name) == series_slug]
    if not series_folders:
        series_folders = [name for name in client.list_series() if normalize_slug(name) == series_slug]
    if not series_folders:
        raise ProviderError(
            f"TheDiscDb series not found for URL slug {parsed['series']!r}",
            provider="thediscdb",
        )
    series_folder = series_folders[0]

    release_slug = normalize_slug(str(parsed["release"]))
    release_folder = next(
        (name for name in client.list_releases(series_folder) if normalize_slug(name) == release_slug),
        None,
    )
    if release_folder is None:
        available = ", ".join(client.list_releases(series_folder)[:8])
        raise ProviderError(
            f"Release {parsed['release']!r} not found in series {series_folder!r}",
            provider="thediscdb",
            hint=f"Available releases: {available} …",
        )

    discs = client.load_release_discs(series_folder, release_folder)
    disc_slug = parsed["disc"]
    if disc_slug:
        wanted = normalize_slug(disc_slug)
        matching = [disc for disc in discs if normalize_slug(disc.slug or "") == wanted]
        if not matching:
            available = ", ".join(sorted(disc.slug or "?" for disc in discs))
            raise ProviderError(
                f"Disc {disc_slug!r} not found in {series_folder}/{release_folder}",
                provider="thediscdb",
                hint=f"Available discs: {available}",
            )
        discs = matching

    meta = {
        "series_folder": series_folder,
        "release_folder": release_folder,
        "disc_slug": disc_slug or "",
        "url": url,
    }
    return discs, meta


def clips_from_disc(disc: ParsedDisc) -> list[DiscPlaylistClip]:
    """Convert a parsed disc into the generic playlist-clip model."""
    clips: list[DiscPlaylistClip] = []
    for order, entry in enumerate(disc.episodes(), start=1):
        clips.append(
            DiscPlaylistClip(
                season=entry.season or 0,
                episode=entry.episode or 0,
                order=order,
                title=entry.episode_title,
            )
        )
    return clips
