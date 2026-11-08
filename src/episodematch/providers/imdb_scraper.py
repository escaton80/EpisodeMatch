"""IMDb scraper: season episode lists, plots and guest stars (HTML + BeautifulSoup).

IMDb has no public API, so this provider scrapes the public ``/title/<id>/episodes/``
pages with a browser-like User-Agent, politeness delays and SQLite caching. All
parsing is done on tolerant CSS selectors; failures degrade gracefully (the TMDB
guide remains the primary source, IMDb enriches it).
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

import httpx

from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import ProviderError
from episodematch.core.models import EpisodeInfo, SeasonGuide

logger = logging.getLogger(__name__)

_IMDB_BASE = "https://www.imdb.com"
_IMDB_TITLE_ID = re.compile(r"^tt(\d{5,})$")
_EPISODE_BLOCK_TTL = 3 * 24 * 3600.0

_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
)

_IMDB_ID_RE = re.compile(r"/title/(tt\d+)/")


def is_imdb_id(text: str) -> bool:
    """True when ``text`` is a canonical IMDb title id (``tt1234567``)."""
    return bool(_IMDB_TITLE_ID.match(text.strip()))


# ---------------------------------------------------------------------------
# Pure HTML parsing (unit-testable without network)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScrapedEpisode:
    episode_number: int | None
    title: str
    plot: str | None
    air_date: str | None
    rating: float | None
    imdb_id: str | None


def _text_of(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ")).strip() if node is not None else ""


def parse_episodes_page(html: str) -> list[ScrapedEpisode]:
    """Parse an IMDb ``/episodes/?season=N`` page into episode summaries.

    IMDb changes markup regularly; several selector strategies are tried in order
    and the first producing results wins.
    """
    soup = BeautifulSoup(html, "html.parser")
    strategies = (
        _parse_episodes_json_ld,
        _parse_episodes_article_blocks,
    )
    for strategy in strategies:
        episodes = strategy(soup)
        if episodes:
            return episodes
    return []


def _parse_episodes_json_ld(soup: BeautifulSoup) -> list[ScrapedEpisode]:
    episodes: list[ScrapedEpisode] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json

            payload = json.loads(script.string or "{}")
        except (ValueError, TypeError):
            continue
        items = payload.get("itemListElement") or []
        for position, wrapper in enumerate(items, start=1):
            item = wrapper.get("item") or {}
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            url = str(item.get("url") or "")
            id_match = _IMDB_ID_RE.search(url)
            rating_block = item.get("aggregateRating") or {}
            try:
                rating = float(rating_block["ratingValue"]) if rating_block.get("ratingValue") else None
            except (TypeError, ValueError):
                rating = None
            episode_number = item.get("episodeNumber")
            if isinstance(episode_number, dict):
                episode_number = episode_number.get("episodeNumber")
            episodes.append(
                ScrapedEpisode(
                    episode_number=int(episode_number) if str(episode_number or "").isdigit() else position,
                    title=name,
                    plot=item.get("description") or None,
                    air_date=item.get("datePublished") or None,
                    rating=rating,
                    imdb_id=id_match.group(1) if id_match else None,
                )
            )
    return episodes


def _parse_episodes_article_blocks(soup: BeautifulSoup) -> list[ScrapedEpisode]:
    episodes: list[ScrapedEpisode] = []
    # Modern markup: each episode is an article with data-testid="list-item".
    blocks = soup.find_all("article")
    for index, block in enumerate(blocks, start=1):
        header = block.find(["h4", "h3", "div"], attrs={"data-testid": "ipc-title__text"})
        title_text = _text_of(header)
        if not title_text:
            continue
        # Title text often looks like "1. Pilot".
        match = re.match(r"^(?P<num>\d+)\.\s*(?P<title>.+)$", title_text)
        number = int(match["num"]) if match and match["num"].isdigit() else index
        title = match["title"] if match else title_text
        link = block.find("a", href=_IMDB_ID_RE)
        id_match = _IMDB_ID_RE.search(str(link.get("href"))) if link else None
        plot_node = block.find("div", attrs={"data-testid": "plot-xl"}) or block.find(
            "div", attrs={"data-testid": "review-overflow"}
        )
        date_node = block.find(attrs={"data-testid": "air-date"})
        rating_node = block.find("span", class_="ipc-rating-star")
        rating = None
        if rating_node:
            rating_match = re.search(r"(\d+[.,]\d+)", _text_of(rating_node))
            if rating_match:
                rating = float(rating_match.group(1).replace(",", "."))
        episodes.append(
            ScrapedEpisode(
                episode_number=number,
                title=title,
                plot=_text_of(plot_node) or None,
                air_date=_text_of(date_node) or None,
                rating=rating,
                imdb_id=id_match.group(1) if id_match else None,
            )
        )
    return episodes


def parse_series_search(html: str) -> list[tuple[str, str]]:
    """Parse an IMDb find-page into (title_id, title) tuples."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href*='/title/tt']"):
        href = anchor.get("href") or ""
        match = _IMDB_ID_RE.search(href)
        if not match or match.group(1) in seen:
            continue
        title = _text_of(anchor)
        if not title or len(title) < 2:
            continue
        seen.add(match.group(1))
        results.append((match.group(1), title))
    return results


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class ImdbScraper:
    """Polite IMDb scraper with caching and graceful degradation."""

    def __init__(
        self,
        cache: CacheDatabase | None = None,
        *,
        client: httpx.Client | None = None,
        delay_seconds: float = 1.5,
    ) -> None:
        self._cache = cache
        self._client = client
        self._owns_client = client is None
        self._delay = delay_seconds
        self._last_request = 0.0

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                follow_redirects=True,
                timeout=30.0,
                headers={"User-Agent": random.choice(_USER_AGENTS), "Accept-Language": "en-US,en;q=0.9"},
            )
        return self._client

    def _polite_get(self, url: str, *, cache_key: str | None = None, ttl: float = _EPISODE_BLOCK_TTL) -> str | None:
        if self._cache is not None and cache_key:
            cached = self._cache.get_json(cache_key)
            if cached is not None:
                return str(cached)
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        try:
            response = self.client.get(url)
        except httpx.HTTPError as exc:
            logger.warning("IMDb request failed (%s): %s", url, exc)
            return None
        self._last_request = time.monotonic()
        if response.status_code != 200:
            logger.warning("IMDb returned HTTP %s for %s", response.status_code, url)
            return None
        text = response.text
        if self._cache is not None and cache_key:
            self._cache.put_json(cache_key, text, ttl_seconds=ttl)
        return text

    # -- public API -----------------------------------------------------------------

    def find_series_id(self, series_name: str) -> str | None:
        """Find the main series title id via IMDb's public find endpoint."""
        from urllib.parse import quote

        html = self._polite_get(f"{_IMDB_BASE}/find/?q={quote(series_name)}&s=tt&ttype=tv", cache_key=f"imdb:find:{series_name.lower()}")
        if not html:
            return None
        results = parse_series_search(html)
        return results[0][0] if results else None

    def season_episodes(self, series_imdb_id: str, season: int) -> list[ScrapedEpisode]:
        """Scrape one season's episode list (empty list on failure - degrade softly)."""
        html = self._polite_get(
            f"{_IMDB_BASE}/title/{series_imdb_id}/episodes/?season={season}",
            cache_key=f"imdb:episodes:{series_imdb_id}:{season}",
        )
        if not html:
            return []
        episodes = parse_episodes_page(html)
        if not episodes:
            logger.warning("IMDb episodes page for %s season %d parsed to 0 episodes", series_imdb_id, season)
        return episodes

    def enrich_season_guide(self, guide: SeasonGuide, series_imdb_id: str | None = None) -> SeasonGuide:
        """Merge IMDb episode data (plots, air dates, ids) into a TMDB season guide."""
        imdb_id = series_imdb_id or guide.series.imdb_id
        if not imdb_id:
            return guide
        scraped = self.season_episodes(imdb_id, guide.season)
        if not scraped:
            return guide
        by_number = {ep.episode_number: ep for ep in scraped if ep.episode_number is not None}
        enriched: list[EpisodeInfo] = []
        for episode in guide.episodes:
            extra = by_number.get(episode.episode)
            if extra is not None:
                episode = episode.model_copy(
                    update={
                        "long_synopsis": extra.plot or episode.long_synopsis,
                        "imdb_id": extra.imdb_id or episode.imdb_id,
                        "air_date": extra.air_date or episode.air_date,
                    }
                )
            enriched.append(episode)
        return guide.model_copy(update={"episodes": enriched})

    def fetch_series_id_for_guide(self, guide: SeasonGuide) -> str | None:
        """Resolve the IMDb id for a guide's series (explicit id -> find page)."""
        if guide.series.imdb_id:
            return guide.series.imdb_id
        return self.find_series_id(guide.series.name)


__all__ = [
    "ImdbScraper",
    "ScrapedEpisode",
    "is_imdb_id",
    "parse_episodes_page",
    "parse_series_search",
]
