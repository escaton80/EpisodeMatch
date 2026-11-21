"""Tests for provider-level HTTP behaviour (TMDB external ids, IMDb bot
protection) observed on real scans (Star Trek Voyager DVDs: IMDb answered
HTTP 202 "Accepted" for the /find scrape — a Cloudflare bot-protection soft
block, not a client bug)."""

from __future__ import annotations

import logging

import httpx
import pytest

from episodematch.core.models import SeriesInfo
from episodematch.providers.imdb_scraper import ImdbScraper
from episodematch.providers.tmdb_client import TmdbClient


class TestTmdbSeasonGuideEnrichesImdbId:
    def test_season_guide_carries_imdb_id_from_external_ids(self) -> None:
        """season_guide must fetch the official /external_ids call so the IMDb
        enrichment does not have to scrape IMDb's /find endpoint (which IMDb
        protects with HTTP 202)."""

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/3/tv/42/season/1":
                return httpx.Response(
                    200,
                    json={
                        "name": "Test Show",
                        "episodes": [
                            {
                                "season_number": 1,
                                "episode_number": 1,
                                "name": "Pilot",
                                "crew": [{"job": "Director", "name": "Jane Doe"}],
                            }
                        ],
                    },
                )
            if path == "/3/tv/42/external_ids":
                return httpx.Response(200, json={"imdb_id": "tt2149175"})
            if path == "/3/tv/42/season/1/credits":
                return httpx.Response(200, json={"cast": [{"name": "Tim Russ"}]})
            return httpx.Response(404)

        client = TmdbClient("test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
        series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id=None, first_air_year=2020)
        guide = client.season_guide(series, 1)
        assert guide.series.imdb_id == "tt2149175"  # enrichment source: official API, not scraping

    def test_external_ids_failure_keeps_guide_usable(self) -> None:
        from episodematch.providers.tmdb_client import TmdbClient

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/3/tv/42/season/1":
                return httpx.Response(200, json={"name": "Test Show", "episodes": []})
            return httpx.Response(500)  # external_ids endpoint broken

        client = TmdbClient("test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
        series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id=None, first_air_year=2020)
        guide = client.season_guide(series, 1)  # must not raise
        assert guide.series.imdb_id is None
        assert [episode.episode for episode in guide.episodes] == []


class TestImdbBotProtection:
    def test_202_is_reported_as_bot_protection_and_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """HTTP 202 "Accepted" is IMDb's bot-protection soft block — the
        scraper reports it clearly at INFO level and degrades to None."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(202, text="<html>challenge page</html>")

        scraper = ImdbScraper(delay_seconds=0.0, client=httpx.Client(transport=httpx.MockTransport(handler)))
        with caplog.at_level(logging.INFO, logger="episodematch.providers.imdb_scraper"):
            assert scraper.find_series_id("Star Trek: Voyager") is None
        assert any("bot protection" in record.message for record in caplog.records)
        assert any("202" in record.message for record in caplog.records)
