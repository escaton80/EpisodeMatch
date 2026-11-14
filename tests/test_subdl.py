"""Tests for the SubDL alternative subtitle source and the source dispatch."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from episodematch.core.config import AppConfig, apply_env_overrides
from episodematch.core.exceptions import ProviderError, RateLimitError
from episodematch.providers.subdl import SubDlClient, to_subdl_languages
from tests.test_pipeline import FakeFfprobe, FakeTmdb, make_guide

_SRT_BYTES = (
    "1\n00:00:01,000 --> 00:00:02,500\nSubDL reference line\n\n"
    "2\n00:00:05,000 --> 00:00:06,000\nSecond SubDL line\n"
).encode("utf-8")


def _zip_bytes(*members: tuple[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buffer.getvalue()


def _subtitle_entry(sub_id: str, url: str, *, episode: Any = 1) -> dict[str, Any]:
    return {"release_name": f"Release-{sub_id}", "name": f"name-{sub_id}", "url": url, "episode": episode}


def _make_client(handler) -> SubDlClient:
    transport = httpx.MockTransport(handler)
    return SubDlClient("subdl-key", client=httpx.Client(transport=transport))


class TestToSubdlLanguages:
    def test_uppercase_two_letter(self) -> None:
        assert to_subdl_languages("de,en") == "DE,EN"

    def test_iso_639_2_mapped(self) -> None:
        assert to_subdl_languages("ger,eng") == "GE,EN"

    def test_empty(self) -> None:
        assert to_subdl_languages(None) == ""


class TestSubDlClient:
    def test_search_params_and_zip_download(self) -> None:
        """Search must pass api_key/tmdb_id/type=tv/season/episode/languages(upper);
        the ZIP download is extracted and the best SRT becomes the transcript."""
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            requests.append(url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                assert "api_key=subdl-key" in url
                assert "tmdb_id=1402" in url
                assert "type=tv" in url
                assert "season_number=1" in url
                assert "episode_number=1" in url
                assert "languages=DE" in url
                return httpx.Response(
                    200,
                    json={
                        "status": True,
                        "subtitles": [
                            _subtitle_entry("s1", "/subtitle/1-1.zip"),
                            _subtitle_entry("s2", "/subtitle/2-2.zip"),
                        ],
                    },
                )
            if url == "https://dl.subdl.com/subtitle/1-1.zip":
                return httpx.Response(200, content=_zip_bytes(("eps/sub.srt", _SRT_BYTES), ("readme.txt", b"junk")))
            if url == "https://dl.subdl.com/subtitle/2-2.zip":
                return httpx.Response(200, content=_zip_bytes(("empty.srt", b"")))
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1402, season=1, episode=1, languages="de")
        assert transcript is not None
        assert transcript.source == "subdl"
        assert transcript.language == "de"
        assert [line.text for line in transcript.lines] == ["SubDL reference line", "Second SubDL line"]

    def test_raw_srt_download_tolerated(self) -> None:
        """A result whose url serves a plain SRT (not a ZIP) also works."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                return httpx.Response(200, json={"status": True, "subtitles": [_subtitle_entry("s1", "/subtitle/1.srt")]})
            if url == "https://dl.subdl.com/subtitle/1.srt":
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is not None and len(transcript.lines) == 2

    def test_status_false_raises_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": False, "error": "invalid api key"})

        client = _make_client(handler)
        with pytest.raises(ProviderError, match="invalid api key"):
            client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")

    def test_no_results_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": True, "subtitles": []})

        client = _make_client(handler)
        assert client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de") is None

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ProviderError, match="SubDL API key"):
            SubDlClient("")


# ---------------------------------------------------------------------------
# Source dispatch in the pipeline
# ---------------------------------------------------------------------------


class FakeSubDl:
    SOURCE_NAME = "SubDL"

    def __init__(self) -> None:
        self.requested: list[str | None] = []

    def reference_transcript(self, *, tmdb_id=None, imdb_id=None, season=None, episode=None, languages=None):  # noqa: ANN001, ANN202
        self.requested.append(languages)
        from episodematch.core.models import SubtitleLine, Transcript

        return Transcript(source="subdl", language="de", lines=[SubtitleLine(start_seconds=0, end_seconds=1, text="x" * 30)])


class TestPipelineSourceDispatch:
    def _run(self, tmp_path: Path, config: AppConfig, **pipeline_kwargs: Any):
        from episodematch.matcher.pipeline import Pipeline, ScanOptions
        from tests.test_pipeline import FakeFfprobe, FakeTmdb, make_guide

        mkv_payload = {
            "format": {"duration": "2500", "size": "1000"},
            "streams": [
                {"index": 0, "codec_name": "h264", "codec_type": "video"},
                {"index": 1, "codec_name": "subrip", "codec_type": "subtitle", "tags": {"language": "ger"}},
            ],
        }
        mkv = __import__(
            "episodematch.extractors.ffprobe_reader", fromlist=["parse_ffprobe_json"]
        ).parse_ffprobe_json(mkv_payload, path=tmp_path / "title_t00.mkv")

        class FakeExtractor:
            def extract_all(self, mkv, *, duration_seconds=None, window_minutes=None, languages_first=None, ocr_max_events=40):  # noqa: ANN001, ANN202
                from episodematch.core.models import SubtitleLine, Transcript

                return [
                    Transcript(
                        source="srt",
                        language="ger",
                        lines=[SubtitleLine(start_seconds=0.0, end_seconds=1.0, text="a long enough german line here")],
                    )
                ]

        folder = tmp_path / "rips"
        folder.mkdir()
        (folder / "title_t00.mkv").write_bytes(b"\x1a\x45\xdf\xa3fake")

        pipeline_kwargs.setdefault("subtitle_extractor", FakeExtractor())
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=None,
            tmdb=FakeTmdb(make_guide()),
            **pipeline_kwargs,
        )
        return pipeline.scan_folder(folder, series="Test Show", season=1, options=ScanOptions(depth="fast", use_disc_db=False))

    def test_config_subdl_source_dispatches_to_subdl_client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = AppConfig(bin_dir=tmp_path / "bin", data_dir=tmp_path / "data", subtitle_source="subdl", subdl_api_key="subdl-key")
        fake = FakeSubDl()
        self._run(tmp_path, config, subdl=fake)
        # one request per candidate episode of the guide (E1 + E2)
        assert fake.requested == ["de", "de"]

    def test_default_source_still_uses_opensubtitles(self, tmp_path: Path) -> None:
        class FakeOSDT:
            SOURCE_NAME = "OpenSubtitles"

            def __init__(self) -> None:
                self.requested: list[str | None] = []

            def reference_transcript(self, *, tmdb_id=None, imdb_id=None, season=None, episode=None, languages=None):  # noqa: ANN001, ANN202
                self.requested.append(languages)
                return None

        fake = FakeOSDT()
        config = AppConfig(bin_dir=tmp_path / "bin", data_dir=tmp_path / "data")
        self._run(tmp_path, config, opensubtitles=fake)
        assert fake.requested == ["de", "de"]

    def test_subdl_source_without_key_reports_missing(self, tmp_path: Path) -> None:
        from episodematch.matcher.pipeline import Pipeline, ScanOptions
        from tests.test_pipeline import FakeFfprobe, FakeTmdb, make_guide

        config = AppConfig(bin_dir=tmp_path / "bin", data_dir=tmp_path / "data", subtitle_source="subdl")
        mkv_payload = {"format": {"duration": "2500"}, "streams": []}
        mkv = __import__(
            "episodematch.extractors.ffprobe_reader", fromlist=["parse_ffprobe_json"]
        ).parse_ffprobe_json(mkv_payload, path=tmp_path / "title_t00.mkv")

        class FakeExtractor:
            def extract_all(self, mkv, **kwargs):  # noqa: ANN001, ANN202
                from episodematch.core.models import SubtitleLine, Transcript

                return [Transcript(source="srt", language="ger", lines=[SubtitleLine(start_seconds=0, end_seconds=1, text="x" * 30)])]

        folder = tmp_path / "rips"
        folder.mkdir()
        (folder / "title_t00.mkv").write_bytes(b"\x1a\x45\xdf\xa3fake")
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            tmdb=FakeTmdb(make_guide()),
            subtitle_extractor=FakeExtractor(),
        )
        report = pipeline.scan_folder(folder, series="Test Show", season=1, options=ScanOptions(depth="fast", use_disc_db=False))
        subtitle_reports = [r for r in report.pillar_reports if r.pillar == "subtitle"]
        assert subtitle_reports and subtitle_reports[0].status == "ok"  # extraction ran
        # No client available -> references simply absent, no crash.


class TestConfigSubtitleSource:
    def test_env_override_source_and_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EPISODEMATCH_SUBTITLE_SOURCE", "SubDL")
        monkeypatch.setenv("EPISODEMATCH_SUBDL_API_KEY", "subdl123")
        config = apply_env_overrides(AppConfig())
        assert config.subtitle_source == "subdl"
        assert config.subdl_api_key == "subdl123"

    def test_invalid_source_rejected(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AppConfig(subtitle_source="kotlin")

class TestMalformedJsonResponse:
    """Audit B6: a 200 response with a non-JSON body (proxy error page,
    HTML rate-limit wall) must surface as ProviderError, not a raw
    json.JSONDecodeError that escapes every error handler."""

    def test_get_with_html_body_raises_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>Service Temporarily Unavailable</html>")

        client = _make_client(handler)
        with pytest.raises(ProviderError, match="not JSON"):
            client._get("https://api3.subdl.com/api/v1/subtitles", {"film_name": "x"})

class TestSubdlCandidateValidation:
    """Audit E1: SubDL's search filter is loose — results for other episodes
    and languages can come back. Wrong candidates must be rejected, and the
    transcript language must come from the actual match, not the request."""

    def _handler_with(self, candidates: list[dict[str, Any]], served: dict[str, bytes]) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                return httpx.Response(200, json={"status": True, "subtitles": candidates})
            for suffix, content in served.items():
                if url == f"https://dl.subdl.com{suffix}":
                    return httpx.Response(200, content=content)
            return httpx.Response(404)

        return handler

    def test_wrong_episode_candidate_is_rejected(self) -> None:
        """An E2 release returned for an E1 request is never downloaded."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                return httpx.Response(
                    200,
                    json={"status": True, "subtitles": [_subtitle_entry("s2", "/subtitle/e2.zip", episode=2)]},
                )
            downloaded.append(url)
            if url == "https://dl.subdl.com/subtitle/e2.zip":
                return httpx.Response(200, content=_zip_bytes(("sub.srt", _SRT_BYTES)))
            return httpx.Response(404)

        downloaded: list[str] = []
        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is None  # the only candidate is the wrong episode
        assert not downloaded  # never even fetched the wrong episode

    def test_wrong_language_candidate_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                entry = _subtitle_entry("s1", "/subtitle/fr.zip")
                entry["lang"] = "FR"
                return httpx.Response(200, json={"status": True, "subtitles": [entry]})
            if url == "https://dl.subdl.com/subtitle/fr.zip":
                return httpx.Response(200, content=_zip_bytes(("sub.srt", _SRT_BYTES)))
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is None  # fr content must not be served for a de request

    def test_language_labeled_from_actual_candidate(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                entry = _subtitle_entry("s1", "/subtitle/de.zip")
                entry["lang"] = "DE"
                return httpx.Response(200, json={"status": True, "subtitles": [entry]})
            if url == "https://dl.subdl.com/subtitle/de.zip":
                return httpx.Response(200, content=_zip_bytes(("sub.srt", _SRT_BYTES)))
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de,en")
        assert transcript is not None
        assert transcript.language == "de"  # from the candidate, not just the request head

    def test_correct_episode_preferred_and_mismatched_url_untouched(self) -> None:
        candidates = [
            _subtitle_entry("wrong", "/subtitle/e2.zip", episode=2),
            _subtitle_entry("right", "/subtitle/e1.zip", episode=1),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                return httpx.Response(200, json={"status": True, "subtitles": candidates})
            downloaded.append(url)
            if url == "https://dl.subdl.com/subtitle/e1.zip":
                return httpx.Response(200, content=_zip_bytes(("sub.srt", _SRT_BYTES)))
            return httpx.Response(404)

        downloaded: list[str] = []
        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is not None
        assert downloaded == ["https://dl.subdl.com/subtitle/e1.zip"]

    def test_candidates_without_fields_are_tolerated(self) -> None:
        """Missing episode/lang fields must not reject a candidate (the fields
        are optional in SubDL responses)."""
        entry = {"release_name": "opaque", "name": "n", "url": "/subtitle/x.zip"}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                return httpx.Response(200, json={"status": True, "subtitles": [entry]})
            if url == "https://dl.subdl.com/subtitle/x.zip":
                return httpx.Response(200, content=_zip_bytes(("sub.srt", _SRT_BYTES)))
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is not None
        assert transcript.language == "de"  # fallback: requested language

class TestSubdlRealResponseShapes:
    """Regression for the v0.11.0 subtitle regression: real SubDL responses
    carry lang="English" (full name, NOT an ISO code), language="EN" (ISO) and
    full-season packages with episode=None / episode_end=0. The v0.11.0
    candidate_ok filter rejected 10/10 real candidates for S01E01 of
    "The Americans" (all because of the lang full-name comparison) and killed
    pillar 2 entirely."""

    def _real_shape_entry(self, **overrides: Any) -> dict[str, Any]:
        entry = {
            "release_name": "The Americans (2013), SEASON 1, 1080p x265 [23.976 FPS]",
            "name": "The Americans (2013), SEASON 1, 1080p x265 [23.976 FPS].zip",
            "lang": "English",
            "author": "uploader",
            "url": "/subtitle/3658757-8575372.zip",
            "season": 1,
            "episode": None,
            "language": "EN",
            "framerate": 2,
            "fps": "23.976",
            "hi": False,
            "episode_from": None,
            "episode_end": 0,
            "full_season": True,
        }
        entry.update(overrides)
        return entry

    def _handler(self, candidates: list[dict[str, Any]], downloaded: list[str]) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://api3.subdl.com/api/v1/subtitles"):
                return httpx.Response(200, json={"status": True, "subtitles": candidates})
            for candidate in candidates:
                if candidate.get("url") and url == f"https://dl.subdl.com{candidate['url']}":
                    downloaded.append(url)
                    return httpx.Response(200, content=_zip_bytes(("sub.srt", _SRT_BYTES)))
            return httpx.Response(404)

        return handler

    def test_full_language_name_is_accepted(self) -> None:
        """lang='English' (full name) must NOT be rejected against a 'en'
        request — validation uses the ISO 'language' field instead."""
        downloaded: list[str] = []
        client = _make_client(self._handler([self._real_shape_entry()], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is not None and transcript.lines

    def test_full_season_package_with_episode_none_is_accepted(self) -> None:
        downloaded: list[str] = []
        entry = self._real_shape_entry(episode=None, full_season=True, episode_end=0)
        client = _make_client(self._handler([entry], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is not None

    def test_episode_string_with_leading_zero_is_accepted(self) -> None:
        downloaded: list[str] = []
        entry = self._real_shape_entry(episode="01", full_season=False, episode_from="01", episode_end="01")
        client = _make_client(self._handler([entry], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is not None

    def test_episode_range_containment_is_accepted_and_ranked_first(self) -> None:
        """A pack covering episodes 4-6 matches a request for episode 5 and
        outranks a full-season pack."""
        downloaded: list[str] = []
        candidates = [
            self._real_shape_entry(release_name="season-pack", full_season=True),
            self._real_shape_entry(
                release_name="episodes-4-6-pack",
                episode=None,
                full_season=False,
                episode_from=4,
                episode_end=6,
                url="/subtitle/range.zip",
            ),
        ]
        client = _make_client(self._handler(candidates, downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=5, languages="en")
        assert transcript is not None
        assert downloaded == ["https://dl.subdl.com/subtitle/range.zip"]  # range pack wins

    def test_zero_episode_counts_as_unknown_not_wrong(self) -> None:
        downloaded: list[str] = []
        entry = self._real_shape_entry(episode=0, full_season=False, episode_from=None, episode_end=0)
        client = _make_client(self._handler([entry], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is not None

    def test_explicit_wrong_single_episode_still_rejected(self) -> None:
        """E1's original intent survives: a package for exactly episode 2 must
        not be served for a request for episode 1."""
        downloaded: list[str] = []
        entry = self._real_shape_entry(episode=2, full_season=False, episode_from=2, episode_end=2)
        client = _make_client(self._handler([entry], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is None

    def test_wrong_iso_language_field_still_rejected(self) -> None:
        downloaded: list[str] = []
        entry = self._real_shape_entry(language="FR")
        client = _make_client(self._handler([entry], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is None

    def test_transcript_language_labeled_from_iso_language_field(self) -> None:
        """The label must be the ISO code ('en'), not the full name
        ('english') and not silently the requested language."""
        downloaded: list[str] = []
        client = _make_client(self._handler([self._real_shape_entry()], downloaded))
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="en")
        assert transcript is not None
        assert transcript.language == "en"

class TestSubdlThrottleRetry:
    """SubDL answers transient 429 "service_busy" with retryAfterSeconds while
    a scan hammers the search endpoint for 13 episodes back to back. The client
    must retry briefly instead of failing the episode's reference entirely
    (observed live: E03/E04 of The Americans disc 1 lost their references)."""

    def test_429_with_retry_after_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(
            "episodematch.providers.subdl.time.sleep",
            lambda seconds: sleeps.append(seconds),
        )
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    429,
                    json={"status": False, "statusCode": 429, "error": "service_busy", "retryAfterSeconds": 5},
                )
            return httpx.Response(200, json={"status": True, "subtitles": []})

        client = _make_client(handler)
        payload = client._get("https://api3.subdl.com/api/v1/subtitles", {"film_name": "x"})
        assert payload.get("status") is True
        assert calls["n"] == 2
        assert sleeps == [5.0]  # retryAfterSeconds honored

    def test_429_persists_raises_ratelimit_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("episodematch.providers.subdl.time.sleep", lambda seconds: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429, json={"error": "service_busy", "retryAfterSeconds": 5})

        client = _make_client(handler)
        with pytest.raises(RateLimitError):
            client._get("https://api3.subdl.com/api/v1/subtitles", {"film_name": "x"})
        assert calls["n"] == 3  # initial + 2 retries
