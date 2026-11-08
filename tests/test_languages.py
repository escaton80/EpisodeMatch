"""Tests for language handling: code mapping, OSDT verification, scorer guard, pipeline chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from episodematch.core.languages import (
    language_candidates,
    languages_match,
    to_osdt_language,
)
from episodematch.core.models import (
    EpisodeInfo,
    SubtitleLine,
    Transcript,
)
from episodematch.matcher.text_scorer import score_episode
from episodematch.providers.opensubtitles import OpenSubtitlesClient


# ---------------------------------------------------------------------------
# Code mapping
# ---------------------------------------------------------------------------


class TestToOsdtLanguage:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("ger", "de"),
            ("deu", "de"),
            ("eng", "en"),
            ("fre", "fr"),
            ("fra", "fr"),
            ("chi", "zh"),
            ("zho", "zh"),
            ("DE", "de"),
            ("en", "en"),
            ("de", "de"),
            (" xyz ", "xyz"),  # unknown -> lowercased passthrough
            ("", ""),
            (None, ""),
        ],
    )
    def test_mapping(self, code: str | None, expected: str) -> None:
        assert to_osdt_language(code) == expected


class TestLanguageCandidates:
    def test_primary_first_then_preferences_then_fallback(self) -> None:
        # preferred_languages defaults on German discs: ger, deu, eng — ger/deu
        # collapse to the same "de".
        assert language_candidates("ger", ["ger", "deu", "eng"]) == ["de", "en"]

    def test_unknown_primary_falls_through(self) -> None:
        assert language_candidates("xyz", ["ger", "eng"]) == ["xyz", "de", "en"]

    def test_no_primary(self) -> None:
        assert language_candidates(None, ["eng"]) == ["en"]
        assert language_candidates(None, None) == ["en"]

    def test_limit(self) -> None:
        chain = language_candidates("ger", ["fre", "spa", "ita", "jpn"], limit=3)
        assert chain == ["de", "fr", "es"]


class TestLanguagesMatch:
    def test_same_language_variants_match(self) -> None:
        assert languages_match("ger", "de")
        assert languages_match("deu", "ger")

    def test_different_languages_do_not_match(self) -> None:
        assert not languages_match("ger", "eng")

    def test_unknown_never_mismatches(self) -> None:
        assert languages_match(None, "en")
        assert languages_match("ger", None)
        assert languages_match("", "")


# ---------------------------------------------------------------------------
# OpenSubtitles client: requested language + per-entry verification
# ---------------------------------------------------------------------------


_SRT_BYTES = (
    "1\n00:00:01,000 --> 00:00:02,500\nHello reference line\n\n"
    "2\n00:00:05,000 --> 00:00:06,000\nSecond reference line\n"
).encode("utf-8")


def _entry(subtitle_id: str, language: str, file_id: int, downloads: int = 10, *, files_at_top_level: bool = False) -> dict[str, Any]:
    """Search-result entry in the REAL OSDT API shape: `files` lives inside
    `attributes` (a top-level variant is tolerated by the client for robustness)."""
    attributes: dict[str, Any] = {"language": language, "download_count": downloads}
    if files_at_top_level:
        return {"id": subtitle_id, "attributes": attributes, "files": [{"file_id": file_id}]}
    attributes["files"] = [{"file_id": file_id}]
    return {"id": subtitle_id, "attributes": attributes}


def _make_client(handler) -> OpenSubtitlesClient:
    transport = httpx.MockTransport(handler)
    return OpenSubtitlesClient("test-key", client=httpx.Client(transport=transport))


class TestOpenSubtitlesLanguageVerification:
    def _client(self, handler) -> OpenSubtitlesClient:
        return _make_client(handler)

    def test_requests_track_language_and_verifies_entries(self) -> None:
        """The requested language reaches the API, and an entry whose own
        language attribute disagrees is skipped even when ranked first."""
        requests: list[str] = []
        fr_entry = _entry("osdt-fr", "fr", 111, downloads=999)  # highest download count
        de_entry = _entry("osdt-de", "de", 222, downloads=5)

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            requests.append(str(request.url))
            if path == "/api/v1/subtitles":
                assert "languages=de" in str(request.url)
                return httpx.Response(200, json={"data": [fr_entry, de_entry]})
            if path == "/api/v1/download":
                return httpx.Response(200, json={"link": "http://test/ref.srt", "release": "r"})
            if path == "/ref.srt":
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=2, languages="de")
        assert transcript is not None
        assert transcript.language == "de"
        assert [line.text for line in transcript.lines] == [
            "Hello reference line",
            "Second reference line",
        ]
        assert len([url for url in requests if "/download" in url]) == 1  # fr was skipped

    def test_all_entries_wrong_language_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                return httpx.Response(200, json={"data": [_entry("osdt-fr", "fr", 111)]})
            return httpx.Response(500)  # must never be reached

        client = _make_client(handler)
        assert client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de") is None

    def test_entry_without_language_attribute_is_accepted(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                entry = _entry("osdt-x", "", 333)
                entry["attributes"].pop("language")
                return httpx.Response(200, json={"data": [entry]})
            if path == "/api/v1/download":
                return httpx.Response(200, json={"link": "http://test/ref.srt"})
            if path == "/ref.srt":
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is not None  # unverifiable language is tolerated
        assert transcript.language == "de"  # falls back to the requested language

    def test_default_language_is_english(self) -> None:
        requested_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, json={"data": []})

        client = _make_client(handler)
        assert client.reference_transcript(tmdb_id=1, season=1, episode=1) is None
        assert "languages=en" in requested_urls[0]

    def test_files_under_attributes_are_downloaded(self) -> None:
        """Regression (2×): the real API nests `files` inside `attributes` AND the
        /download endpoint requires POST (GET returns the website's HTML 404) —
        both bugs silently killed every reference download in live use."""
        fr_entry = _entry("osdt-fr", "fr", 111, downloads=999)  # real shape, wrong language
        de_entry = _entry("osdt-de", "de", 222, downloads=5)  # real shape, right language
        methods: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            methods.append(request.method)
            if path == "/api/v1/subtitles":
                assert request.method == "GET"
                return httpx.Response(200, json={"data": [fr_entry, de_entry]})
            if path == "/api/v1/download":
                assert request.method == "POST", "OSDT /download requires POST"
                assert "file_id=222" in str(request.url)
                return httpx.Response(200, json={"link": "http://test/ref.srt"})
            if path == "/ref.srt":
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=2, languages="de")
        assert transcript is not None and transcript.language == "de"
        assert len(transcript.lines) == 2
        assert methods == ["GET", "POST", "GET"]

    def test_files_at_top_level_are_tolerated(self) -> None:
        legacy_entry = _entry("osdt-de", "de", 222, files_at_top_level=True)

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                return httpx.Response(200, json={"data": [legacy_entry]})
            if path == "/api/v1/download":
                return httpx.Response(200, json={"link": "http://test/ref.srt"})
            if path == "/ref.srt":
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is not None and len(transcript.lines) == 2


class TestOpenSubtitlesThrottling:
    """429 with a short reset window is per-second throttling, not the daily
    quota — the client must wait and retry instead of giving up permanently."""

    def test_429_with_short_window_is_retried_and_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time as time_module

        from episodematch.core.exceptions import RateLimitError  # noqa: F401

        sleeps: list[float] = []
        monkeypatch.setattr(
            "episodematch.providers.opensubtitles.time.sleep",
            lambda seconds: sleeps.append(seconds),
        )

        download_calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                return httpx.Response(200, json={"data": [_entry("osdt-de", "de", 222)]})
            if path == "/api/v1/download":
                download_calls.append(1)
                if len(download_calls) == 1:
                    # First attempt throttled with a 1-second window.
                    return httpx.Response(429, headers={"x-ratelimit-reset": "1"})
                return httpx.Response(200, json={"link": "http://test/ref.srt"})
            if path == "/ref.srt":
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is not None and len(transcript.lines) == 2
        assert download_calls == [1, 1]  # retried after the throttle
        assert sleeps and sleeps[0] >= 1.0

    def test_429_twice_degrades_to_none_with_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After exhausting the built-in throttle retries the episode degrades to
        plot-only matching (None + WARNING), it does not crash the scan."""
        monkeypatch.setattr("episodematch.providers.opensubtitles.time.sleep", lambda seconds: None)

        download_calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                return httpx.Response(200, json={"data": [_entry("osdt-de", "de", 222)]})
            if path == "/api/v1/download":
                download_calls.append(1)
                return httpx.Response(429, headers={"x-ratelimit-reset": "1"})
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is None
        assert download_calls == [1, 1, 1]  # initial + 2 throttle retries
        # After exhausting the throttle retries the loop breaks silently —
        # the scan degrades to plot-only matching for this episode.

    def test_long_window_is_daily_quota_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from episodematch.core.exceptions import RateLimitError  # noqa: F401

        monkeypatch.setattr(
            "episodematch.providers.opensubtitles.time.sleep",
            lambda seconds: pytest.fail("must not sleep on a long daily-quota window"),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                return httpx.Response(200, json={"data": [_entry("osdt-de", "de", 222)]})
            if path == "/api/v1/download":
                return httpx.Response(429, headers={"x-ratelimit-reset": "86400"})
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is None  # degrades, breaks the candidate loop, no sleep

    def test_zero_line_srt_skips_candidate_and_tries_next(self) -> None:
        """A downloaded but unparsable reference must be skipped loudly and the
        next candidate gets a chance instead of failing the episode silently."""
        broken = _entry("osdt-broken", "de", 111, downloads=999)  # ranked first
        good = _entry("osdt-good", "de", 222, downloads=5)
        downloads_attempted: list[str] = []
        state = {"file_id": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/api/v1/subtitles":
                return httpx.Response(200, json={"data": [broken, good]})
            if path == "/api/v1/download":
                downloads_attempted.append(str(request.url))
                state["file_id"] += 1
                return httpx.Response(200, json={"link": f"http://test/ref{state['file_id']}.srt"})
            if path.endswith("ref1.srt"):
                return httpx.Response(200, content=b"<html>not a subtitle</html>")  # 0 lines
            if path.endswith("ref2.srt"):
                return httpx.Response(200, content=_SRT_BYTES)
            return httpx.Response(404)

        client = _make_client(handler)
        transcript = client.reference_transcript(tmdb_id=1, season=1, episode=1, languages="de")
        assert transcript is not None and len(transcript.lines) == 2
        assert len(downloads_attempted) == 2  # broken candidate, then the good one


# ---------------------------------------------------------------------------
# Scorer guard: dialog scoring is skipped on language mismatch
# ---------------------------------------------------------------------------


def _transcript_with_language(language: str | None, lines: list[str]) -> Transcript:
    return Transcript(
        source="srt",
        language=language,
        lines=[
            SubtitleLine(start_seconds=float(i) * 2.0, end_seconds=float(i) * 2.0 + 1.5, text=text)
            for i, text in enumerate(lines)
        ],
    )


_DIALOG = ["we need to talk about the money you owe me", "i will get your money by friday"]


class TestScorerLanguageGuard:
    def _episode(self) -> EpisodeInfo:
        return EpisodeInfo(season=1, episode=3, title="Money", plot="The money problem continues")

    def test_mismatched_language_skips_dialog(self) -> None:
        extracted = _transcript_with_language("ger", [line.capitalize() + "!" for line in _DIALOG])
        reference = _transcript_with_language("en", _DIALOG)  # identical text otherwise
        score = score_episode(extracted, self._episode(), reference_transcript=reference)
        assert score.language_mismatch == "de vs en"
        assert score.dialog_score == 0.0
        assert score.matched_pairs == []
        assert score.final_score == pytest.approx(0.35 * score.plot_score)
        assert "language mismatch (de vs en)" in score.summary

    def test_matching_language_scores_dialog_normally(self) -> None:
        extracted = _transcript_with_language("ger", [line.capitalize() + "!" for line in _DIALOG])
        reference = _transcript_with_language("de", _DIALOG)
        score = score_episode(extracted, self._episode(), reference_transcript=reference)
        assert score.language_mismatch is None
        assert score.dialog_score > 0.85

    def test_unknown_languages_never_trigger_guard(self) -> None:
        extracted = _transcript_with_language(None, [line.capitalize() + "!" for line in _DIALOG])
        reference = _transcript_with_language(None, _DIALOG)
        score = score_episode(extracted, self._episode(), reference_transcript=reference)
        assert score.language_mismatch is None
        assert score.dialog_score > 0.85

    def test_no_reference_still_scores_plot_only(self) -> None:
        extracted = _transcript_with_language("ger", [line.capitalize() + "!" for line in _DIALOG])
        score = score_episode(extracted, self._episode(), reference_transcript=None)
        assert score.language_mismatch is None
        assert score.dialog_score == 0.0


# ---------------------------------------------------------------------------
# Pipeline: reference chain uses the extracted track's language
# ---------------------------------------------------------------------------


class FakeOpenSubtitles:
    def __init__(self, transcript: Transcript | None) -> None:
        self.requested: list[str | None] = []
        self._transcript = transcript

    def reference_transcript(self, *, tmdb_id=None, imdb_id=None, season=None, episode=None, languages=None):  # noqa: ANN001, ANN202
        self.requested.append(languages)
        return self._transcript


class FakeOpenSubtitlesByLanguage:
    """Returns a transcript only for specific languages (records every request)."""

    def __init__(self, by_language: dict[str, Transcript | None]) -> None:
        self.requested: list[str | None] = []
        self._by_language = by_language

    def reference_transcript(self, *, tmdb_id=None, imdb_id=None, season=None, episode=None, languages=None):  # noqa: ANN001, ANN202
        self.requested.append(languages)
        return self._by_language.get(languages or "")


class TestPipelineLanguageChain:
    @staticmethod
    def _run(tmp_path: Path, fake_osdt, *, transcript_language: str | None = "ger"):
        from episodematch.core.config import AppConfig
        from episodematch.matcher.pipeline import Pipeline, ScanOptions
        from tests.test_pipeline import FakeFfprobe, FakeTmdb, make_guide

        mkv_payload = {
            "format": {"duration": "2500", "size": "1000"},
            "streams": [
                {"index": 0, "codec_name": "h264", "codec_type": "video"},
                {
                    "index": 1,
                    "codec_name": "subrip",
                    "codec_type": "subtitle",
                    "tags": {"language": transcript_language} if transcript_language else {},
                },
            ],
        }
        mkv = __import__(
            "episodematch.extractors.ffprobe_reader", fromlist=["parse_ffprobe_json"]
        ).parse_ffprobe_json(mkv_payload, path=tmp_path / "title_t00.mkv")

        class FakeExtractor:
            def extract_all(self, mkv, *, duration_seconds=None, window_minutes=None, languages_first=None, ocr_max_events=40):  # noqa: ANN001, ANN202
                return [
                    Transcript(
                        source="srt",
                        language=transcript_language,
                        lines=_transcript_with_language(transcript_language, _DIALOG).lines,
                    )
                ]

        folder = tmp_path / "rips"
        folder.mkdir()
        (folder / "title_t00.mkv").write_bytes(b"\x1a\x45\xdf\xa3fake")
        config = AppConfig(bin_dir=tmp_path / "bin", data_dir=tmp_path / "data")
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=None,
            tmdb=FakeTmdb(make_guide()),
            opensubtitles=fake_osdt,
            subtitle_extractor=FakeExtractor(),
        )
        return pipeline.scan_folder(
            folder, series="Test Show", season=1, options=ScanOptions(depth="fast", use_disc_db=True)
        )

    def test_reference_requested_in_track_language(self, tmp_path: Path) -> None:
        fake_osdt = FakeOpenSubtitles(
            _transcript_with_language("de", _DIALOG)
        )
        self._run(tmp_path, fake_osdt)
        # One request per candidate episode of the guide (E1 + E2), always German.
        assert fake_osdt.requested == ["de", "de"]

    def test_known_language_never_falls_back_to_english(self, tmp_path: Path) -> None:
        """German track, no German reference on OSDT: an English download would be
        discarded by the scorer's language guard anyway — it must not be attempted
        (wasted quota + wrong-language dialog)."""
        fake_osdt = FakeOpenSubtitlesByLanguage({"de": None, "en": _transcript_with_language("en", _DIALOG)})
        self._run(tmp_path, fake_osdt, transcript_language="ger")
        assert fake_osdt.requested == ["de", "de"]  # no "en" attempt at all

    def test_unknown_track_language_uses_preferred_chain(self, tmp_path: Path) -> None:
        fake_osdt = FakeOpenSubtitlesByLanguage({"en": _transcript_with_language("en", _DIALOG)})
        self._run(tmp_path, fake_osdt, transcript_language=None)
        # Per episode the chain is de -> en; both episodes scored.
        assert fake_osdt.requested == ["de", "en", "de", "en"]

    def test_unknown_track_language_stops_at_first_hit(self, tmp_path: Path) -> None:
        fake_osdt = FakeOpenSubtitlesByLanguage({"de": _transcript_with_language("de", _DIALOG)})
        self._run(tmp_path, fake_osdt, transcript_language=None)
        assert fake_osdt.requested == ["de", "de"]  # stops at the first hit per episode
