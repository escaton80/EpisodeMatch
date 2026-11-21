"""End-to-end pipeline integration tests with fully injected fake services.

These tests run the whole 5-pillar pipeline offline: probed metadata comes from a
fake ffprobe reader, disc data from a local TheDiscDb fixture clone, the season
guide from a fake TMDB client, subtitles from a fake extractor, and OCR from a
deterministic fake backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from episodematch.core.config import AppConfig
from episodematch.core.models import (
    EpisodeInfo,
    MkvFile,
    SeasonGuide,
    SeriesInfo,
    StreamInfo,
    StreamType,
    SubtitleFormat,
    SubtitleLine,
    Transcript,
    ProgressEvent,
)
from episodematch.extractors.ffprobe_reader import FfprobeReader
from episodematch.extractors.subtitle_extractor import SubtitleExtractor
from episodematch.core.exceptions import LlmError
from episodematch.matcher.llm_arbiter import LlmArbiter, LlmSettings
from episodematch.matcher.pipeline import Pipeline, ScanOptions
from episodematch.providers.thediscdb import LocalDiscDbImporter
from episodematch.providers.tmdb_client import TmdbClient


class FakeFfprobe(FfprobeReader):
    """Returns prepared MkvFile objects instead of running ffprobe."""

    def __init__(self, probed: dict[str, MkvFile]) -> None:
        self._probed = probed

    def probe(self, path, *, with_ebml: bool = True):  # noqa: ANN001, ANN202
        return self._probed[Path(path).name]


class FakeTmdb(TmdbClient):
    """Season guide without any HTTP."""

    def __init__(self, guide: SeasonGuide) -> None:
        self._guide = guide

    def find_series(self, query: str, *, year: int | None = None):  # noqa: ANN001, ANN202, D102
        return self._guide.series

    def season_guide(self, series, season):  # noqa: ANN001, ANN202
        return self._guide


class FakeSubtitleExtractor(SubtitleExtractor):
    """Returns per-file transcripts without touching the filesystem."""

    def __init__(self, transcripts: dict[str, Transcript]) -> None:
        self._transcripts = transcripts

    def extract_all(self, mkv, *, duration_seconds=None, window_minutes=None, languages_first=None, ocr_max_events=40):  # noqa: ANN001, ANN202
        transcript = self._transcripts.get(mkv.file_name)
        return [transcript] if transcript else []


def make_guide() -> SeasonGuide:
    series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id="tt1234567", first_air_year=2020)
    episodes = [
        EpisodeInfo(
            season=1,
            episode=1,
            title="Pilot",
            plot="A chemistry teacher decides to cook and the desert aftermath begins",
            directors=["Michelle MacLaren"],
            writers=["Vince Gilligan"],
            runtime_seconds=2550,
        ),
        EpisodeInfo(
            season=1,
            episode=2,
            title="Cat's in the Bag...",
            plot="Jesse deals with the bag problem while Walt recovers at home",
            directors=["Adam Bernstein"],
            writers=["Vince Gilligan"],
            runtime_seconds=2515,
        ),
    ]
    return SeasonGuide(series=series, season=1, episodes=episodes)


def make_mkv(name: str, *, mpls: str, duration: float, size: int) -> MkvFile:
    return MkvFile(
        path=Path("D:/rips") / name,
        size_bytes=size,
        duration_seconds=duration,
        container_name="matroska",
        title_tag=mpls,
        source_id=mpls,
        streams=[StreamInfo(index=0, stream_type=StreamType.VIDEO, codec_name="h264")],
    )


@pytest.fixture()
def rip_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "rips"
    folder.mkdir()
    (folder / "title_t00.mkv").write_bytes(b"\x1a\x45\xdf\xa3fake1")
    (folder / "title_t01.mkv").write_bytes(b"\x1a\x45\xdf\xa3fake2")
    return folder


def build_pipeline(config: AppConfig, **overrides: Any) -> Pipeline:
    services = {
        "ffprobe": overrides.get("ffprobe"),
        "disc_db_client": overrides.get("disc_db_client"),
        "tmdb": overrides.get("tmdb"),
        "imdb": overrides.get("imdb", False) or None,
        "opensubtitles": overrides.get("opensubtitles"),
        "ocr": overrides.get("ocr"),
        "subtitle_extractor": overrides.get("subtitle_extractor"),
        "audio_snipper": overrides.get("audio_snipper"),
        "frame_scanner": overrides.get("frame_scanner"),
        "llm_arbiter": overrides.get("llm_arbiter"),
    }
    return Pipeline(
        config,
        database=overrides.get("database"),
        progress=overrides.get("progress"),
        **services,
    )


class TestPipelineFastPath:
    def test_discdb_only_fast_scan_assigns_everything(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        mkv2 = make_mkv("title_t01.mkv", mpls="00801.mpls", duration=41 * 60 + 55, size=2_011_000_000)
        probed = {mkv1.file_name: mkv1, mkv2.file_name: mkv2}

        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe(probed),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
        )
        report = pipeline.scan_folder(rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast"))

        assert report.assignment.assigned_count == 2
        by_file = {entry.file_path.name: entry for entry in report.assignment.entries}
        assert by_file["title_t00.mkv"].episode == 1
        assert by_file["title_t00.mkv"].episode_title == "Pilot"
        assert by_file["title_t00.mkv"].confidence >= 0.6
        assert by_file["title_t01.mkv"].episode == 2
        discdb_reports = [r for r in report.pillar_reports if r.pillar == "disc_db"]
        assert all(r.status == "ok" for r in discdb_reports)

    def test_probe_failures_are_reported_not_fatal(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        broken_name = "title_t01.mkv"

        class PartialFfprobe(FakeFfprobe):
            def probe(self, path, *, with_ebml: bool = True):  # noqa: ANN001, ANN202
                if Path(path).name == broken_name:
                    raise RuntimeError("simulated probe failure")
                return self._probed[Path(path).name]

        pipeline = build_pipeline(
            config,
            ffprobe=PartialFfprobe({"title_t00.mkv": mkv1}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
        )
        report = pipeline.scan_folder(rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast"))
        failed = [r for r in report.pillar_reports if r.status == "failed" and r.file_name == broken_name]
        assert failed
        assert report.assignment.assigned_count == 1

    def test_progress_events_emitted(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        events: list[ProgressEvent] = []
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv1}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
            progress=events.append,
        )
        pipeline.scan_folder(rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast"))
        stages = [event.stage.value for event in events]
        assert "probe" in stages and "solve" in stages

    def test_cancelled_scan_raises(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv1}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
            cancel=lambda: True,
        )
        from episodematch.core.exceptions import EpisodeMatchError

        with pytest.raises(EpisodeMatchError, match="[Cc]ancel"):
            pipeline.scan_folder(rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast"))


class TestPipelineDeepPath:
    def test_subtitle_pillar_scores_reference_dialog(
        self,
        config: AppConfig,
        rip_folder: Path,
    ) -> None:
        # No TheDiscDb match possible: files have unknown MPLS names.
        mkv1 = make_mkv("title_t00.mkv", mpls="09999.mpls", duration=2500.0, size=1)
        guide = make_guide()
        reference_lines = ["where is the money jesse", "we need to talk about the lab"]
        transcripts = {
            "title_t00.mkv": Transcript(
                source="srt",
                lines=[SubtitleLine(start_seconds=100.0 + i, end_seconds=101.0 + i, text=line.title()) for i, line in enumerate(reference_lines)],
            )
        }
        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv1}),
            disc_db_client=None,
            tmdb=FakeTmdb(guide),
            subtitle_extractor=FakeSubtitleExtractor(transcripts),
        )
        report = pipeline.scan_folder(rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast", use_disc_db=False))
        subtitle_reports = [r for r in report.pillar_reports if r.pillar == "subtitle"]
        assert subtitle_reports and subtitle_reports[0].status == "ok"
        # Episode 2 plot mentions "Jesse"/"bag" - but the dialog exactly matches neither;
        # the key assertion is that the pillar ran and produced scored evidence.
        candidate = report.candidates[0]
        subtitle_evidence = [e for e in candidate.evidence if e.pillar.value == "subtitle"]
        assert subtitle_evidence
        assert report.assignment.assigned_count >= 0  # score may be below threshold; pillar ran

    def test_full_deep_scan_with_all_fakes(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        mkv2 = make_mkv("title_t01.mkv", mpls="00801.mpls", duration=41 * 60 + 55, size=2_011_000_000)

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                from episodematch.core.models import TitleCardHit, TitleCardKind

                if mkv_path.name == "title_t00.mkv":
                    return [
                        TitleCardHit(
                            timestamp_seconds=100.0,
                            kind=TitleCardKind.CREDITS,
                            raw_lines=["Directed by Michelle MacLaren"],
                            directors=["Michelle MacLaren"],
                        )
                    ]
                return []

        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv1, "title_t01.mkv": mkv2}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
            frame_scanner=FakeVisual(),
        )
        report = pipeline.scan_folder(
            rip_folder,
            series="Test Show",
            season=1,
            options=ScanOptions(depth="deep", use_subtitles=False, use_audio=False, use_llm=False),
        )
        by_file = {entry.file_path.name: entry for entry in report.assignment.entries}
        assert by_file["title_t00.mkv"].episode == 1
        visual_reports = [r for r in report.pillar_reports if r.pillar == "visual"]
        assert visual_reports and visual_reports[0].status == "ok"
        # The director evidence must be attached to the solved entry.
        entry = by_file["title_t00.mkv"]
        assert any(e.pillar.value == "visual" for e in entry.evidence)


class TestBestTranscriptSelection:
    """The scoring winner is chosen by language first — a long Spanish OCR track
    must never outrank a shorter German one (it would derail the OSDT chain)."""

    @staticmethod
    def _transcript(language: str | None, source: str, *, chars: int) -> Transcript:
        text = "x" * max(2, chars)
        return Transcript(
            source=source,
            language=language,
            lines=[SubtitleLine(start_seconds=0.0, end_seconds=1.0, text=text)],
        )

    def test_preferred_language_beats_longer_foreign(self) -> None:
        from episodematch.ui.views.workspace_view import WorkspaceView  # noqa: F401 - ensures view module imports

        spanish_long = self._transcript("spa", "pgs-ocr", chars=5000)
        german_short = self._transcript("ger", "pgs-ocr", chars=40)
        winner = Pipeline._best_transcript([spanish_long, german_short], ["ger", "deu", "eng"])
        assert winner is german_short

    def test_native_sources_outrank_python_ocr(self) -> None:
        german_srt = self._transcript("ger", "srt", chars=100)
        german_cli = self._transcript("ger", "pgs-se-cli", chars=200)
        german_ocr = self._transcript("ger", "pgs-ocr", chars=300)
        winner = Pipeline._best_transcript([german_ocr, german_cli, german_srt], ["ger"])
        assert winner.source in ("srt", "pgs-se-cli")

    def test_volume_breaks_remaining_ties(self) -> None:
        small = self._transcript("ger", "srt", chars=30)
        big = self._transcript("ger", "srt", chars=300)
        assert Pipeline._best_transcript([small, big], ["ger"]) is big

    def test_no_transcripts(self) -> None:
        assert Pipeline._best_transcript([], ["ger"]) is None


class TestLlmArbiterOffline:
    def test_verdict_parsing_and_merge(self, config: AppConfig) -> None:
        from episodematch.core.models import Evidence, MatchCandidate, Pillar
        from episodematch.matcher.llm_arbiter import merge_verdict_into_candidates

        verdict_payload = {
            "assignments": [
                {"file": "title_t00.mkv", "episode": 2, "confidence": 0.9, "reason": "dialog matches E2"},
            ],
            "notes": "clean mapping",
        }
        arbiter = LlmArbiter.__new__(LlmArbiter)
        arbiter._settings = LlmSettings(api_key="k")
        arbiter._client = None
        arbiter._owns_client = False
        arbiter._max_retries = 1
        arbiter._retry_backoff = 0.0
        verdict = arbiter._validate(verdict_payload, json.dumps(verdict_payload))
        assert verdict.assignments[0].episode == 2
        assert verdict.notes == "clean mapping"

        candidate = MatchCandidate(file_path=Path("title_t00.mkv"))
        candidate.add_evidence(Evidence(pillar=Pillar.SUBTITLE, kind="x", score=0.4, summary="s", episode_hint=2))
        merge_verdict_into_candidates(verdict, [candidate], boost=0.1)
        assert candidate.episode_scores[2] >= 0.9 + 0.1
        assert any(e.pillar.value == "llm" for e in candidate.evidence)

    def test_malformed_verdict_rejected(self, config: AppConfig) -> None:
        arbiter = LlmArbiter.__new__(LlmArbiter)
        arbiter._settings = LlmSettings(api_key="k")
        with pytest.raises(Exception, match="assignments"):
            arbiter._validate({"wrong": []}, "{}")

    def test_json_extraction_with_fences(self) -> None:
        from episodematch.matcher.llm_arbiter import extract_json_object

        text = 'Sure! Here you go:\n```json\n{"assignments": [{"file": "a", "episode": 1}]}\n```\nDone.'
        payload = extract_json_object(text)
        assert payload["assignments"][0]["file"] == "a"

    def test_missing_api_key_raises(self) -> None:
        from episodematch.core.exceptions import LlmError

        with pytest.raises(LlmError):
            LlmArbiter(LlmSettings(api_key=""))


class TestLlmConnectionTest:
    """The Settings view's "Test LLM connection" probe."""

    def _settings(self, base_url: str = "https://llm.test/v4") -> LlmSettings:
        return LlmSettings(api_key="sk-test", base_url=base_url, model="test-model")

    def test_success_reports_model_reply(self) -> None:
        import httpx

        from episodematch.matcher.llm_arbiter import test_connection

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/chat/completions" in str(request.url)
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ok, message = test_connection(self._settings(), client=client)
        assert ok is True
        assert "test-model" in message and "OK" in message

    def test_bad_key_reports_401(self) -> None:
        import httpx

        from episodematch.matcher.llm_arbiter import test_connection

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ok, message = test_connection(self._settings(), client=client)
        assert ok is False
        assert "401" in message

    def test_wrong_endpoint_reports_404_with_url(self) -> None:
        import httpx

        from episodematch.matcher.llm_arbiter import test_connection

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "no such model"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ok, message = test_connection(self._settings(), client=client)
        assert ok is False
        assert "404" in message and "llm_base_url" in message

    def test_network_error_reported(self) -> None:
        import httpx

        from episodematch.matcher.llm_arbiter import test_connection

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ok, message = test_connection(self._settings(), client=client)
        assert ok is False and "Connection failed" in message

    def test_empty_key_reported_without_request(self) -> None:
        from episodematch.matcher.llm_arbiter import test_connection

        ok, message = test_connection(LlmSettings(api_key=""))
        assert ok is False and "No API key" in message

class TestLlmMalformedJson:
    """Audit B6: a 200 response with a non-JSON body must raise LlmError so the
    arbiter's retry loop can catch it — a raw JSONDecodeError escapes retries."""

    def test_chat_with_html_body_raises_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway error page</html>")

        arbiter = LlmArbiter(
            LlmSettings(api_key="test-key"),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LlmError, match="not JSON"):
            arbiter._chat("some prompt")

class _FakeResolver:
    """A ToolResolver stub so service construction works without binaries."""

    def tool_path(self, name: str) -> Path:
        return Path("D:/fake/bin") / name

    def find(self, name: str) -> Any:
        from episodematch.core.models import ToolInfo, ToolStatus

        return ToolInfo(
            name=name,
            status=ToolStatus.MISSING,
            path=Path("D:/fake/bin") / name,
            version=None,
        )


class TestScanWindowConfigPlumbing:
    """Audit F4: the pipeline must pass ExtractionSettings windows into the
    AudioSnipper/FrameScanner it constructs."""

    def _config(self, tmp_path: Path) -> AppConfig:
        from episodematch.core.config import ExtractionSettings

        return AppConfig(
            bin_dir=tmp_path / "bin",
            data_dir=tmp_path / "data",
            extraction=ExtractionSettings(
                audio_window_1_start_min=0.5,
                audio_window_1_end_min=1.0,
                audio_window_2_minutes_before_end=1.5,
                audio_window_2_length_min=0.75,
                title_card_window_minutes=2.5,
            ),
        )

    def test_audio_service_receives_configured_windows(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(self._config(tmp_path))
        pipeline._resolver = _FakeResolver()
        snipper = pipeline._audio_service()
        assert snipper.window_1_start == pytest.approx(30.0)  # 0.5 min -> seconds
        assert snipper.window_1_end == pytest.approx(60.0)
        assert snipper.minutes_before_end == pytest.approx(90.0)
        assert snipper.window_2_length == pytest.approx(45.0)

    def test_visual_service_receives_configured_title_window(self, tmp_path: Path) -> None:
        from tests.conftest import FakeOcrBackend

        pipeline = build_pipeline(self._config(tmp_path))
        pipeline._ocr = FakeOcrBackend([])
        pipeline._resolver = _FakeResolver()
        scanner = pipeline._visual_service()
        assert scanner.title_card_minutes == pytest.approx(2.5)

class TestSeasonRangeReleaseMatching:
    """Audit F5: multi-season boxes ("seasons-6-10-blu-ray") must match any
    season inside the range, not only their first season."""

    class _FakeClient:
        def __init__(self, folders: list[str]) -> None:
            self._folders = folders

        def list_releases(self, series_folder: str) -> list[str]:
            return list(self._folders)

    def test_season_inside_range_matches(self) -> None:
        client = self._FakeClient(["season-6-blu-ray", "seasons-6-10-blu-ray", "season-12-blu-ray"])
        assert Pipeline._disc_release_folders(client, "south-park-1997", 8) == ["seasons-6-10-blu-ray"]

    def test_range_boundaries_match(self) -> None:
        client = self._FakeClient(["seasons-6-10-blu-ray"])
        assert Pipeline._disc_release_folders(client, "x", 6) == ["seasons-6-10-blu-ray"]
        assert Pipeline._disc_release_folders(client, "x", 10) == ["seasons-6-10-blu-ray"]

    def test_season_outside_range_does_not_match(self) -> None:
        client = self._FakeClient(["seasons-6-10-blu-ray", "season-12-blu-ray"])
        assert Pipeline._disc_release_folders(client, "x", 12) == ["season-12-blu-ray"]

    def test_no_match_falls_back_to_all_folders(self) -> None:
        client = self._FakeClient(["complete-series-blu-ray"])
        assert Pipeline._disc_release_folders(client, "x", 8) == ["complete-series-blu-ray"]

class TestProgressPerFile:
    """The GUI progress bar must keep moving through the deep stages — each
    file reports its stage-local percent and the file currently being worked
    on, instead of only the initial probe showing percentages."""

    def test_visual_stage_reports_percent_per_file(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        mkv2 = make_mkv("title_t01.mkv", mpls="00801.mpls", duration=41 * 60 + 55, size=2_011_000_000)
        events: list[ProgressEvent] = []

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                return []

        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv1, "title_t01.mkv": mkv2}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
            frame_scanner=FakeVisual(),
            progress=events.append,
        )
        pipeline.scan_folder(
            rip_folder,
            series="Test Show",
            season=1,
            options=ScanOptions(depth="deep", use_subtitles=False, use_audio=False, use_llm=False),
        )
        percent_events = [e for e in events if e.stage.value == "visual" and e.percent is not None]
        assert len(percent_events) >= 2
        percents = [e.percent for e in percent_events]
        assert percents == sorted(percents) and percents[-1] == 100.0
        assert all(e.file_name for e in percent_events)
        assert any("title_t01.mkv" in e.message for e in percent_events)

    def test_subtitles_stage_reports_percent_per_file(
        self,
        config: AppConfig,
        rip_folder: Path,
        thediscdb_repo: Path,
    ) -> None:
        mkv1 = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        mkv2 = make_mkv("title_t01.mkv", mpls="00801.mpls", duration=41 * 60 + 55, size=2_011_000_000)
        events: list[ProgressEvent] = []
        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv1, "title_t01.mkv": mkv2}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
            progress=events.append,
        )
        pipeline.scan_folder(
            rip_folder,
            series="Test Show",
            season=1,
            options=ScanOptions(depth="deep", use_subtitles=True, use_audio=False, use_llm=False),
        )
        percent_events = [e for e in events if e.stage.value == "subtitles" and e.percent is not None]
        assert percent_events, "subtitles stage must emit per-file percent"
        percents = [e.percent for e in percent_events]
        assert percents == sorted(percents) and percents[-1] == 100.0
        assert all(e.file_name for e in percent_events)

class TestDiscTitlePrecedence:
    """Review 1.3: the disc's own episode title wins over the TMDB title
    (broadcast vs production naming), but generic placeholder titles
    ("Episode 1") fall back to TMDB. Disagreements become visible evidence."""

    def _disc_repo(self, tmp_path: Path, e01_title: str) -> Path:
        import json as _json

        root = tmp_path / "discdb"
        release_dir = root / "data" / "series" / "Test Show (2020)" / "season-1"
        release_dir.mkdir(parents=True)
        payload = {
            "Index": 1,
            "Slug": "blu-ray",
            "Name": "Test Show Season 1 Disc 1",
            "Format": "Blu-Ray",
            "Titles": [
                {
                    "Index": 0,
                    "SourceFile": "00800.mpls",
                    "SegmentMap": "1-11,12",
                    "Duration": "0:42:30",
                    "Size": 2_028_000_000,
                    "Item": {"Title": e01_title, "Type": "Episode", "Season": "1", "Episode": "1"},
                },
            ],
        }
        (release_dir / "disc01.json").write_text(_json.dumps(payload), encoding="utf-8")
        return root

    def _scan(self, config: AppConfig, repo: Path, rip_folder: Path) -> ScanReport:
        mkv = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        events: list[ProgressEvent] = []
        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(repo),
            tmdb=FakeTmdb(make_guide()),
            progress=events.append,
        )
        return pipeline.scan_folder(
            rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast")
        )

    def test_real_disc_title_wins_and_records_disagreement(
        self, config: AppConfig, rip_folder: Path, tmp_path: Path
    ) -> None:
        report = self._scan(config, self._disc_repo(tmp_path, "The Premiere"), rip_folder)
        entry = report.assignment.entries[0]
        assert entry.episode_title == "The Premiere"  # disc title beats TMDB "Pilot"
        evidence = [e for e in entry.evidence if e.kind == "title_precedence"]
        assert evidence, "the disagreement must be visible as evidence"
        assert "The Premiere" in evidence[0].summary and "Pilot" in evidence[0].summary

    def test_generic_disc_title_falls_back_to_tmdb(
        self, config: AppConfig, rip_folder: Path, tmp_path: Path
    ) -> None:
        report = self._scan(config, self._disc_repo(tmp_path, "Episode 1"), rip_folder)
        entry = report.assignment.entries[0]
        assert entry.episode_title == "Pilot"  # TMDB wins over a placeholder


class TestEpisodeRangeAssignment:
    """Review 1.1: a two-part story published as `Episode: "1-2"` is one file
    covering two episodes — it must be a candidate, assigned to its start
    episode, carry the span, and consume the covered follow-up episode."""

    def _disc_repo(self, tmp_path: Path) -> Path:
        import json as _json

        root = tmp_path / "discdb"
        release_dir = root / "data" / "series" / "Test Show (2020)" / "season-1"
        release_dir.mkdir(parents=True)
        payload = {
            "Index": 1,
            "Slug": "blu-ray",
            "Name": "Test Show Season 1 Disc 1",
            "Format": "Blu-Ray",
            "Titles": [
                {
                    "Index": 0,
                    "SourceFile": "00800.mpls",
                    "SegmentMap": "1-11,12",
                    "Duration": "0:43:00",
                    "Size": 2_028_000_000,
                    "Item": {"Title": "London", "Type": "Episode", "Season": "1", "Episode": "1-2"},
                },
            ],
        }
        (release_dir / "disc01.json").write_text(_json.dumps(payload), encoding="utf-8")
        return root

    def test_range_entry_assigned_with_span(
        self, config: AppConfig, rip_folder: Path, tmp_path: Path
    ) -> None:
        mkv = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=43 * 60, size=2_100_000_000)
        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(self._disc_repo(tmp_path)),
            tmdb=FakeTmdb(make_guide()),
        )
        report = pipeline.scan_folder(
            rip_folder, series="Test Show", season=1, options=ScanOptions(depth="fast")
        )
        entry = report.assignment.entries[0]
        assert entry.episode == 1
        assert entry.episode_end == 2  # the span reaches the assignment
        assert entry.episode_title == "London"

class TestReferenceIdfAndSync:
    """Review 4.1/4.2 wiring in the pipeline: the coverage is IDF-weighted
    across the loaded references (>= 3 non-empty), and a reference timed to a
    different master (> 20 s) is tagged as evidence instead of silently
    depressing the score."""

    @staticmethod
    def _sub(lines: list[tuple[float, float, str]], language: str = "ger") -> Transcript:
        from episodematch.core.models import SubtitleLine

        return Transcript(
            source="test",
            language=language,
            track_index=1,
            lines=[
                SubtitleLine(start_seconds=start, end_seconds=end, text=line)
                for start, end, line in lines
            ],
        )

    @staticmethod
    def _three_episode_guide() -> SeasonGuide:
        from episodematch.core.models import EpisodeInfo, SeriesInfo

        series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id="tt1234567", first_air_year=2020)
        episodes = [
            EpisodeInfo(season=1, episode=1, title="Pilot", plot="e1 plot filler"),
            EpisodeInfo(season=1, episode=2, title="Second", plot="e2 plot filler"),
            EpisodeInfo(season=1, episode=3, title="Third", plot="e3 plot filler"),
        ]
        return SeasonGuide(series=series, season=1, episodes=episodes)

    GENERIC = "the springfield police department arrived"

    def test_idf_keeps_generic_small_reference_from_winning(self, config: AppConfig) -> None:
        from episodematch.core.models import MatchCandidate

        pipeline = build_pipeline(config)
        candidate = MatchCandidate(file_path=Path("D:/rips/title_t00.mkv"))
        guide = self._three_episode_guide()

        e01_ref = self._sub(
            [(float(i), float(i) + 1.5, line) for i, line in enumerate(
                [self.GENERIC, "the lighthouse keeper whispers to the tide",
                 "a lantern burns alone on the northern pier"])]
        )
        e02_lines = [self.GENERIC] + [
            "the counterfeit passports were hidden under the floorboards",
            "elizabeth burned the microfilm before the courier arrived",
        ] + [f"martha packed the radio transmitter crate {i} for the run" for i in range(20)]
        e02_ref = self._sub([(float(i), float(i) + 1.5, line) for i, line in enumerate(e02_lines)])
        e03_ref = self._sub(
            [(float(i), float(i) + 1.5, line) for i, line in enumerate(
                [self.GENERIC, "grain prices collapse after the storm"])]
        )
        refs = {1: e01_ref, 2: e02_ref, 3: e03_ref}
        # The rip contains every generic line plus the true E02 dialog.
        transcript = self._sub([(float(i), float(i) + 1.5, line) for i, line in enumerate(e02_lines)])

        pipeline._score_transcript(candidate, transcript, guide, lambda episode, lang: refs.get(episode.episode))
        text_evidence = [e for e in candidate.evidence if e.kind == "text_match" and e.episode_hint]
        best = max(text_evidence, key=lambda e: e.score)
        # Unweighted, E01's 3/4 coverage would outrank E02's 23/24 — with IDF the
        # generic lines drop out and the true episode wins.
        assert best.episode_hint == 2

    def test_shifted_reference_tagged_as_different_master(self, config: AppConfig) -> None:
        from episodematch.core.models import MatchCandidate

        pipeline = build_pipeline(config)
        candidate = MatchCandidate(file_path=Path("D:/rips/title_t00.mkv"))
        guide = self._three_episode_guide()

        own = self._sub([(10.0, 12.0, "own dialog line one"), (100.0, 103.0, "own dialog line two")])
        shifted = self._sub([(70.0, 72.0, "own dialog line one"), (160.0, 163.0, "own dialog line two")])
        refs = {1: shifted, 2: None, 3: None}
        pipeline._score_transcript(candidate, own, guide, lambda episode, lang: refs.get(episode.episode))
        sync_evidence = [e for e in candidate.evidence if e.kind == "sync_check"]
        assert sync_evidence, "a >20 s reference offset must be reported as evidence"
        assert "different master" in sync_evidence[0].summary

class TestVisualOnlyDvdScan:
    """PILLAR4_PLAN batch 5 (user case): DVDs outside the DiscDB with no
    subtitles — pillar 4 alone must assign the episode from a 100% title-card
    match after the active-pillar normalization."""

    def test_visual_only_scan_assigns_by_title_match(
        self, config: AppConfig, tmp_path: Path
    ) -> None:
        from episodematch.core.models import MkvFile, TitleCardHit, TitleCardKind

        (tmp_path / "dvd_rip.mkv").write_bytes(b"x")  # the scan globs the folder
        mkv = MkvFile(path=tmp_path / "dvd_rip.mkv", size_bytes=1, duration_seconds=2641.0)
        hits = [
            TitleCardHit(
                timestamp_seconds=248.0,
                kind=TitleCardKind.TITLE_CARD,
                raw_lines=["PROTOTYPE"],
                title_candidates=["PROTOTYPE"],
                ocr_confidence=0.99,
            )
        ]

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                return hits

        from episodematch.core.models import EpisodeInfo, SeriesInfo

        guide = SeasonGuide(
            series=SeriesInfo(
                name="Star Trek: Voyager",
                tmdb_id=46533,
                imdb_id="tt2149175",
                first_air_year=1995,
            ),
            season=2,
            episodes=[
                EpisodeInfo(
                    season=2,
                    episode=13,
                    title="Prototype",
                    plot="The Doctor builds a robot prototype",
                )
            ],
        )
        (tmp_path / "empty-discdb" / "data" / "series").mkdir(parents=True)
        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"dvd_rip.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(tmp_path / "empty-discdb"),
            tmdb=FakeTmdb(guide),
            frame_scanner=FakeVisual(),
        )
        report = pipeline.scan_folder(
            tmp_path,
            series="Star Trek: Voyager",
            season=2,
            options=ScanOptions(depth="deep", use_disc_db=False, use_subtitles=False, use_audio=False, use_llm=False),
        )
        assert report.assignment.entries, "the 100% title match must be assignable"
        entry = report.assignment.entries[0]
        assert entry.episode == 13
        assert entry.episode_title == "Prototype"
        # honest absolute score (0.35 title weight) — the assignment decision
        # used the scaled threshold (only the visual pillar is active)
        assert entry.confidence == pytest.approx(0.35)

    def test_enabled_but_empty_pillars_do_not_raise_threshold(
        self, config: AppConfig, tmp_path: Path
    ) -> None:
        """User case v0.12.0: the GUI enables disc_db + subtitles + visual, but
        the disc is outside the DiscDB and the reference fetch yields nothing —
        only pillar 4 produces evidence. The threshold must normalize by the
        pillars that CONTRIBUTED evidence, not by the enabled options —
        otherwise the 100% title match (0.35) is suppressed by the full-scale
        0.55 x (0.35+0.30+0.20) = 0.47 gate and EVERYTHING stays unassigned."""
        from episodematch.core.exceptions import EpisodeMatchError
        from episodematch.core.models import TitleCardHit, TitleCardKind

        (tmp_path / "dvd_rip.mkv").write_bytes(b"x")  # the scan globs the folder
        mkv = MkvFile(path=tmp_path / "dvd_rip.mkv", size_bytes=1, duration_seconds=2641.0)
        hits = [
            TitleCardHit(
                timestamp_seconds=248.0,
                kind=TitleCardKind.TITLE_CARD,
                raw_lines=["PROTOTYPE"],
                title_candidates=["PROTOTYPE"],
                ocr_confidence=0.99,
            )
        ]

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                return hits

        class FakeBrokenExtractor:
            """Subtitle stage enabled, but extraction fails per file (no tools
            / no usable track) — exactly the 'enabled but empty' case."""

            def extract_all(self, mkv, **kwargs):  # noqa: ANN001, ANN202
                raise EpisodeMatchError("no mkvextract available")

        guide = SeasonGuide(
            series=SeriesInfo(
                name="Star Trek: Voyager",
                tmdb_id=46533,
                imdb_id="tt2149175",
                first_air_year=1995,
            ),
            season=2,
            episodes=[
                EpisodeInfo(
                    season=2,
                    episode=13,
                    title="Prototype",
                    plot="The Doctor builds a robot prototype",
                )
            ],
        )
        (tmp_path / "empty-discdb" / "data" / "series").mkdir(parents=True)
        pipeline = build_pipeline(
            config,
            ffprobe=FakeFfprobe({"dvd_rip.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(tmp_path / "empty-discdb"),
            tmdb=FakeTmdb(guide),
            subtitle_extractor=FakeBrokenExtractor(),
            frame_scanner=FakeVisual(),
        )
        report = pipeline.scan_folder(
            tmp_path,
            series="Star Trek: Voyager",
            season=2,
            options=ScanOptions(depth="deep", use_disc_db=True, use_subtitles=True, use_audio=False, use_llm=False),
        )
        assert report.assignment.entries, (
            "a 0.35 title match must assign when disc_db/subtitle produced NO evidence "
            "(weight_sum must be visual-only 0.20, not the enabled-option sum 0.85)"
        )
        entry = report.assignment.entries[0]
        assert entry.episode == 13
        assert entry.confidence == pytest.approx(0.35)
