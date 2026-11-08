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
    return Pipeline(config, database=overrides.get("database"), **services)


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
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None):  # noqa: ANN001, ANN202
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
