"""Shared pytest fixtures for the EpisodeMatch offline test suite."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

from episodematch.core.config import AppConfig, apply_env_overrides
from episodematch.extractors.pgs_parser import (
    Palette,
    PaletteEntry,
    PgsBitmap,
    build_sup_event,
)

_session_start: float | None = None
_counts: dict[str, int] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}


def pytest_configure(config: pytest.Config) -> None:
    global _session_start
    _session_start = time.time()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Count outcomes for the compact EM summary line (see sessionfinish)."""
    if os.environ.get("EM_QT_TESTS") != "1":
        return
    if report.when == "call":
        _counts[report.outcome] = _counts.get(report.outcome, 0) + 1
    elif report.when == "setup" and report.outcome != "passed":
        _counts["error"] = _counts.get("error", 0) + 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """Deterministic process exit after Qt test sessions.

    PySide6 can segfault during interpreter finalization — strictly AFTER every
    test has run and passed — when QThreads/widgets outlive the QApplication.
    Exiting right here (before that teardown region) keeps the run reliable;
    pytest's own fancy summary line is replaced by a compact EM summary.
    Trade-off: with the Qt suite in the same run, ``--cov`` data would not be
    flushed.
    """
    if os.environ.get("EM_QT_TESTS") != "1" or os.environ.get("EM_NO_HARD_EXIT") == "1":
        return
    duration = (time.time() - _session_start) if _session_start else 0.0
    print(
        f"\nEM SUMMARY: {_counts['passed']} passed, {_counts['failed']} failed, "
        f"{_counts['error']} error(s), {_counts['skipped']} skipped in {duration:.1f}s "
        f"[exitstatus={exitstatus}, testsfailed={getattr(session, 'testsfailed', '?')}]",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exitstatus))


@pytest.fixture()
def config(tmp_path: Path) -> AppConfig:
    """AppConfig pointed at temp dirs, with env overrides cleared."""
    import os

    saved = {}
    env_keys = [
        "EPISODEMATCH_TMDB_API_KEY",
        "EPISODEMATCH_OSDT_API_KEY",
        "EPISODEMATCH_LLM_API_KEY",
        "EPISODEMATCH_LLM_BASE_URL",
        "EPISODEMATCH_LLM_MODEL",
        "EPISODEMATCH_BIN_DIR",
        "EPISODEMATCH_DATA_DIR",
    ]
    for key in env_keys:
        saved[key] = os.environ.pop(key, None)
    try:
        cfg = AppConfig(
            bin_dir=tmp_path / "bin",
            data_dir=tmp_path / "data",
        )
        yield apply_env_overrides(cfg)
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


class FakeOcrBackend:
    """Deterministic OCR backend returning canned lines per image size/color."""

    name = "fake"

    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = lines or []
        self.calls = 0

    def probe(self):
        from episodematch.extractors.ocr_engine import BackendAvailability

        return BackendAvailability(self.name, True)

    def recognize(self, image):
        from episodematch.core.models import OcrLine, OcrResult

        self.calls += 1
        lines = []
        for index, text in enumerate(self._lines):
            lines.append(
                OcrLine(
                    text=text,
                    x=10,
                    y=10 + index * 60,
                    width=max(60, len(text) * 22),
                    height=48,
                    confidence=0.95,
                )
            )
        return OcrResult(lines=lines, backend=self.name)


@pytest.fixture()
def fake_ocr() -> FakeOcrBackend:
    return FakeOcrBackend(["Directed by Michelle MacLaren", "Written by Vince Gilligan"])


@pytest.fixture()
def standard_palette() -> Palette:
    """A simple palette: 1 = white text, 0 = transparent black outline."""
    return Palette(
        palette_id=0,
        version=0,
        entries={
            0: PaletteEntry(color_id=0, y=16, cr=128, cb=128, alpha=0),
            1: PaletteEntry(color_id=1, y=235, cr=128, cb=128, alpha=255),
            2: PaletteEntry(color_id=2, y=128, cr=128, cb=128, alpha=128),
        },
    )


def make_bitmap(indices: bytes, width: int, height: int, *, forced: bool = False) -> PgsBitmap:
    return PgsBitmap(object_id=0, x=640, y=900, width=width, height=height, indices=indices, forced=forced)


def make_sup_bytes(
    events: list[tuple[float, float, PgsBitmap]],
    palette: Palette,
) -> bytes:
    """Build a multi-event .sup stream."""
    out = bytearray()
    for composition_number, (start, end, bitmap) in enumerate(events):
        out += build_sup_event(
            start_seconds=start,
            end_seconds=end,
            bitmaps=[bitmap],
            palette=palette,
            composition_number=composition_number * 10,
        )
    return bytes(out)


@pytest.fixture()
def sample_ffprobe_json() -> dict:
    """An ffprobe JSON document shaped like real MakeMKV output."""
    return {
        "format": {
            "filename": "title_t00.mkv",
            "format_name": "matroska,webm",
            "duration": "1273.512000",
            "size": "2028240896",
            "bit_rate": "12739102",
            "tags": {"TITLE": "00800.mpls", "ENCODER": "MakeMKV 1.17"},
        },
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24000/1001",
                "disposition": {"default": 1},
            },
            {
                "index": 1,
                "codec_name": "dts",
                "codec_type": "audio",
                "channels": 6,
                "sample_rate": "48000",
                "tags": {"language": "eng", "title": "DTS-HD 5.1"},
                "disposition": {"default": 1},
            },
            {
                "index": 2,
                "codec_name": "dts",
                "codec_type": "audio",
                "channels": 2,
                "sample_rate": "48000",
                "tags": {"language": "ger", "title": "Kommentar"},
                "disposition": {"comment": 1},
            },
            {
                "index": 3,
                "codec_name": "hdmv_pgs_subtitle",
                "codec_type": "subtitle",
                "tags": {"language": "ger"},
                "disposition": {"forced": 1},
            },
            {
                "index": 4,
                "codec_name": "subrip",
                "codec_type": "subtitle",
                "tags": {"language": "eng"},
            },
        ],
        "chapters": [
            {"start_time": "0.000000", "end_time": "600.000000", "tags": {"title": "Chapter 1"}},
            {"start_time": "600.000000", "end_time": "1273.512000", "tags": {"title": "Chapter 2"}},
        ],
    }


@pytest.fixture()
def thediscdb_repo(tmp_path: Path) -> Path:
    """A local TheDiscDb/data checkout skeleton with one series and one release."""
    root = tmp_path / "discdb"
    release_dir = root / "data" / "series" / "Test Show (2020)" / "season-1"
    release_dir.mkdir(parents=True)
    disc_payload = {
        "Index": 1,
        "Slug": "blu-ray",
        "Name": "Test Show Season 1 Disc 1",
        "Format": "Blu-Ray",
        "ContentHash": "H1",
        "GlobalDiscId": "G1",
        "Titles": [
            {
                "Index": 0,
                "SourceFile": "00800.mpls",
                "SegmentMap": "1-11,12",
                "Duration": "0:42:30",
                "Size": 2028000000,
                "Item": {"Title": "Pilot", "Type": "Episode", "Season": "1", "Episode": "1"},
            },
            {
                "Index": 1,
                "SourceFile": "00801.mpls",
                "SegmentMap": "1-11,12",
                "Duration": "0:41:55",
                "Size": 2011000000,
                "Item": {"Title": "Second Wave", "Type": "Episode", "Season": "1", "Episode": "2"},
            },
            {
                "Index": 2,
                "SourceFile": "00055.mpls",
                "SegmentMap": "1012",
                "Duration": "0:00:10",
                "Size": 270336,
                "Item": {"Type": "Extra"},
            },
        ],
    }
    (release_dir / "disc01.json").write_text(json.dumps(disc_payload), encoding="utf-8")
    return root
