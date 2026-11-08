"""Render screenshots of every GUI page for design verification.

Usage:
    python scripts/screenshot_ui.py [--out DIR] [--platform NAME]

Defaults to the native platform (real fonts) on Windows; pass
``--platform offscreen`` for headless CI — note that the offscreen plugin has no
font directory, so text renders as tofu boxes there.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PARSER = argparse.ArgumentParser(description="Render GUI page screenshots")
PARSER.add_argument("--out", default="data/screens", help="output directory (default: data/screens)")
PARSER.add_argument(
    "--platform",
    default=None,
    help="Qt platform plugin (default: 'windows' on win32, else 'offscreen')",
)
_args = PARSER.parse_args()

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    _args.platform or ("windows" if sys.platform == "win32" else "offscreen"),
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from episodematch.core.models import (  # noqa: PLC0415 - after Qt import is fine
        AssignmentEntry,
        Evidence,
        MatchCandidate,
        Pillar,
        SeasonAssignment,
        SeriesInfo,
        UnassignedFile,
    )
    from episodematch.matcher.pipeline import ScanReport
    from episodematch.ui.main_window import FluentWindow, create_app_context
    from episodematch.ui.theme import apply_theme

    out_dir = Path(_args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication([])
    apply_theme(app)
    window = FluentWindow(create_app_context())
    window.resize(1280, 820)
    window.show()
    app.processEvents()

    # Seed the workspace with a realistic finished scan so screenshots show content.
    evidence = [
        Evidence(pillar=Pillar.DISC_DB, kind="fingerprint_match", score=1.0,
                 summary="Disc fingerprint matches Season 1 Disc 1 (MPLS 00800.mpls) -> S01E01 Pilot",
                 details={"reasons": ["MPLS 00800.mpls matches disc entry", "Duration 2550s ±0.4s", "Size within 2%"]}),
        Evidence(pillar=Pillar.SUBTITLE, kind="text_match", score=0.71,
                 summary="S01E01: 79% text match (4 aligned dialog lines, plot overlap 0.41)",
                 details={"dialog_score": 0.79, "plot_score": 0.41}, episode_hint=1),
        Evidence(pillar=Pillar.VISUAL, kind="director_ocr", score=0.97,
                 summary="Director 'Michelle MacLare' recognized on screen at 01:42 matches 'Michelle MacLaren'",
                 details={"role": "director", "ratio": 97.4}, timestamp_seconds=102.0, episode_hint=1),
        Evidence(pillar=Pillar.LLM, kind="arbiter_assignment", score=0.93,
                 summary="LLM arbiter (glm-4-flash) assigns title_t00.mkv to E01 at 93%: dialog and credits agree",
                 details={"reason": "dialog and credits agree"}, episode_hint=1),
    ]
    entry = AssignmentEntry(
        file_path=Path("D:/rips/title_t00.mkv"), season=1, episode=1, episode_title="Pilot",
        confidence=0.98, methods=["disc_db", "subtitle", "visual", "llm"],
        reasoning=[e.summary for e in evidence], evidence=evidence,
    )
    entry2 = AssignmentEntry(
        file_path=Path("D:/rips/title_t01.mkv"), season=1, episode=2, episode_title="Cat's in the Bag...",
        confidence=0.96, methods=["disc_db"], reasoning=["MPLS 00801.mpls match"], evidence=evidence[:1],
    )
    from episodematch.core.models import MatchCandidate, UnassignedFile  # noqa: PLC0415

    candidates = [MatchCandidate(file_path=entry.file_path), MatchCandidate(file_path=entry2.file_path)]
    for candidate, ev in zip(candidates, (evidence, evidence[:1])):
        for item in ev:
            candidate.add_evidence(item)
    assignment = SeasonAssignment(
        series=SeriesInfo(name="Breaking Bad", tmdb_id=1396, first_air_year=2008),
        season=1,
        entries=[entry, entry2],
        unassigned=[UnassignedFile(file_path=Path("D:/rips/title_t02.mkv"), reason="Lost the global assignment conflict", best_guess_episode=3, best_guess_confidence=0.41)],
    )
    report = ScanReport(assignment=assignment, candidates=candidates)
    window._workspace._context.last_report = report
    window._workspace._context.last_folder = Path("D:/rips")
    window._workspace._folder_edit.setText("D:/rips/Breaking Bad S01")
    window._workspace._fill_results(report)
    window._workspace._table.selectRow(0)
    window._workspace._on_selection_changed()
    window._workspace._show_banner("success", "2/3 assigned, average confidence 97%")
    app.processEvents()

    pages = {
        0: "workspace",
        1: "visual_verify",
        2: "postprocess",
        3: "settings",
    }
    for index, name in pages.items():
        window.navigate(index)
        app.processEvents()
        pixmap = window.grab()
        target = out_dir / f"{name}.png"
        pixmap.save(str(target))
        print("saved", target)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
