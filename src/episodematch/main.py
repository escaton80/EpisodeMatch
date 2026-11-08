"""EpisodeMatch entry point: version, check-env, fetch-tools and the match CLI.

The PySide6 GUI arrives in phase 3; this module is the complete command-line
surface and works without any GUI dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from episodematch import __app_name__, __version__
from episodematch.core.binaries import TOOL_NAMES, ToolResolver
from episodematch.core.config import AppConfig, ConfigStore, apply_env_overrides
from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import EpisodeMatchError
from episodematch.core.models import ProgressEvent
from episodematch.matcher.pipeline import Pipeline, ScanOptions, ScanReport

logger = logging.getLogger("episodematch")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname).1s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # httpx logs every request at INFO — noise for users; keep for --verbose.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbose else logging.WARNING)


def _print_progress(event: ProgressEvent) -> None:
    prefix = event.file_name or ""
    percent = f" [{event.percent:5.1f}%]" if event.percent is not None else ""
    print(f"\x1b[2m{event.stage.value:>9}\x1b[0m {prefix}{percent} {event.message}", file=sys.stderr)


def _print_report(report: ScanReport, *, as_json: bool) -> None:
    assignment = report.assignment
    if as_json:
        payload = {
            "series": assignment.series.name,
            "season": assignment.season,
            "entries": [
                {
                    "file": entry.file_path.name,
                    "code": entry.code,
                    "title": entry.episode_title,
                    "confidence": round(entry.confidence, 4),
                    "methods": entry.methods,
                    "reasoning": entry.reasoning,
                }
                for entry in assignment.entries
            ],
            "unassigned": [
                {
                    "file": item.file_path.name,
                    "reason": item.reason,
                    "best_guess_episode": item.best_guess_episode,
                    "best_guess_confidence": round(item.best_guess_confidence, 4),
                }
                for item in assignment.unassigned
            ],
            "pillar_reports": [
                {"file": item.file_name, "pillar": item.pillar, "status": item.status, "detail": item.detail}
                for item in report.pillar_reports
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\n{assignment.series.name} — Season {assignment.season}")
    print("=" * (len(assignment.series.name) + 12))
    for entry in assignment.entries:
        confidence_bar = _bar(entry.confidence)
        print(f"  {entry.file_path.name:<40} -> {entry.code}  {confidence_bar} {entry.confidence:5.0%}  {entry.episode_title or ''}")
        for reason in entry.reasoning[:4]:
            print(f"      • {reason}")
    if assignment.unassigned:
        print("\n  Unassigned:")
        for item in assignment.unassigned:
            guess = f" (best guess: E{item.best_guess_episode:02d} at {item.best_guess_confidence:.0%})" if item.best_guess_episode else ""
            print(f"  {item.file_path.name:<40} -> ?   {item.reason}{guess}")
    print(f"\n  {assignment.assigned_count} assigned, avg confidence {assignment.average_confidence:.0%}")

    failed = [r for r in report.pillar_reports if r.status == "failed"]
    if failed:
        print("  Pillar problems:")
        for item in failed[:10]:
            print(f"      ✗ [{item.pillar}] {item.file_name}: {item.detail[:120]}")


def _bar(value: float, width: int = 12) -> str:
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_gui(_args: argparse.Namespace) -> int:
    """Launch the PySide6 GUI (requires the [gui] extra)."""
    try:
        import PySide6  # noqa: F401,PLC0415
    except ImportError:
        print("PySide6 is not installed — run: pip install -e .[gui]", file=sys.stderr)
        return 1
    from episodematch.ui.main_window import run_gui

    config = _load_config()
    return run_gui(config)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"{__app_name__} {__version__}")
    return 0


def _load_config() -> AppConfig:
    store = ConfigStore()
    try:
        return store.load()
    except EpisodeMatchError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return apply_env_overrides(AppConfig())


def _cmd_check_env(args: argparse.Namespace) -> int:
    config = _load_config()
    resolver = ToolResolver(config.resolved_bin_dir)
    status = resolver.status(refresh=True)
    print(f"{__app_name__} {__version__} environment check")
    print("-" * 60)
    for name in TOOL_NAMES:
        info = status.get(name)
        if info.is_available and info.path:
            source = {"local_bin": "./bin/", "system_path": "PATH", "program_files": "Program Files"}.get(info.source.value, info.source.value)
            print(f"  ✓ {name:<12} {str(info.path):<60} v{info.version or '?'} ({source})")
        else:
            print(f"  ✗ {name:<12} not found — run: episodematch fetch-tools")
    from episodematch.extractors.ocr_engine import OcrEngine

    ocr = OcrEngine(bin_dir=config.resolved_bin_dir)
    availability = ocr.availability
    if availability.available:
        print(f"  ✓ {'ocr':<12} backend: {availability.name}")
    else:
        print(f"  ✗ {'ocr':<12} {availability.reason}")
    from episodematch.core.binaries import locate_subtitle_edit

    subtitle_edit = locate_subtitle_edit(config.resolved_bin_dir)
    if subtitle_edit is not None:
        print(f"  ✓ {'subedit':<12} {subtitle_edit}  (fast PGS→SRT conversion)")
    else:
        print(
            f"  · {'subedit':<12} optional — PGS pillar uses the Python OCR fallback (first "
            f"{config.extraction.pgs_max_events} events only)"
        )
    for key_name, field_name in (
        ("TMDB", "tmdb_api_key"),
        ("OpenSubtitles", "opensubtitles_api_key"),
        ("SubDL", "subdl_api_key"),
        ("LLM (GLM)", "llm_api_key"),
    ):
        value = getattr(config, field_name)
        state = "set" if value else "missing (pillar will be skipped)"
        print(f"  {'✓' if value else '·'} {'key':<12} {key_name:<14} {state}")
    print(f"  · subs source  {config.subtitle_source}")
    print(f"  · data dir: {config.resolved_data_dir}")
    if args.fix:
        return _cmd_fetch_tools(args)
    missing = status.missing
    if missing:
        print(f"\n  Missing tools: {', '.join(missing)} — run 'episodematch fetch-tools' to download them.")
    return 0


def _cmd_fetch_tools(args: argparse.Namespace) -> int:
    from episodematch.core.bin_downloader import BinaryDownloader

    config = _load_config()
    downloader = BinaryDownloader(config.resolved_bin_dir)

    def progress(label: str, done: int, total: int | None) -> None:
        if total:
            percent = done / total * 100
            sys.stderr.write(f"\r  downloading {label}: {percent:5.1f}% ({done // 1024 // 1024} MiB)")
            if done >= total:
                sys.stderr.write("\n")

    try:
        result = downloader.download_missing(progress=progress, force=args.force or [])
    except EpisodeMatchError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    if result.installed:
        print(f"  ✓ installed into {config.resolved_bin_dir}: {', '.join(result.installed)}")
    else:
        print("  All required tools are already present.")
    resolver = ToolResolver(config.resolved_bin_dir)
    remaining = resolver.status(refresh=True).missing
    if remaining:
        print(f"  Still missing: {', '.join(remaining)}", file=sys.stderr)
        return 1
    return 0


def _cmd_match(args: argparse.Namespace) -> int:
    config = _load_config()
    folder = Path(args.folder)
    options = ScanOptions(
        depth=args.depth,
        use_disc_db=not args.no_discdb,
        use_subtitles=not args.no_subtitles,
        use_audio=not args.no_audio,
        use_visual=not args.no_visual,
        use_llm=not args.no_llm,
        local_discdb_root=Path(args.local_discdb) if args.local_discdb else None,
        discdb_url=args.discdb_url,
    )
    database: CacheDatabase | None = None
    try:
        database = CacheDatabase(config.database_path)
        pipeline = Pipeline(config, database=database, progress=_print_progress)
        report = pipeline.scan_folder(folder, series=args.series, season=args.season, options=options)
    except EpisodeMatchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if database is not None:
            database.close()

    _print_report(report, as_json=args.json)
    exit_code = 0
    if report.assignment.unassigned:
        exit_code = 2
    if args.rename and report.assignment.entries:
        exit_code = _apply_renames(config, report, as_json=args.json) or exit_code
    if args.tag and report.assignment.entries and not args.json:
        exit_code = _apply_tags(config, report) or exit_code
    return exit_code


def _apply_renames(config: AppConfig, report: ScanReport, *, as_json: bool) -> int | None:
    from episodematch.postprocess.renamer import Renamer, build_rename_plan

    template = config.naming.episode_template
    plan = build_rename_plan(
        report.assignment.entries,
        series=report.assignment.series.name,
        template=template,
        season_folder=config.naming.season_folder,
        lowercase_series=config.naming.lowercase_series,
        replace_spaces_with_dots=config.naming.replace_spaces_with_dots,
        dry_run=False,
    )
    renamer = Renamer(config.undo_dir)
    try:
        applied = renamer.apply(plan)
    except EpisodeMatchError as exc:
        print(f"Rename failed: {exc}", file=sys.stderr)
        return 1
    for item in applied:
        print(f"  renamed: {item.file_name} -> {item.destination}")
    return None


def _apply_tags(config: AppConfig, report: ScanReport) -> int | None:
    from episodematch.postprocess.mkv_tagger import MkvTagger

    resolver = ToolResolver(config.resolved_bin_dir)
    try:
        propedit = resolver.tool_path("mkvpropedit")
    except EpisodeMatchError:
        print("  mkvpropedit not found - tagging skipped", file=sys.stderr)
        return None
    tagger = MkvTagger(propedit)
    for entry in report.assignment.entries:
        try:
            tagger.apply_entry(entry)
        except EpisodeMatchError as exc:
            print(f"  tagging failed for {entry.file_path.name}: {exc}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="episodematch",
        description=f"{__app_name__} {__version__} — identify ripped series episodes with near-100% certainty.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command")

    gui_parser = sub.add_parser("gui", help="launch the Windows 11 GUI (requires PySide6)")
    gui_parser.set_defaults(func=_cmd_gui)

    version_parser = sub.add_parser("version", help="print the version")
    version_parser.set_defaults(func=_cmd_version)

    env_parser = sub.add_parser("check-env", help="check external tools, OCR and API keys")
    env_parser.add_argument("--fix", action="store_true", help="also download missing tools")
    env_parser.set_defaults(func=_cmd_check_env)

    fetch_parser = sub.add_parser("fetch-tools", help="download missing ffmpeg/MKVToolNix binaries into ./bin/")
    fetch_parser.add_argument("--force", nargs="*", choices=list(TOOL_NAMES), help="re-download even if present")
    fetch_parser.set_defaults(func=_cmd_fetch_tools)

    match_parser = sub.add_parser("match", help="identify all .mkv files of a folder")
    match_parser.add_argument("folder", help="folder containing the ripped .mkv files")
    match_parser.add_argument("--series", required=True, help='series name as known by TMDB, e.g. "Breaking Bad"')
    match_parser.add_argument("--season", required=True, type=int, help="season number of the disc")
    match_parser.add_argument("--depth", choices=["fast", "deep"], default="fast", help="fast (metadata+subs) or deep (all pillars)")
    match_parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    match_parser.add_argument("--rename", action="store_true", help="apply the renamer after matching")
    match_parser.add_argument("--tag", action="store_true", help="write episode titles via mkvpropedit")
    match_parser.add_argument("--no-discdb", action="store_true", help="disable pillar 1 (TheDiscDB)")
    match_parser.add_argument("--no-subtitles", action="store_true", help="disable pillar 2 (subtitles)")
    match_parser.add_argument("--no-audio", action="store_true", help="disable pillar 3 (whisper)")
    match_parser.add_argument("--no-visual", action="store_true", help="disable pillar 4 (title-card OCR)")
    match_parser.add_argument("--no-llm", action="store_true", help="disable pillar 5 (LLM arbiter)")
    match_parser.add_argument("--local-discdb", help="use a local TheDiscDb/data clone instead of GitHub")
    match_parser.add_argument(
        "--discdb-url",
        help=(
            "target one TheDiscDB disc/release directly, e.g. "
            "https://thediscdb.com/series/south-park-1997/releases/seasons-6-10-blu-ray/discs/season-06-disc-01"
        ),
    )
    match_parser.set_defaults(func=_cmd_match)
    return parser


def _force_utf8_stdio() -> None:
    """Make ✓/✗/· output safe on legacy Windows consoles (cp1252 and friends).

    Frozen executables in particular inherit the console codepage; without this,
    any non-ASCII status glyph would raise UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # pragma: no cover - exotic streams
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    if not getattr(args, "command", None):
        # No command given: launch the GUI when PySide6 is available, else help.
        try:
            import PySide6  # noqa: F401,PLC0415
        except ImportError:
            parser.print_help()
            return 0
        return _cmd_gui(args)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
