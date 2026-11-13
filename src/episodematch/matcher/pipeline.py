"""The five-pillar scan pipeline: fast metadata first, deep OCR/audio/LLM on demand.

Stages (in order, with early exit when confidence is already conclusive):

1. **PROBE**  — ffprobe every ``.mkv`` in the folder (cached by path/size/mtime).
2. **FAST**   — TheDiscDB fingerprint matching (MPLS/duration/size/segments).
3. Season guide construction (TMDB, optionally enriched by IMDb).
4. **SUBTITLES** — text subtitle extraction or PGS decode + OCR, scored against
   every episode (OpenSubtitles reference when a key is configured, else plot TF-IDF).
5. **AUDIO** — Whisper transcription (only when no usable subtitles exist).
6. **VISUAL** — title-card/credits OCR scoring against the season credits.
7. **LLM** — deep-context arbiter over the whole folder (deep mode or ambiguity).
8. **SOLVE** — Hungarian assignment producing the final :class:`SeasonAssignment`.

Every stage is individually disableable, every pillar failure is non-fatal (it
just means less evidence), and progress/cancellation callbacks are supported for
the future UI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from episodematch.core.config import AppConfig
from episodematch.core.database import CacheDatabase
from episodematch.core.models import (
    DiscFingerprint,
    EpisodeInfo,
    Evidence,
    MatchCandidate,
    MkvFile,
    Pillar,
    PipelineStage,
    ProgressEvent,
    SeasonAssignment,
    SeasonGuide,
    SeriesInfo,
    Transcript,
)
from episodematch.core.exceptions import EpisodeMatchError, ProviderError
from episodematch.extractors.audio_snipper import AudioSnipper
from episodematch.extractors.frame_scanner import FrameScanner
from episodematch.extractors.ffprobe_reader import FfprobeReader
from episodematch.extractors.ocr_engine import OcrEngine
from episodematch.extractors.subtitle_extractor import SubtitleExtractor
from episodematch.matcher.credit_scorer import credit_evidence, score_episode_credits
from episodematch.matcher.llm_arbiter import LlmArbiter, LlmSettings, merge_verdict_into_candidates
from episodematch.matcher.solver import solve_assignment
from episodematch.matcher.text_scorer import score_episode
from episodematch.providers.imdb_scraper import ImdbScraper
from episodematch.providers.opensubtitles import OpenSubtitlesClient, compute_opensubtitles_hash
from episodematch.providers.thediscdb import (
    LocalDiscDbImporter,
    ParsedDisc,
    TheDiscDbClient,
    match_fingerprint,
)
from episodematch.providers.tmdb_client import TmdbClient

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], None]
CancelCallback = Callable[[], bool]


@dataclass
class ScanOptions:
    """Knobs for one folder scan."""

    depth: str = "fast"  # "fast" | "deep"
    use_disc_db: bool = True
    use_subtitles: bool = True
    use_audio: bool = True
    use_visual: bool = True
    use_llm: bool = True
    subtitle_window_minutes: float = 10.0
    ocr_max_events: int = 40
    local_discdb_root: Path | None = None  # offline import from a repo clone
    # Direct TheDiscDb disc/release URL (e.g. pasted from the website) — skips
    # the season heuristic and loads exactly that disc instead of all releases.
    discdb_url: str | None = None


@dataclass
class PillarReport:
    """Per-file status of one pillar (surfaced in the CLI/UI)."""

    file_name: str
    pillar: str
    status: str  # "ok" | "skipped" | "failed"
    detail: str = ""


@dataclass
class ScanReport:
    """Complete result of one folder scan."""

    assignment: SeasonAssignment
    candidates: list[MatchCandidate] = field(default_factory=list)
    pillar_reports: list[PillarReport] = field(default_factory=list)
    guide: SeasonGuide | None = None

    def reports_for(self, file_name: str) -> list[PillarReport]:
        return [report for report in self.pillar_reports if report.file_name == file_name]


class Pipeline:
    """Orchestrates the 5-pillar identification over one folder of MKV files."""

    def __init__(
        self,
        config: AppConfig,
        *,
        database: CacheDatabase | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
        # Injectable services (tests provide fakes):
        ffprobe: FfprobeReader | None = None,
        disc_db_client: TheDiscDbClient | LocalDiscDbImporter | None = None,
        tmdb: TmdbClient | None = None,
        imdb: ImdbScraper | None = None,
        opensubtitles: OpenSubtitlesClient | None = None,
        subdl: Any = None,  # SubDlClient (lazy import to avoid cycles)
        ocr: OcrEngine | None = None,
        subtitle_extractor: SubtitleExtractor | None = None,
        audio_snipper: AudioSnipper | None = None,
        frame_scanner: FrameScanner | None = None,
        llm_arbiter: LlmArbiter | None = None,
    ) -> None:
        self._config = config
        self._database = database
        self._progress_cb = progress or (lambda event: None)
        self._cancel_cb = cancel or (lambda: False)

        from episodematch.core.binaries import ToolResolver

        self._resolver = ToolResolver(config.resolved_bin_dir)
        self._ffprobe = ffprobe
        self._disc_db = disc_db_client
        self._tmdb = tmdb
        self._imdb = imdb
        self._osdt = opensubtitles
        self._subdl = subdl
        self._ocr = ocr
        self._subtitle_extractor = subtitle_extractor
        self._audio_snipper = audio_snipper
        self._frame_scanner = frame_scanner
        self._llm = llm_arbiter
        # When tests/hosts inject their own scanner, it owns its OCR internally.
        self._injected_visual = frame_scanner is not None

    # -- service lazy accessors ---------------------------------------------------

    def _ffprobe_reader(self) -> FfprobeReader:
        if self._ffprobe is None:
            self._ffprobe = FfprobeReader(self._resolver.tool_path("ffprobe"))
        return self._ffprobe

    def _disc_client(self, options: ScanOptions) -> TheDiscDbClient | LocalDiscDbImporter | None:
        if self._disc_db is not None:
            return self._disc_db
        if options.local_discdb_root is not None:
            try:
                self._disc_db = LocalDiscDbImporter(options.local_discdb_root)
            except EpisodeMatchError as exc:
                logger.warning("Local TheDiscDb root unusable: %s", exc)
                return None
            return self._disc_db
        self._disc_db = TheDiscDbClient(self._database)
        return self._disc_db

    def _tmdb_client(self) -> TmdbClient | None:
        if self._tmdb is not None:
            return self._tmdb
        if not self._config.tmdb_api_key:
            return None
        self._tmdb = TmdbClient(self._config.tmdb_api_key, self._database)
        return self._tmdb

    def _imdb_client(self) -> ImdbScraper | None:
        if self._imdb is not None:
            return self._imdb
        self._imdb = ImdbScraper(self._database)
        return self._imdb

    def _osdt_client(self) -> OpenSubtitlesClient | None:
        if self._osdt is not None:
            return self._osdt
        if not self._config.opensubtitles_api_key:
            return None
        self._osdt = OpenSubtitlesClient(
            self._config.opensubtitles_api_key,
            self._database,
            subtitle_dir=self._config.subtitle_cache_dir,
        )
        return self._osdt

    def _subdl_client(self):
        if self._subdl is not None:
            return self._subdl
        if not self._config.subdl_api_key:
            return None
        from episodematch.providers.subdl import SubDlClient

        self._subdl = SubDlClient(
            self._config.subdl_api_key,
            self._database,
            subtitle_dir=self._config.subtitle_cache_dir,
        )
        return self._subdl

    def _reference_client(self):
        """The subtitle-reference client selected by ``subtitle_source``.

        Both clients implement the same ``reference_transcript`` contract, so
        the pillar-2 code is source-agnostic. Returns None when the selected
        source has no API key configured.
        """
        source = self._config.subtitle_source.lower()
        if source == "subdl":
            return self._subdl_client()
        return self._osdt_client()

    def _ocr_engine(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = OcrEngine(bin_dir=self._config.resolved_bin_dir, language="en-US")
        return self._ocr

    def _subtitle_service(self) -> SubtitleExtractor:
        if self._subtitle_extractor is None:
            mkvextract = None
            ffinfo = self._resolver.find("mkvextract")
            if ffinfo.is_available and ffinfo.path:
                mkvextract = ffinfo.path
            ffmpeg = None
            ffinfo = self._resolver.find("ffmpeg")
            if ffinfo.is_available and ffinfo.path:
                ffmpeg = ffinfo.path
            ocr = None
            if self._ocr_engine().available:
                ocr = self._ocr_engine()
            subtitle_edit = None
            seconv = None
            if self._config.extraction.use_subtitle_edit_cli:
                from episodematch.core.binaries import locate_seconv, locate_subtitle_edit

                seconv = locate_seconv(self._config.resolved_bin_dir)
                subtitle_edit = locate_subtitle_edit(self._config.resolved_bin_dir)
                if seconv is not None:
                    logger.info("PGS fast path enabled via seconv CLI (headless OCR): %s", seconv)
                elif subtitle_edit is not None:
                    logger.info(
                        "PGS fast path via SubtitleEdit CLI: %s (headless OCR needs seconv.exe — "
                        "otherwise the %0.0fs timeout falls back to the Python pipeline)",
                        subtitle_edit,
                        self._config.extraction.subtitle_edit_timeout,
                    )
            self._subtitle_extractor = SubtitleExtractor(
                mkvextract_path=mkvextract,
                ffmpeg_path=ffmpeg,
                ocr=ocr,
                work_dir=self._config.subtitle_cache_dir,
                subtitle_edit_path=subtitle_edit,
                seconv_path=seconv,
                subtitle_edit_timeout=self._config.extraction.subtitle_edit_timeout,
                seconv_ocr_engine=self._config.extraction.seconv_ocr_engine,
                seconv_ocr_db=self._config.extraction.seconv_ocr_db or None,
            )
        return self._subtitle_extractor

    def _audio_service(self) -> AudioSnipper:
        if self._audio_snipper is None:
            ffmpeg = self._resolver.tool_path("ffmpeg")
            self._audio_snipper = AudioSnipper(
                ffmpeg_path=ffmpeg,
                work_dir=self._config.audio_cache_dir,
                model_size=self._config.extraction.whisper_model,
                language=self._config.extraction.whisper_language,
                # Audit F4: configurable scan windows (config stores minutes).
                window_1_start=self._config.extraction.audio_window_1_start_min * 60.0,
                window_1_end=self._config.extraction.audio_window_1_end_min * 60.0,
                minutes_before_end=self._config.extraction.audio_window_2_minutes_before_end * 60.0,
                window_2_length=self._config.extraction.audio_window_2_length_min * 60.0,
            )
        return self._audio_snipper

    def _visual_service(self) -> FrameScanner:
        if self._frame_scanner is None:
            self._frame_scanner = FrameScanner(
                ffmpeg_path=self._resolver.tool_path("ffmpeg"),
                ffprobe_path=self._resolver.tool_path("ffprobe"),
                ocr=self._ocr_engine(),
                work_dir=self._config.frame_cache_dir,
                scene_threshold=self._config.extraction.scene_change_threshold,
                max_frames_per_window=self._config.extraction.max_frames_per_window,
                title_card_minutes=self._config.extraction.title_card_window_minutes,
            )
        return self._frame_scanner

    def _llm_arbiter(self) -> LlmArbiter | None:
        if self._llm is not None:
            return self._llm
        if not self._config.llm_api_key:
            return None
        settings = LlmSettings(
            api_key=self._config.llm_api_key,
            base_url=self._config.llm_base_url,
            model=self._config.llm_model,
        )
        self._llm = LlmArbiter(settings)
        return self._llm

    # -- helpers -------------------------------------------------------------------

    def _progress(self, stage: PipelineStage, message: str, *, file_name: str | None = None, percent: float | None = None) -> None:
        self._progress_cb(ProgressEvent(stage=stage, message=message, file_name=file_name, percent=percent))

    def _cancelled(self) -> bool:
        return bool(self._cancel_cb())

    def _report(self, report: ScanReport, file_name: str, pillar: str, status: str, detail: str = "") -> None:
        report.pillar_reports.append(PillarReport(file_name=file_name, pillar=pillar, status=status, detail=detail))

    def _probe_file(self, path: Path, report: ScanReport) -> MkvFile | None:
        """Probe one MKV with caching by (path, size, mtime)."""
        try:
            stat = path.stat()
        except OSError as exc:
            self._report(report, path.name, "probe", "failed", str(exc))
            return None
        payload = None
        if self._database is not None:
            payload = self._database.get_mkv_scan(path, stat.st_size, stat.st_mtime)
        if payload is not None:
            try:
                return MkvFile.model_validate(payload)
            except Exception as exc:  # noqa: BLE001 - corrupted cache entries
                logger.debug("MKV scan cache invalid for %s: %s", path, exc)
        try:
            mkv = self._ffprobe_reader().probe(path)
        except EpisodeMatchError as exc:
            self._report(report, path.name, "probe", "failed", str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - external tool crashes must not kill the scan
            logger.exception("Unexpected probe failure for %s", path)
            self._report(report, path.name, "probe", "failed", f"unexpected error: {exc}")
            return None
        if self._database is not None:
            self._database.put_mkv_scan(path, stat.st_size, stat.st_mtime, mkv.model_dump(mode="json"))
        return mkv

    # -- main entry point -------------------------------------------------------------

    def scan_folder(
        self,
        folder: Path | str,
        *,
        series: str,
        season: int,
        options: ScanOptions | None = None,
    ) -> ScanReport:
        """Scan all ``.mkv`` files in ``folder`` and solve the global assignment."""
        options = options or ScanOptions()
        folder = Path(folder)
        if not folder.is_dir():
            raise EpisodeMatchError(f"Folder not found: {folder}")
        files = sorted(folder.glob("*.mkv"))
        if not files:
            raise EpisodeMatchError(f"No .mkv files found in {folder}")

        report = ScanReport(assignment=SeasonAssignment(series=SeriesInfo(name=series), season=season))
        self._progress(PipelineStage.PROBE, f"Probing {len(files)} files…", percent=0.0)

        # Stage 1: probe
        probed: list[tuple[MkvFile, MatchCandidate]] = []
        for index, path in enumerate(files):
            if self._cancelled():
                raise EpisodeMatchError("Scan cancelled by user")
            mkv = self._probe_file(path, report)
            if mkv is None:
                continue
            self._report(report, path.name, "probe", "ok", f"{len(mkv.streams)} streams, {len(mkv.chapters)} chapters")
            probed.append((mkv, MatchCandidate(file_path=path)))
            self._progress(PipelineStage.PROBE, f"Probed {path.name}", file_name=path.name, percent=(index + 1) / len(files) * 100)
        if not probed:
            raise EpisodeMatchError(f"No file could be probed in {folder}")

        # Stage 2: TheDiscDB (fast pillar)
        if options.use_disc_db:
            self._stage_disc_db(probed, report, options, series=series, season=season)

        # Stage 3: season guide
        guide = self._build_season_guide(report, series=series, season=season)
        report.guide = guide
        if guide is None:
            raise ProviderError(
                "TMDB API key is missing or the series could not be found - episode titles are unavailable",
                provider="tmdb",
                hint="Set EPISODEMATCH_TMDB_API_KEY or use a local TheDiscDb clone.",
            )
        report.assignment.series = guide.series

        # Early exit: every file conclusively matched by pillar 1?
        if self._all_conclusive(probed, guide, options):
            self._progress(PipelineStage.SOLVE, "TheDiscDB already conclusive - skipping deep pillars")
        else:
            if options.use_subtitles:
                self._stage_subtitles(probed, guide, report, options)
            if options.use_audio and options.depth == "deep":
                self._stage_audio(probed, report, options)
            if options.use_visual and options.depth == "deep":
                self._stage_visual(probed, guide, report, options)
            if options.use_llm and options.depth == "deep":
                self._stage_llm(probed, guide, report)

        # Stage 8: solve
        self._progress(PipelineStage.SOLVE, "Solving global assignment…")
        episode_numbers = [episode.episode for episode in guide.episodes]
        titles = {episode.episode: episode.title for episode in guide.episodes}
        candidates = [candidate for _mkv, candidate in probed]
        solved = solve_assignment(
            candidates,
            episode_numbers,
            titles,
            season=season,
            min_confidence=self._config.matching.min_assignment_confidence,
        )
        report.assignment.entries = solved.entries
        report.assignment.unassigned = solved.unassigned
        report.candidates = candidates
        self._progress(
            PipelineStage.SOLVE,
            f"Assigned {len(solved.entries)}/{len(candidates)} files (avg confidence {report.assignment.average_confidence:.0%})",
            percent=100.0,
        )
        return report

    # -- pillar implementations -----------------------------------------------------------

    def _stage_disc_db(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        report: ScanReport,
        options: ScanOptions,
        *,
        series: str,
        season: int,
    ) -> None:
        if options.discdb_url:
            self._stage_disc_db_from_url(probed, report, options, url=options.discdb_url, season=season)
            return
        client = self._disc_client(options)
        if client is None:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "disc_db", "skipped", "No TheDiscDb source configured")
            return
        discs: list[ParsedDisc]
        try:
            if isinstance(client, LocalDiscDbImporter):
                matching = client.find_series(series)
                if not matching:
                    for mkv, _candidate in probed:
                        self._report(report, mkv.file_name, "disc_db", "skipped", f"No local series for {series!r}")
                    return
                discs = client.load_all_releases(matching[0])
            else:
                series_folders = client.find_series(series)
                if not series_folders:
                    for mkv, _candidate in probed:
                        self._report(report, mkv.file_name, "disc_db", "skipped", f"No TheDiscDb series for {series!r}")
                    return
                discs = []
                for folder in series_folders[:3]:
                    releases = self._disc_release_folders(client, folder, season)
                    if len(releases) > 6:
                        # Un-narrowed series (20+ releases): loading everything is
                        # slow and breeds cross-season noise — cap it.
                        logger.info(
                            "TheDiscDb: %d releases for %r, loading first 6 (pass a disc URL for precision)",
                            len(releases),
                            folder,
                        )
                        releases = releases[:6]
                    for release in releases:
                        discs.extend(client.load_release_discs(folder, release))
        except EpisodeMatchError as exc:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "disc_db", "failed", str(exc))
            return
        self._apply_disc_matches(probed, report, discs, source="thediscdb", season=season)

    def _stage_disc_db_from_url(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        report: ScanReport,
        options: ScanOptions,
        *,
        url: str,
        season: int,
    ) -> None:
        """Load exactly the disc/release a pasted thediscdb.com URL points at."""
        from episodematch.providers.thediscdb import resolve_discs_from_url

        client = self._disc_client(options)
        if client is None:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "disc_db", "skipped", "No TheDiscDb source configured")
            return
        self._progress(PipelineStage.FAST, "Resolving TheDiscDB URL…")
        try:
            discs, meta = resolve_discs_from_url(client, url)
        except EpisodeMatchError as exc:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "disc_db", "failed", str(exc))
            return
        logger.info(
            "TheDiscDB URL resolved: %s / %s (%d disc(s), disc=%r)",
            meta["series_folder"],
            meta["release_folder"],
            len(discs),
            meta["disc_slug"],
        )
        # With the disc pinned by URL, the MakeMKV title index in the file name
        # (…_t07.mkv) identifies the title near-conclusively.
        self._apply_disc_matches(probed, report, discs, source=url, index_weight=0.90, season=season)

    def _apply_disc_matches(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        report: ScanReport,
        discs: list[ParsedDisc],
        *,
        source: str,
        index_weight: float = 0.25,
        season: int,
    ) -> None:
        """Score every file against the loaded disc entries and record evidence."""
        if not discs:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "disc_db", "skipped", "No disc entries loaded")
            return
        tolerance = self._config.matching.discdb_duration_tolerance_seconds
        for mkv, candidate in probed:
            fingerprint = DiscFingerprint.from_mkv(mkv)
            matches = match_fingerprint(
                fingerprint,
                discs,
                duration_tolerance=tolerance,
                index_weight=index_weight,
            )
            if not matches or matches[0].episode is None:
                self._report(report, mkv.file_name, "disc_db", "ok", "No disc fingerprint match")
                continue
            best = matches[0]
            # Audit A1: evidence for a *different* season must never become
            # evidence for the requested season (S02E01 rip was assigned as
            # S01E01 @ 1.0). Extras with unknown season pass through untouched.
            if best.season is not None and best.season != season:
                self._report(
                    report,
                    mkv.file_name,
                    "disc_db",
                    "ok",
                    f"Disc fingerprint matches Season {best.season} — ignored (requested season {season})",
                )
                continue
            # Gate: a duration/size-only coincidence across many releases (often
            # a different season) must not become evidence — require a real
            # identifier (MPLS or index match) or a solid score.
            has_identifier = any(
                "MPLS" in reason or "title index" in reason for reason in best.reasons
            )
            if best.score < 0.45 and not has_identifier:
                self._report(
                    report,
                    mkv.file_name,
                    "disc_db",
                    "ok",
                    f"Weak fingerprint match ignored ({best.score:.0%} vs {best.disc.disc_name})",
                )
                continue
            candidate.add_evidence(
                Evidence(
                    pillar=Pillar.DISC_DB,
                    kind="fingerprint_match",
                    score=best.score,
                    summary=(
                        f"Disc fingerprint matches {best.disc.disc_name} title "
                        f"(MPLS {best.entry.mpls_name or '?'}) -> S{best.season:02d}E{best.episode:02d} "
                        f"{best.entry.episode_title or ''}"
                    ),
                    details={
                        "reasons": best.reasons,
                        "source": best.disc.source if source == "thediscdb" else source,
                        "disc": best.disc.disc_name,
                    },
                    episode_hint=best.episode,
                    season_hint=best.season,
                )
            )
            self._report(report, mkv.file_name, "disc_db", "ok", f"{best.score:.0%} via {best.reasons[:2]}")

    @staticmethod
    def _disc_release_folders(client: TheDiscDbClient | LocalDiscDbImporter, series_folder: str, season: int) -> list[str]:
        """Find release folders that likely contain the wanted season."""
        folders = client.list_releases(series_folder)
        # Prefer folders explicitly mentioning the season number — including
        # combined boxes like "seasons-6-10-blu-ray", which (audit F5) must
        # match every season inside its range, not only its first season.
        exact = re.compile(rf"seasons?[-_ ]?0*{season}\b", re.IGNORECASE)
        season_range = re.compile(
            r"seasons?[-_ ]?(\d{1,2})[-_ ]?(?:to|through|[-_ ])[-_ ]?(\d{1,2})",
            re.IGNORECASE,
        )

        def matches(name: str) -> bool:
            lowered = name.lower()
            if exact.search(lowered):
                return True
            return any(int(lo) <= season <= int(hi) for lo, hi in season_range.findall(lowered))

        preferred = [name for name in folders if matches(name)]
        return preferred or folders

    def _build_season_guide(self, report: ScanReport, *, series: str, season: int) -> SeasonGuide | None:
        tmdb = self._tmdb_client()
        if tmdb is None:
            return None
        try:
            series_info = tmdb.find_series(series)
            guide = tmdb.season_guide(series_info, season)
        except EpisodeMatchError as exc:
            self._report(report, "*", "tmdb", "failed", str(exc))
            return None
        try:
            imdb = self._imdb_client()
            if imdb is not None:
                guide = imdb.enrich_season_guide(guide, imdb.fetch_series_id_for_guide(guide))
        except EpisodeMatchError as exc:
            logger.info("IMDb enrichment skipped: %s", exc)
        return guide

    def _all_conclusive(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        guide: SeasonGuide,
        options: ScanOptions,
    ) -> bool:
        if options.depth == "deep":
            return False
        threshold = self._config.matching.auto_accept_confidence
        if not probed:
            return False
        return all(
            any(
                evidence.pillar is Pillar.DISC_DB and evidence.score >= threshold
                for evidence in candidate.evidence
            )
            for _mkv, candidate in probed
        )

    def _stage_subtitles(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        guide: SeasonGuide,
        report: ScanReport,
        options: ScanOptions,
    ) -> None:
        from episodematch.core.languages import language_candidates, to_osdt_language

        extractor = self._subtitle_service()
        reference_client = self._reference_client()
        provider_name = getattr(reference_client, "SOURCE_NAME", "OpenSubtitles")
        # Reference lookups are cached per (episode, language) — different files
        # with different subtitle languages reuse the same fetches.
        references: dict[tuple[int, str], Transcript | None] = {}

        def reference_for(episode: EpisodeInfo, transcript_language: str | None) -> Transcript | None:
            """Fetch a reference subtitle in the SAME language as the extracted
            track — a German track must be fuzzy-matched against German dialog.

            Known track language -> exactly that language is requested (an English
            fallback would be discarded by the scorer's language guard anyway and
            would only burn download quota). Unknown track language -> the
            preferred language chain, then English.
            """
            if reference_client is None:
                return None
            track_language = to_osdt_language(transcript_language)
            if track_language:
                candidates = [track_language]
            else:
                candidates = language_candidates(
                    None,
                    self._config.preferred_languages,
                    fallback="en",
                )
            logger.info(
                "%s reference language(s) for %s: %s (track language: %r)",
                provider_name,
                episode.code,
                " -> ".join(candidates),
                transcript_language,
            )
            for lang in candidates:
                cache_key = (episode.episode, lang)
                if cache_key in references:
                    cached = references[cache_key]
                else:
                    try:
                        cached = reference_client.reference_transcript(
                            tmdb_id=guide.series.tmdb_id,
                            imdb_id=episode.imdb_id,
                            season=episode.season,
                            episode=episode.episode,
                            languages=lang,
                        )
                    except EpisodeMatchError as exc:
                        logger.warning(
                            "%s reference fetch failed for %s (%s): %s",
                            provider_name,
                            episode.code,
                            lang,
                            exc,
                        )
                        cached = None
                    references[cache_key] = cached
                if cached is not None:
                    return cached
                logger.info(
                    "No usable %s reference on %s for %s — dialog matching "
                    "falls back to plot overlap only for this language.",
                    lang,
                    provider_name,
                    episode.code,
                )
            return None

        for mkv, candidate in probed:
            if self._cancelled():
                raise EpisodeMatchError("Scan cancelled by user")
            try:
                transcripts = extractor.extract_all(
                    mkv,
                    duration_seconds=mkv.duration_seconds,
                    window_minutes=options.subtitle_window_minutes,
                    languages_first=self._config.preferred_languages,
                    ocr_max_events=min(options.ocr_max_events, self._config.extraction.pgs_max_events),
                )
            except EpisodeMatchError as exc:
                self._report(report, mkv.file_name, "subtitle", "failed", str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - tool crash isolation
                logger.exception("Unexpected subtitle extraction failure for %s", mkv.file_name)
                self._report(report, mkv.file_name, "subtitle", "failed", f"unexpected error: {exc}")
                continue
            transcript = self._best_transcript(transcripts, self._config.preferred_languages)
            if transcript is None or not transcript.lines:
                self._report(report, mkv.file_name, "subtitle", "skipped", "No usable subtitle track")
                continue
            self._score_transcript(candidate, transcript, guide, reference_for)
            self._report(
                report,
                mkv.file_name,
                "subtitle",
                "ok",
                f"{len(transcript.lines)} lines via {transcript.source}"
                + (f" [{transcript.language or '?'}]" if transcript.language else ""),
            )

    @staticmethod
    def _best_transcript(
        transcripts: list[Transcript],
        preferred_languages: Sequence[str] | None = None,
    ) -> Transcript | None:
        """Pick the transcript to score with — language first, then source, then volume.

        A long Spanish PGS track must never outrank a shorter German one: the
        winner's language drives the OpenSubtitles reference chain, so preferred
        languages always come first. Native conversions (text tracks and the
        SubtitleEdit CLI) outrank the capped Python OCR fallback.
        """
        if not transcripts:
            return None
        from episodematch.core.languages import to_osdt_language

        order: list[str] = []
        for code in preferred_languages or []:
            mapped = to_osdt_language(code)
            if mapped and mapped not in order:
                order.append(mapped)

        def rank(transcript: Transcript) -> tuple[int, int, int]:
            lang = to_osdt_language(transcript.language)
            lang_rank = order.index(lang) if lang in order else len(order)
            source_rank = 0 if transcript.source in ("srt", "ass", "vtt", "pgs-se-cli") else 1
            return (lang_rank, source_rank, -transcript.character_count)

        return sorted(transcripts, key=rank)[0]

    def _score_transcript(
        self,
        candidate: MatchCandidate,
        transcript: Transcript,
        guide: SeasonGuide,
        reference_for,
    ) -> None:
        best_score = 0.0
        best_episode: int | None = None
        for episode in guide.episodes:
            score = score_episode(
                transcript,
                episode,
                reference_transcript=reference_for(episode, transcript.language),
                dialog_cutoff=self._config.matching.dialog_match_threshold,
            )
            if score.final_score <= 0.0:
                continue
            candidate.add_evidence(
                Evidence(
                    pillar=Pillar.SUBTITLE,
                    kind="text_match",
                    score=score.final_score,
                    summary=f"{episode.code}: {score.summary}",
                    details={
                        "dialog_score": score.dialog_score,
                        "plot_score": score.plot_score,
                        "matched_pairs": score.matched_pairs,
                    },
                    episode_hint=episode.episode,
                )
            )
            if score.final_score > best_score:
                best_score, best_episode = score.final_score, episode.episode
        if best_episode is not None:
            self._progress(
                PipelineStage.SUBTITLES,
                f"{candidate.file_path.name}: best text match E{best_episode:02d} ({best_score:.0%})",
                file_name=candidate.file_path.name,
            )

    def _stage_audio(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        report: ScanReport,
        options: ScanOptions,
    ) -> None:
        try:
            snipper = self._audio_service()
        except EpisodeMatchError as exc:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "audio", "skipped", str(exc))
            return
        for mkv, candidate in probed:
            if self._cancelled():
                raise EpisodeMatchError("Scan cancelled by user")
            try:
                transcript = snipper.transcribe_file(mkv.path, mkv.duration_seconds)
            except EpisodeMatchError as exc:
                self._report(report, mkv.file_name, "audio", "skipped", str(exc))
                continue
            if not transcript.lines:
                self._report(report, mkv.file_name, "audio", "skipped", "Whisper produced no segments")
                continue
            # Audio evidence is keyword overlap against episode plots (TF-IDF only).
            for episode in self._guide_for_audio(report):
                score = score_episode(
                    transcript,
                    episode,
                    reference_transcript=None,
                )
                if score.final_score <= 0.0:
                    continue
                candidate.add_evidence(
                    Evidence(
                        pillar=Pillar.AUDIO,
                        kind="whisper_plot_overlap",
                        score=score.final_score,
                        summary=f"{episode.code}: Whisper transcript overlaps plot ({score.summary})",
                        details={"plot_score": score.plot_score},
                        episode_hint=episode.episode,
                    )
                )
            self._report(report, mkv.file_name, "audio", "ok", f"{len(transcript.lines)} Whisper segments")

    def _guide_for_audio(self, report: ScanReport) -> list[EpisodeInfo]:
        guide = report.guide
        return list(guide.episodes) if guide else []

    def _stage_visual(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        guide: SeasonGuide,
        report: ScanReport,
        options: ScanOptions,
    ) -> None:
        if not self._injected_visual and not self._ocr_engine().available:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "visual", "skipped", "No OCR backend installed")
            return
        try:
            scanner = self._visual_service()
        except EpisodeMatchError as exc:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "visual", "failed", str(exc))
            return
        weights = {
            "director": self._config.matching.credit_weight_director,
            "writer": self._config.matching.credit_weight_writer,
            "guest": self._config.matching.credit_weight_guest,
            "title": self._config.matching.credit_weight_title,
        }
        for mkv, candidate in probed:
            if self._cancelled():
                raise EpisodeMatchError("Scan cancelled by user")
            try:
                hits = scanner.scan(mkv.path, duration_seconds=mkv.duration_seconds)
            except EpisodeMatchError as exc:
                self._report(report, mkv.file_name, "visual", "failed", str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - tool crash isolation
                logger.exception("Unexpected visual scan failure for %s", mkv.file_name)
                self._report(report, mkv.file_name, "visual", "failed", f"unexpected error: {exc}")
                continue
            if not hits:
                self._report(report, mkv.file_name, "visual", "skipped", "No title-card/credits frames found")
                continue
            best_score = 0.0
            best_episode: int | None = None
            for episode in guide.episodes:
                credit_score = score_episode_credits(hits, episode, weights=weights, name_threshold=self._config.matching.name_match_threshold)
                if credit_score.score <= 0.0:
                    continue
                for evidence in credit_evidence(hits, credit_score):
                    candidate.add_evidence(evidence)
                candidate.add_evidence(
                    Evidence(
                        pillar=Pillar.VISUAL,
                        kind="credits_summary",
                        score=credit_score.score,
                        summary=f"{episode.code}: {len(credit_score.matches)} credit matches ({credit_score.score:.0%})",
                        details={"matches": [m.credited_name for m in credit_score.matches]},
                        episode_hint=episode.episode,
                    )
                )
                if credit_score.score > best_score:
                    best_score, best_episode = credit_score.score, episode.episode
            if best_episode is not None:
                self._report(report, mkv.file_name, "visual", "ok", f"Best credits match E{best_episode:02d} ({best_score:.0%})")
            else:
                self._report(report, mkv.file_name, "visual", "ok", f"{len(hits)} frames, no credit matches")

    def _stage_llm(
        self,
        probed: list[tuple[MkvFile, MatchCandidate]],
        guide: SeasonGuide,
        report: ScanReport,
    ) -> None:
        arbiter = self._llm_arbiter()
        if arbiter is None:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "llm", "skipped", "No LLM API key configured")
            return
        self._progress(PipelineStage.LLM, "Consulting LLM arbiter…")
        candidates = [candidate for _mkv, candidate in probed]
        try:
            verdict = arbiter.arbitrate(candidates, guide.episodes, season=guide.season)
        except EpisodeMatchError as exc:
            for mkv, _candidate in probed:
                self._report(report, mkv.file_name, "llm", "failed", str(exc))
            return
        merge_verdict_into_candidates(
            verdict,
            candidates,
            boost=self._config.matching.llm_boost,
        )
        for mkv, _candidate in probed:
            self._report(report, mkv.file_name, "llm", "ok", f"Arbiter verdict merged ({verdict.model})")


def compute_file_hash_for_osdt(mkv_path: Path) -> tuple[str, int]:
    """OpenSubtitles hash + size for a ripped file (used by tests and CLI)."""
    size = mkv_path.stat().st_size
    return compute_opensubtitles_hash(mkv_path), size
