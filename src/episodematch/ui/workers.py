"""QThread workers that run the blocking backend services off the UI thread.

Each worker owns its signals (progress / finished / failed), supports
cancellation via the pipeline's cancel callback, and never touches widgets
directly — results are delivered to the UI through Qt signal queues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import httpx
from PySide6.QtCore import QThread, Signal

from episodematch.core.config import AppConfig
from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import EpisodeMatchError
from episodematch.core.models import ProgressEvent, SeasonAssignment, ToolInfo
from episodematch.matcher.pipeline import Pipeline, ScanOptions, ScanReport

logger = logging.getLogger(__name__)


class SeriesSearchWorker(QThread):
    """Searches TMDB for series (graceful when no API key is configured)."""

    finished_ok = Signal(list)  # list[SeriesInfo]
    failed = Signal(str)

    def __init__(self, config: AppConfig, query: str, *, database: CacheDatabase | None = None, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._query = query
        self._database = database

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.providers.tmdb_client import TmdbClient

        if not self._config.tmdb_api_key:
            self.failed.emit(
                "No TMDB API key configured — set it in Settings (free account at themoviedb.org)."
            )
            return
        try:
            client = TmdbClient(self._config.tmdb_api_key, self._database)
            try:
                results = client.search_series(self._query)
            finally:
                client.close()
        except EpisodeMatchError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Series search crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_ok.emit(results)


class ScanWorker(QThread):
    """Runs :meth:`Pipeline.scan_folder` in a worker thread."""

    progress = Signal(ProgressEvent)
    finished_ok = Signal(object)  # ScanReport
    failed = Signal(str)
    pillar_report = Signal(str, str, str, str)  # file, pillar, status, detail

    def __init__(
        self,
        config: AppConfig,
        folder: Path,
        *,
        series: str,
        season: int,
        options: ScanOptions,
        database: CacheDatabase | None,
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._folder = folder
        self._series = series
        self._season = season
        self._options = options
        self._database = database
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: D102 - QThread override
        def on_progress(event: ProgressEvent) -> None:
            self.progress.emit(event)

        def on_report(file_name: str, pillar: str, status: str, detail: str) -> None:
            self.pillar_report.emit(file_name, pillar, status, detail)

        class _ReportingPipeline(Pipeline):
            """Pipeline subclass that mirrors pillar reports into a signal."""

            def _report(self, report: ScanReport, file_name: str, pillar: str, status: str, detail: str = "") -> None:  # noqa: D102
                super()._report(report, file_name, pillar, status, detail)
                on_report(file_name, pillar, status, detail)

        try:
            pipeline = _ReportingPipeline(
                self._config,
                database=self._database,
                progress=on_progress,
                cancel=lambda: self._cancelled,
            )
            report = pipeline.scan_folder(
                self._folder,
                series=self._series,
                season=self._season,
                options=self._options,
            )
        except EpisodeMatchError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - the UI must see every crash
            logger.exception("Scan crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_ok.emit(report)


class LlmTestWorker(QThread):
    """Sends a minimal chat request to the configured LLM endpoint (Settings:
    "Test LLM connection"). Tests the values currently in the form, so the user
    can validate before saving."""

    finished_ok = Signal(str)  # success message incl. model reply
    failed = Signal(str)

    def __init__(self, settings, *, client: httpx.Client | None = None, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._client = client

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.matcher.llm_arbiter import test_connection

        try:
            ok, message = test_connection(self._settings, client=self._client)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM connection test crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        if ok:
            self.finished_ok.emit(message)
        else:
            self.failed.emit(message)


class FetchToolsWorker(QThread):
    """Downloads missing external tools with progress forwarding."""

    progress = Signal(str, int, object)  # label, done, total (None when unknown)
    finished_ok = Signal(tuple)  # installed tool names
    failed = Signal(str)

    def __init__(self, config: AppConfig, *, force: Sequence[str] = (), parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._force = list(force)

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.core.bin_downloader import BinaryDownloader

        try:
            downloader = BinaryDownloader(self._config.resolved_bin_dir)
            result = downloader.download_missing(
                progress=lambda label, done, total: self.progress.emit(label, done, total),
                force=self._force,
            )
        except EpisodeMatchError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool download crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_ok.emit(result.installed)


class RenameWorker(QThread):
    """Applies a rename plan (with undo journal) off the UI thread."""

    finished_ok = Signal(int)  # number of files renamed
    failed = Signal(str)

    def __init__(self, config: AppConfig, assignment: SeasonAssignment, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._assignment = assignment

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.postprocess.renamer import Renamer, build_rename_plan

        try:
            plan = build_rename_plan(
                self._assignment.entries,
                series=self._assignment.series.name,
                template=self._config.naming.episode_template,
                season_folder=self._config.naming.season_folder,
                lowercase_series=self._config.naming.lowercase_series,
                replace_spaces_with_dots=self._config.naming.replace_spaces_with_dots,
                dry_run=False,
            )
            renamer = Renamer(self._config.undo_dir)
            applied = renamer.apply(plan)
        except EpisodeMatchError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rename crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_ok.emit(len(applied))


class TagWorker(QThread):
    """Writes episode titles into MKV headers via mkvpropedit."""

    finished_ok = Signal(int)
    failed = Signal(str)

    def __init__(self, config: AppConfig, assignment: SeasonAssignment, *, tool_path: Path, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._assignment = assignment
        self._tool_path = tool_path

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.postprocess.mkv_tagger import MkvTagger

        tagger = MkvTagger(self._tool_path)
        count = 0
        try:
            for entry in self._assignment.entries:
                tagger.apply_entry(entry)
                count += 1
        except EpisodeMatchError as exc:
            self.failed.emit(f"{exc} ({count} files were already tagged)")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tagging crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_ok.emit(count)
