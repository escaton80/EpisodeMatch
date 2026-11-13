"""Post-processing view: optional stream cleanup and the AV1 encoding queue."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from episodematch.core.config import AppConfig
from episodematch.core.exceptions import EpisodeMatchError
from episodematch.core.models import MkvFile
from episodematch.extractors.ffprobe_reader import FfprobeReader
from episodematch.postprocess.av1_encoder import Av1EncoderQueue, EncoderJob, parse_progress_output
from episodematch.ui.context import AppContext
from episodematch.ui.theme import Icon
from episodematch.ui.widgets.common import (
    AccentButton,
    Banner,
    CaptionLabel,
    Card,
    DangerButton,
    SubtleButton,
    SubtitleLabel,
    TitleLabel,
)

logger = logging.getLogger(__name__)


class AnalyzeWorker(QThread):
    """Probes a folder and builds cleanup plans off the UI thread."""

    finished_ok = Signal(list, list)  # list[MkvFile], list[CleanupPlan]
    failed = Signal(str)

    def __init__(self, context: AppContext, folder: Path, *, forced_only: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._folder = folder
        self._forced_only = forced_only

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.postprocess.stream_cleaner import analyze_subtitle_streams, build_cleanup_plan

        mkvmerge = self._context.tool_path("mkvmerge")
        ffprobe = self._context.tool_path("ffprobe")
        if ffprobe is None:
            self.failed.emit("ffprobe not found — run 'episodematch fetch-tools'.")
            return
        files: list[MkvFile] = []
        plans: list = []
        try:
            reader = FfprobeReader(ffprobe)
            for path in sorted(self._folder.glob("*.mkv")):
                mkv = reader.probe(path, with_ebml=False)
                files.append(mkv)
                plans.append(
                    build_cleanup_plan(
                        mkv,
                        preferred_languages=self._context.config.preferred_languages,
                        keep_commentary=self._context.config.keep_commentary,
                        forced_only_subtitles=self._forced_only,
                    )
                )
        except EpisodeMatchError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cleanup analysis crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self._mkvmerge = mkvmerge  # noqa: B010 - stored for the applier below
        self.finished_ok.emit(files, plans)


class CleanWorker(QThread):
    """Applies one cleanup plan (safe remux with duration verification)."""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, mkvmerge_path: Path, plan, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mkvmerge = mkvmerge_path
        self._plan = plan

    def run(self) -> None:  # noqa: D102 - QThread override
        from episodematch.postprocess.stream_cleaner import StreamCleaner

        try:
            new_path = StreamCleaner(self._mkvmerge).apply(self._plan)
        except EpisodeMatchError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cleanup crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.finished_ok.emit(str(new_path))


class EncodeWorker(QThread):
    """Runs the AV1 encoding queue sequentially."""

    job_progress = Signal(str, float)  # source name, ratio
    job_done = Signal(str, bool, str)  # source name, success, detail
    finished_all = Signal(int, int)  # done count, failed count
    failed = Signal(str)

    def __init__(self, config: AppConfig, jobs: list[EncoderJob], *, ffmpeg_path: Path, state_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._jobs = jobs
        self._ffmpeg = ffmpeg_path
        self._state_path = state_path
        self._cancel_requested = False
        self._queue: Av1EncoderQueue | None = None

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._queue is not None:
            self._queue.cancel()

    def run(self) -> None:  # noqa: D102 - QThread override
        self._queue = Av1EncoderQueue(
            self._ffmpeg,
            state_path=self._state_path,
            progress=lambda job, ratio: self.job_progress.emit(job.source.name, ratio),
        )
        # Reuse the caller-provided jobs by seeding the queue state.
        for job in self._jobs:
            self._queue.add_job(job.source, job.destination, preset=job.preset, crf=job.crf)
        if self._cancel_requested:
            self.finished_all.emit(0, 0)
            return
        try:
            results = self._queue.run()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Encoding queue crashed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        done = sum(1 for job in results if job.status.value == "done")
        failed = sum(1 for job in results if job.status.value == "failed")
        for job in results:
            detail = job.error or ""
            self.job_done.emit(job.source.name, job.status.value == "done", detail)
        self.finished_all.emit(done, failed)


class PostprocessView(QWidget):
    """Optional cleanup of unneeded tracks plus the SVT-AV1 batch queue."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._analyze_worker: AnalyzeWorker | None = None
        self._clean_worker: CleanWorker | None = None
        self._encode_worker: EncodeWorker | None = None
        self._mkv_files: list[MkvFile] = []
        self._plans: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(TitleLabel("Post-Processing"))
        header.addWidget(SubtitleLabel(
            "Optional second pass: remove unneeded audio/subtitle tracks (forced-subtitle aware) "
            "or queue the finished episodes for SVT-AV1 encoding."
        ))
        root.addLayout(header)

        self._beta_banner = Banner(
            "Not ready yet — this page is still under development. Treat cleanup plans as "
            "experimental and keep your backups.",
            level="warning",
        )
        self._beta_banner.show()
        root.addWidget(self._beta_banner)

        root.addWidget(self._build_cleanup_card())
        self._banner = Banner("")
        self._banner.hide()
        root.addWidget(self._banner)

        root.addWidget(self._build_queue_card(), 1)

    # -- cleanup ------------------------------------------------------------------

    def _build_cleanup_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.REPAIR, 14)}  1 · Track cleanup")
        title.setObjectName("SectionLabel")
        card.add(title)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Folder with the finished episodes…")
        browse = SubtleButton("Browse…", glyph=Icon.FOLDER_OPEN)
        browse.clicked.connect(self._on_browse)
        self._analyze_button = SubtleButton("Analyze tracks", glyph=Icon.SEARCH)
        self._analyze_button.clicked.connect(self._on_analyze)
        self._clean_button = AccentButton("Remove tracks")
        self._clean_button.clicked.connect(self._on_clean)
        self._clean_button.setEnabled(False)
        row.addWidget(self._folder_edit, 1)
        row.addWidget(browse)
        row.addWidget(self._analyze_button)
        row.addWidget(self._clean_button)
        card.add_layout(row)

        self._cleanup_table = QTableWidget(0, 4)
        self._cleanup_table.setHorizontalHeaderLabels(["File", "Keep audio", "Keep subs", "Action"])
        self._cleanup_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._cleanup_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._cleanup_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._cleanup_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._cleanup_table.verticalHeader().setVisible(False)
        self._cleanup_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._cleanup_table.setMinimumHeight(180)
        card.add(self._cleanup_table, stretch=1)

        self._forced_only = QPushButton("Forced-only subtitles")
        self._forced_only.setCheckable(True)
        self._forced_only.setToolTip("Keep only tracks flagged/behaving as forced (event coverage ≤ 15%).")
        row2 = QHBoxLayout()
        row2.addWidget(CaptionLabel(
            "Analysis reads stream layouts; cleanup remuxes losslessly via mkvmerge with duration verification (±1 s) and keeps a .bak."
        ))
        row2.addStretch(1)
        row2.addWidget(self._forced_only)
        card.add_layout(row2)
        return card

    def _on_browse(self) -> None:
        start = self._folder_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select the episode folder", start)
        if folder:
            self._folder_edit.setText(folder)

    def prefill_folder(self, folder: Path | None) -> None:
        if folder is not None:
            self._folder_edit.setText(str(folder))

    def _on_analyze(self) -> None:
        folder_text = self._folder_edit.text().strip()
        folder = Path(folder_text) if folder_text else None
        if folder is None or not folder.is_dir():
            self._show_banner("warning", "Choose a valid folder first.")
            return
        self._analyze_button.setEnabled(False)
        self._show_banner("info", f"Analyzing {folder.name}…")
        self._analyze_worker = AnalyzeWorker(self._context, folder, forced_only=self._forced_only.isChecked())
        self._analyze_worker.finished_ok.connect(self._on_analyze_done)
        self._analyze_worker.failed.connect(self._on_analyze_failed)
        self._analyze_worker.start()

    def _on_analyze_done(self, files: list, plans: list) -> None:
        self._analyze_button.setEnabled(True)
        self._mkv_files = files
        self._plans = plans
        table = self._cleanup_table
        table.setRowCount(len(plans))
        removable = 0
        for row, plan in enumerate(plans):
            table.setItem(row, 0, QTableWidgetItem(plan.file_path.name))
            table.setItem(row, 1, QTableWidgetItem(f"{len(plan.keep_audio_indices)} of {len(plan.keep_audio_indices) + len(plan.remove_audio)}"))
            table.setItem(row, 2, QTableWidgetItem(f"{len(plan.keep_subtitle_indices)} of {len(plan.keep_subtitle_indices) + len(plan.remove_subtitles)}"))
            action = plan.summary() if not plan.is_noop else "nothing to remove"
            table.setItem(row, 3, QTableWidgetItem(action))
            if not plan.is_noop:
                removable += 1
        self._clean_button.setEnabled(removable > 0)
        self._show_banner(
            "success" if removable else "info",
            f"{removable} file(s) with removable tracks found." if removable else "Nothing to remove — all files are already clean.",
        )

    def _on_analyze_failed(self, message: str) -> None:
        self._analyze_button.setEnabled(True)
        self._show_banner("error", message)

    def _on_clean(self) -> None:
        mkvmerge = self._context.tool_path("mkvmerge")
        if mkvmerge is None:
            self._show_banner("warning", "mkvmerge not found — run 'episodematch fetch-tools'.")
            return
        targets = [plan for plan in self._plans if not plan.is_noop]
        if not targets:
            return
        self._clean_button.setEnabled(False)
        self._show_banner("info", f"Remuxing {len(targets)} file(s)…")
        self._clean_all(mkvmerge, targets)

    def _clean_all(self, mkvmerge: Path, targets: list) -> None:
        """Sequential cleanup with per-file worker chaining."""
        if not targets:
            self._clean_button.setEnabled(True)
            self._show_banner("success", "Cleanup finished.")
            return
        plan = targets[0]

        def on_done(new_path: str) -> None:
            self._show_banner("success", f"Cleaned {Path(new_path).name}")
            self._clean_all(mkvmerge, targets[1:])

        def on_failed(message: str) -> None:
            self._show_banner("error", message)
            self._clean_all(mkvmerge, targets[1:])

        self._clean_worker = CleanWorker(mkvmerge, plan)
        self._clean_worker.finished_ok.connect(on_done)
        self._clean_worker.failed.connect(on_failed)
        self._clean_worker.start()

    # -- AV1 queue -------------------------------------------------------------------

    def _build_queue_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.MOVIE, 14)}  2 · AV1 encoding queue (SVT-AV1)")
        title.setObjectName("SectionLabel")
        card.add(title)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._preset_spin = QSpinBox()
        self._preset_spin.setRange(0, 13)
        self._preset_spin.setValue(self._context.config.av1_preset)
        self._crf_spin = QSpinBox()
        self._crf_spin.setRange(0, 63)
        self._crf_spin.setValue(self._context.config.av1_crf)
        self._add_button = SubtleButton("Add folder", glyph=Icon.DOWNLOAD)
        self._add_button.clicked.connect(self._on_add_folder)
        self._start_button = AccentButton("Start encoding", glyph=Icon.PLAY)
        self._start_button.clicked.connect(self._on_start_encode)
        self._stop_button = DangerButton("Stop")
        self._stop_button.clicked.connect(self._on_stop_encode)
        self._stop_button.setEnabled(False)
        row.addWidget(QLabel("Preset"))
        row.addWidget(self._preset_spin)
        row.addWidget(QLabel("CRF"))
        row.addWidget(self._crf_spin)
        row.addStretch(1)
        row.addWidget(self._add_button)
        row.addWidget(self._start_button)
        row.addWidget(self._stop_button)
        card.add_layout(row)

        self._queue_table = QTableWidget(0, 4)
        self._queue_table.setHorizontalHeaderLabels(["Source", "Destination", "Status", "Progress"])
        self._queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._queue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._queue_table.setColumnWidth(3, 220)
        self._queue_table.verticalHeader().setVisible(False)
        self._queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        card.add(self._queue_table, stretch=1)
        return card

    def _on_add_folder(self) -> None:
        folder_text = self._folder_edit.text().strip()
        folder = Path(folder_text) if folder_text else None
        if folder is None or not folder.is_dir():
            folder = QFileDialog.getExistingDirectory(self, "Select the episode folder")
            folder = Path(folder) if folder else None
        if folder is None or not folder.is_dir():
            return
        self._folder_edit.setText(str(folder))
        # Audit B3: skip files whose encoded output already exists — re-queueing
        # them would silently overwrite finished encodes at the final replace().
        count = 0
        skipped = 0
        for path in sorted(folder.glob("*.mkv")):
            if path.stem.endswith(".av1"):
                continue
            destination = path.with_name(path.stem + ".av1.mkv")
            if destination.exists():
                skipped += 1
                continue
            self._append_queue_row(path, destination)
            count += 1
        if skipped:
            self._show_banner("info", f"{count} file(s) queued — {skipped} skipped (output already exists).")
        else:
            self._show_banner("info", f"{count} file(s) queued for AV1 encoding.")

    def _append_queue_row(self, source: Path, destination: Path) -> int:
        row = self._queue_table.rowCount()
        self._queue_table.insertRow(row)
        self._queue_table.setItem(row, 0, QTableWidgetItem(str(source)))
        self._queue_table.setItem(row, 1, QTableWidgetItem(str(destination)))
        self._queue_table.setItem(row, 2, QTableWidgetItem("pending"))
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        self._queue_table.setCellWidget(row, 3, bar)
        return row

    def _find_row(self, source_name: str) -> int:
        for row in range(self._queue_table.rowCount()):
            item = self._queue_table.item(row, 0)
            if item is not None and item.text().endswith(source_name) or (item is not None and item.text() == source_name):
                return row
        return -1

    def _on_start_encode(self) -> None:
        ffmpeg = self._context.tool_path("ffmpeg")
        if ffmpeg is None:
            self._show_banner("warning", "ffmpeg not found — run 'episodematch fetch-tools'.")
            return
        jobs: list[EncoderJob] = []
        for row in range(self._queue_table.rowCount()):
            source_item = self._queue_table.item(row, 0)
            dest_item = self._queue_table.item(row, 1)
            if source_item is None or dest_item is None:
                continue
            jobs.append(
                EncoderJob(
                    source=Path(source_item.text()),
                    destination=Path(dest_item.text()),
                    preset=self._preset_spin.value(),
                    crf=self._crf_spin.value(),
                )
            )
        if not jobs:
            self._show_banner("warning", "Queue is empty — add a folder first.")
            return
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._show_banner("info", f"Encoding {len(jobs)} file(s) with SVT-AV1 preset {self._preset_spin.value()} / CRF {self._crf_spin.value()}…")
        self._encode_worker = EncodeWorker(
            self._context.config,
            jobs,
            ffmpeg_path=ffmpeg,
            state_path=self._context.data_dir / "av1-queue.json",
        )
        self._encode_worker.job_progress.connect(self._on_encode_progress)
        self._encode_worker.job_done.connect(self._on_encode_job_done)
        self._encode_worker.finished_all.connect(self._on_encode_finished)
        self._encode_worker.failed.connect(self._on_encode_failed)
        self._encode_worker.start()

    def _on_encode_progress(self, source_name: str, ratio: float) -> None:
        row = self._find_row(source_name)
        if row < 0:
            return
        bar = self._queue_table.cellWidget(row, 3)
        if isinstance(bar, QProgressBar):
            bar.setValue(int(ratio * 100))
        status_item = self._queue_table.item(row, 2)
        if status_item is not None:
            status_item.setText("encoding")

    def _on_encode_job_done(self, source_name: str, success: bool, detail: str) -> None:
        row = self._find_row(source_name)
        if row < 0:
            return
        status_item = self._queue_table.item(row, 2)
        if status_item is not None:
            status_item.setText("done" if success else "failed")
            status_item.setForeground(Qt.GlobalColor.green if success else Qt.GlobalColor.red)

    def _on_encode_finished(self, done: int, failed: int) -> None:
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._show_banner("success" if failed == 0 else "warning", f"Encoding finished: {done} done, {failed} failed.")

    def _on_encode_failed(self, message: str) -> None:
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._show_banner("error", message)

    def _on_stop_encode(self) -> None:
        if self._encode_worker is not None:
            self._encode_worker.cancel()
            self._show_banner("info", "Stopping after the current file…")

    def _show_banner(self, level: str, text: str) -> None:
        self._banner.set_level(level, text)
        self._banner.show()

    def shutdown(self) -> None:
        for worker in (self._analyze_worker, self._clean_worker, self._encode_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "cancel"):
                    worker.cancel()
                worker.wait(1500)

    def is_busy(self) -> bool:
        """True while any worker thread is still alive (audit D2 handshake)."""
        workers = (self._analyze_worker, self._clean_worker, self._encode_worker)
        return any(worker is not None and worker.isRunning() for worker in workers)
