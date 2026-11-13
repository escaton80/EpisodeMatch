"""Matcher workspace: series search, MKV list, live confidence bars, evidence inspector."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from episodematch.core.exceptions import EpisodeMatchError
from episodematch.core.models import AssignmentEntry
from episodematch.matcher.pipeline import ScanReport
from episodematch.ui.context import AppContext
from episodematch.ui.theme import Icon, Palette
from episodematch.ui.widgets.common import (
    AccentButton,
    Banner,
    Card,
    CaptionLabel,
    DangerButton,
    SubtleButton,
    SubtitleLabel,
    TitleLabel,
)
from episodematch.ui.widgets.evidence_card import EvidenceList
from episodematch.ui.workers import RenameWorker, ScanWorker, SeriesSearchWorker, TagWorker

logger = logging.getLogger(__name__)

_COL_FILE = 0
_COL_CODE = 1
_COL_TITLE = 2
_COL_CONF = 3
_COL_METHODS = 4


class WorkspaceView(QWidget):
    """Main view: everything needed to identify and rename one disc folder."""

    scan_finished = Signal(object)  # ScanReport — broadcast to other views
    files_available = Signal(list)  # list[Path] — consumed by the visual verify view

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._scan_worker: ScanWorker | None = None
        self._search_worker: SeriesSearchWorker | None = None
        self._rename_worker: RenameWorker | None = None
        self._tag_worker: TagWorker | None = None
        self._search_results: list = []
        self._selected_series: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(TitleLabel("Matcher Workspace"))
        header.addWidget(SubtitleLabel(
            "Identify every ripped .mkv of a disc folder — five pillars, one confidence score, transparent evidence."
        ))
        root.addLayout(header)

        root.addWidget(self._build_input_card())
        root.addWidget(self._build_action_card())

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)
        self._progress_caption = CaptionLabel("")
        self._progress_caption.setVisible(False)
        root.addWidget(self._progress_caption)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_results_card())
        splitter.addWidget(self._build_inspector_card())
        splitter.setSizes([620, 380])
        root.addWidget(splitter, 1)

        self._banner = Banner("")
        self._banner.hide()
        root.addWidget(self._banner)

    # -- UI construction ---------------------------------------------------------

    def _build_input_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.SEARCH, 14)}  1 · Series, season and folder")
        title.setObjectName("SectionLabel")
        card.add(title)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Series name as known by TMDB, e.g. “Breaking Bad”…")
        self._search_edit.returnPressed.connect(self._on_search)
        self._search_button = SubtleButton("Search TMDB", glyph=Icon.SEARCH)
        self._search_button.clicked.connect(self._on_search)
        self._series_combo = QComboBox()
        self._series_combo.setMinimumWidth(280)
        self._series_combo.setEnabled(False)
        self._series_combo.currentIndexChanged.connect(self._on_series_selected)
        self._season_spin = QSpinBox()
        self._season_spin.setRange(0, 200)
        self._season_spin.setValue(1)
        row1.addWidget(self._search_edit, 2)
        row1.addWidget(self._search_button)
        row1.addWidget(QLabel("Season"))
        row1.addWidget(self._season_spin)
        row1.addWidget(self._series_combo, 2)
        card.add_layout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Folder containing title_tNN.mkv files…")
        browse = SubtleButton("Browse…", glyph=Icon.FOLDER_OPEN)
        browse.clicked.connect(self._on_browse)
        self._depth_combo = QComboBox()
        self._depth_combo.addItems([
            "Fast — DiscDB only",
            "Medium — DiscDB + Subtitles",
            "Deep — all five pillars",
        ])
        self._depth_combo.setCurrentIndex(1)  # Medium: previous default behavior
        self._depth_combo.currentIndexChanged.connect(self._on_depth_changed)
        self._depth_combo.setToolTip(
            "Preset for the pillar checkboxes below:\n"
            "Fast = Disc DB only · Medium = Disc DB + Subtitles · Deep = all five pillars.\n"
            "Individual checkboxes stay adjustable afterwards."
        )
        row2.addWidget(self._folder_edit, 1)
        row2.addWidget(browse)
        row2.addWidget(QLabel("Depth"))
        row2.addWidget(self._depth_combo)
        card.add_layout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        self._discdb_url_edit = QLineEdit()
        self._discdb_url_edit.setPlaceholderText(
            "Optional: TheDiscDB disc URL, e.g. https://thediscdb.com/series/south-park-1997/releases/seasons-6-10-blu-ray/discs/season-06-disc-01"
        )
        row3.addWidget(QLabel("DiscDB URL"))
        row3.addWidget(self._discdb_url_edit, 1)
        card.add_layout(row3)

        pillars = QHBoxLayout()
        pillars.setSpacing(16)
        self._pillar_checks: dict[str, QCheckBox] = {}
        # Pillars still under development — visible but marked honestly.
        NOT_READY_PILLARS = {"use_audio", "use_visual", "use_llm"}
        for key, label in (
            ("use_disc_db", "Disc DB"),
            ("use_subtitles", "Subtitles"),
            ("use_audio", "Audio/Whisper"),
            ("use_visual", "Title-card OCR"),
            ("use_llm", "LLM arbiter"),
        ):
            check = QCheckBox(label)
            if key in NOT_READY_PILLARS:
                check.setText(f"{label} (not ready)")
                check.setToolTip(
                    "This pillar is still under development — treat its results "
                    "as experimental and verify assignments yourself."
                )
                check.setStyleSheet(f"color: {Palette.TEXT_TERTIARY};")
            check.setChecked(key not in ("use_audio", "use_llm"))
            self._pillar_checks[key] = check
            pillars.addWidget(check)
        pillars.addStretch(1)
        card.add_layout(pillars)
        # Apply the Fast preset to the checkboxes so the UI state matches the
        # selected depth right from the start (visual OCR is deep-only).
        self._apply_depth_preset(self._depth_combo.currentIndex())
        return card

    #: Pillar presets per depth combo index (Fast / Medium / Deep).
    _DEPTH_PRESETS: dict[int, dict[str, bool]] = {
        0: {"use_disc_db": True, "use_subtitles": False, "use_audio": False, "use_visual": False, "use_llm": False},
        1: {"use_disc_db": True, "use_subtitles": True, "use_audio": False, "use_visual": False, "use_llm": False},
        2: {"use_disc_db": True, "use_subtitles": True, "use_audio": True, "use_visual": True, "use_llm": True},
    }

    def _on_depth_changed(self, index: int) -> None:
        """The Depth selector applies a pillar preset — switching to Deep ticks
        Audio/Visual/LLM, switching back to Fast unticks them again."""
        self._apply_depth_preset(index)

    def _apply_depth_preset(self, index: int) -> None:
        preset = self._DEPTH_PRESETS.get(index)
        if preset is None:
            return
        for key, checked in preset.items():
            self._pillar_checks[key].setChecked(checked)

    def _build_action_card(self) -> QWidget:
        card = Card(flat=True)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._scan_button = AccentButton("Scan folder", glyph=Icon.PLAY)
        self._scan_button.clicked.connect(self._on_scan)
        self._cancel_button = DangerButton("Cancel scan")
        self._cancel_button.clicked.connect(self._on_cancel_scan)
        self._cancel_button.setEnabled(False)
        self._rename_button = SubtleButton("Apply rename", glyph=Icon.SAVE)
        self._rename_button.clicked.connect(self._on_rename)
        self._rename_button.setEnabled(False)
        self._tag_button = SubtleButton("Write MKV titles", glyph=Icon.SUBTITLE)
        self._tag_button.clicked.connect(self._on_tag)
        self._tag_button.setEnabled(False)
        row.addWidget(self._scan_button)
        row.addWidget(self._cancel_button)
        row.addStretch(1)
        row.addWidget(self._rename_button)
        row.addWidget(self._tag_button)
        card.add_layout(row)
        return card

    def _build_results_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.MOVIE, 14)}  2 · Results")
        title.setObjectName("SectionLabel")
        card.add(title)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["File", "→", "Episode", "Confidence", "Pillars"])
        self._table.horizontalHeader().setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(_COL_CODE, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(_COL_TITLE, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(_COL_CONF, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(_COL_METHODS, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setColumnWidth(_COL_CONF, 130)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        card.add(self._table, stretch=1)
        self._summary_label = CaptionLabel("")
        card.add(self._summary_label)
        return card

    def _build_inspector_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.EYE, 14)}  3 · Evidence inspector")
        title.setObjectName("SectionLabel")
        card.add(title)
        self._inspector_title = CaptionLabel("Why this episode?")
        card.add(self._inspector_title)
        self._evidence_list = EvidenceList()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.addWidget(self._evidence_list)
        scroll.setWidget(host)
        card.add(scroll, stretch=1)
        return card

    # -- search -----------------------------------------------------------------

    def _on_search(self) -> None:
        query = self._search_edit.text().strip()
        if not query:
            self._show_banner("warning", "Enter a series name first.")
            return
        self._search_button.setEnabled(False)
        self._show_banner("info", f"Searching TMDB for “{query}”…")
        self._search_worker = SeriesSearchWorker(self._context.config, query, database=self._context.database)
        self._search_worker.finished_ok.connect(self._on_search_done)
        self._search_worker.failed.connect(self._on_search_failed)
        self._search_worker.start()

    def _on_search_done(self, results: list) -> None:
        self._search_button.setEnabled(True)
        self._search_results = results
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        for info in results:
            year = f" ({info.first_air_year})" if info.first_air_year else ""
            self._series_combo.addItem(f"{info.name}{year}", userData=info)
        self._series_combo.setEnabled(bool(results))
        self._series_combo.blockSignals(False)
        if results:
            self._selected_series = results[0].name
            self._show_banner("success", f"{len(results)} series found — selected “{results[0].name}”.")
        else:
            self._show_banner("warning", "No series found for that query.")

    def _on_search_failed(self, message: str) -> None:
        self._search_button.setEnabled(True)
        self._show_banner("error", message)

    def _on_series_selected(self, index: int) -> None:
        if 0 <= index < len(self._search_results):
            self._selected_series = self._search_results[index].name

    # -- scan --------------------------------------------------------------------

    def _on_browse(self) -> None:
        start = self._folder_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select the ripped folder", start)
        if folder:
            self._folder_edit.setText(folder)

    def _on_scan(self) -> None:
        folder_text = self._folder_edit.text().strip()
        if not folder_text:
            self._show_banner("warning", "Choose a folder containing the ripped .mkv files.")
            return
        if not self._selected_series:
            self._show_banner("warning", "Search and select the series first (pillar TMDB needs it).")
            return
        folder = Path(folder_text)
        options = self._build_scan_options()
        self._set_scan_running(True)
        self._progress.setVisible(True)
        self._progress_caption.setVisible(True)
        self._progress.setValue(0)
        self._show_banner("info", f"Scanning {folder.name} for {self._selected_series} S{self._season_spin.value():02d}…")
        self._scan_worker = ScanWorker(
            self._context.config,
            folder,
            series=self._selected_series,
            season=self._season_spin.value(),
            options=options,
            database=self._context.database,
        )
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.pillar_report.connect(self._on_pillar_report)
        self._scan_worker.finished_ok.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _build_scan_options(self):
        from episodematch.matcher.pipeline import ScanOptions

        checks = self._pillar_checks
        return ScanOptions(
            # Only the Deep preset enables the deep-only pillars (audio/visual/LLM
            # are additionally gated by their checkboxes); Fast/Medium allow the
            # early exit when the DiscDB evidence is already conclusive.
            depth="deep" if self._depth_combo.currentIndex() == 2 else "fast",
            use_disc_db=checks["use_disc_db"].isChecked(),
            use_subtitles=checks["use_subtitles"].isChecked(),
            use_audio=checks["use_audio"].isChecked(),
            use_visual=checks["use_visual"].isChecked(),
            use_llm=checks["use_llm"].isChecked(),
            discdb_url=self._discdb_url_edit.text().strip() or None,
        )

    def _set_scan_running(self, running: bool) -> None:
        self._scan_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._search_button.setEnabled(not running)

    def _on_scan_progress(self, event) -> None:  # noqa: ANN001 - ProgressEvent
        if event.percent is not None:
            self._progress.setValue(int(event.percent))
        self._progress_caption.setText(f"{event.stage.value}: {event.message}")

    def _on_pillar_report(self, file_name: str, pillar: str, status: str, detail: str) -> None:
        logger.debug("pillar %s/%s: %s (%s)", file_name, pillar, status, detail)

    def _on_scan_finished(self, report: object) -> None:
        scan_report: ScanReport = report  # type: ignore[assignment]
        self._set_scan_running(False)
        self._progress.setValue(100)
        assignment = scan_report.assignment
        self._context.last_report = scan_report
        self._context.last_folder = Path(self._folder_edit.text().strip())
        self._fill_results(scan_report)
        self._rename_button.setEnabled(bool(assignment.entries))
        self._tag_button.setEnabled(bool(assignment.entries))
        status = (
            f"{assignment.assigned_count}/{len(scan_report.candidates)} assigned, "
            f"average confidence {assignment.average_confidence:.0%}"
        )
        if assignment.unassigned:
            self._show_banner("warning", f"{status} — {len(assignment.unassigned)} file(s) remain unassigned.")
        else:
            self._show_banner("success", status)
        self.scan_finished.emit(scan_report)
        self.files_available.emit([candidate.file_path for candidate in scan_report.candidates])

    def _on_scan_failed(self, message: str) -> None:
        self._set_scan_running(False)
        self._show_banner("error", message)

    def _on_cancel_scan(self) -> None:
        if self._scan_worker is not None:
            self._scan_worker.cancel()
            self._show_banner("info", "Cancelling after the current file…")

    # -- results ----------------------------------------------------------------------

    def _fill_results(self, report: ScanReport) -> None:
        assignment = report.assignment
        rows: list[tuple[Path, AssignmentEntry | None]] = [
            (entry.file_path, entry) for entry in assignment.entries
        ]
        for unassigned in assignment.unassigned:
            rows.append((unassigned.file_path, None))
        rows.sort(key=lambda item: item[0].name)

        self._table.setRowCount(len(rows))
        for row_index, (path, entry) in enumerate(rows):
            file_item = QTableWidgetItem(path.name)
            file_item.setData(Qt.ItemDataRole.UserRole, path)
            if entry is None:
                code = QTableWidgetItem("?")
                title = QTableWidgetItem("unassigned")
                title.setForeground(Qt.GlobalColor.gray)
                conf_item = QTableWidgetItem()
                methods = QTableWidgetItem("")
                self._table.setItem(row_index, _COL_FILE, file_item)
                self._table.setItem(row_index, _COL_CODE, code)
                self._table.setItem(row_index, _COL_TITLE, title)
                self._table.setItem(row_index, _COL_CONF, conf_item)
                self._table.setItem(row_index, _COL_METHODS, methods)
            else:
                code = QTableWidgetItem(entry.code)
                code.setForeground(Qt.GlobalColor.cyan)
                title = QTableWidgetItem(entry.episode_title or "")
                conf_item = QTableWidgetItem()
                conf_item.setData(Qt.ItemDataRole.DisplayRole, f"{entry.confidence:.0%}")
                conf_item.setData(Qt.ItemDataRole.UserRole, entry.confidence)
                methods = QTableWidgetItem(", ".join(entry.methods))
                self._table.setItem(row_index, _COL_FILE, file_item)
                self._table.setItem(row_index, _COL_CODE, code)
                self._table.setItem(row_index, _COL_TITLE, title)
                self._table.setItem(row_index, _COL_CONF, conf_item)
                self._table.setItem(row_index, _COL_METHODS, methods)

        counts = f"{assignment.assigned_count} assigned"
        if assignment.unassigned:
            counts += f", {len(assignment.unassigned)} unassigned"
        self._summary_label.setText(f"{counts} · average confidence {assignment.average_confidence:.0%}")

    def _on_selection_changed(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        file_item = self._table.item(row, _COL_FILE)
        if file_item is None:
            return
        path: Path = file_item.data(Qt.ItemDataRole.UserRole)
        report = self._context.last_report
        if report is None:
            return
        candidates = {c.file_path: c for c in report.candidates}
        candidate = candidates.get(path)
        entry = next((e for e in report.assignment.entries if e.file_path == path), None)
        if entry is not None:
            heading = f"{entry.code} - {entry.episode_title or ''} · {path.name}"
            self._inspector_title.setText(heading)
            self._evidence_list.set_evidence(entry.evidence)
        elif candidate is not None:
            self._inspector_title.setText(path.name)
            self._evidence_list.set_evidence(candidate.evidence)
        else:
            self._inspector_title.setText(path.name)
            self._evidence_list.set_evidence([])

    # -- post actions -------------------------------------------------------------------

    def _on_rename(self) -> None:
        assignment = self._context.assignment
        if not assignment or not assignment.entries:
            return
        self._rename_button.setEnabled(False)
        self._show_banner("info", f"Renaming {len(assignment.entries)} files (undo journal written)…")
        self._rename_worker = RenameWorker(self._context.config, assignment)
        self._rename_worker.finished_ok.connect(self._on_rename_done)
        self._rename_worker.failed.connect(self._on_rename_failed)
        self._rename_worker.start()

    def _on_rename_done(self, count: int) -> None:
        self._rename_button.setEnabled(True)
        self._show_banner("success", f"Renamed {count} files. Undo via journal in {self._context.config.undo_dir}.")

    def _on_rename_failed(self, message: str) -> None:
        self._rename_button.setEnabled(True)
        self._show_banner("error", message)

    def _on_tag(self) -> None:
        assignment = self._context.assignment
        propedit = self._context.tool_path("mkvpropedit")
        if not assignment or not assignment.entries:
            return
        if propedit is None:
            self._show_banner("warning", "mkvpropedit not found — run 'episodematch fetch-tools' first.")
            return
        self._tag_button.setEnabled(False)
        self._tag_worker = TagWorker(self._context.config, assignment, tool_path=propedit)
        self._tag_worker.finished_ok.connect(self._on_tag_done)
        self._tag_worker.failed.connect(self._on_tag_failed)
        self._tag_worker.start()

    def _on_tag_done(self, count: int) -> None:
        self._tag_button.setEnabled(True)
        self._show_banner("success", f"Wrote titles into {count} MKV headers.")

    def _on_tag_failed(self, message: str) -> None:
        self._tag_button.setEnabled(True)
        self._show_banner("error", message)

    # -- banner -------------------------------------------------------------------------

    def _show_banner(self, level: str, text: str) -> None:
        self._banner.set_level(level, text)
        self._banner.show()

    def prefill(
        self,
        folder: Path | None = None,
        series: str = "",
        season: int = 1,
        *,
        discdb_url: str = "",
    ) -> None:
        """Pre-fill inputs (used when launched with CLI arguments)."""
        if folder is not None:
            self._folder_edit.setText(str(folder))
        if series:
            self._search_edit.setText(series)
            self._selected_series = series
        self._season_spin.setValue(season)
        if discdb_url:
            self._discdb_url_edit.setText(discdb_url)

    def shutdown(self) -> None:
        for worker in (self._scan_worker, self._search_worker, self._rename_worker, self._tag_worker):
            if worker is not None and worker.isRunning():
                worker.cancel() if hasattr(worker, "cancel") else None
                worker.wait(1500)

    def is_busy(self) -> bool:
        """True while any worker thread is still alive (audit D2 handshake)."""
        workers = (self._scan_worker, self._search_worker, self._rename_worker, self._tag_worker)
        return any(worker is not None and worker.isRunning() for worker in workers)
