"""Visual verification: compare frames of all disc files at the same timestamp."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from episodematch.ui.context import AppContext
from episodematch.ui.theme import Icon
from episodematch.ui.widgets.common import (
    AccentButton,
    CaptionLabel,
    Card,
    SubtitleLabel,
    SubtleButton,
    TitleLabel,
)
from episodematch.ui.widgets.multi_player import MultiFrameScrubber


class VisualVerifyView(QWidget):
    """Synchronized frame comparison for every file of the current folder."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._files: list[Path] = []
        self._duration: float | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(TitleLabel("Visual Verify"))
        header.addWidget(SubtitleLabel(
            "Scrub to any timestamp and compare the same frame across all files — "
            "the fastest way to confirm an assignment with your own eyes."
        ))
        root.addLayout(header)

        from episodematch.ui.widgets.common import Banner

        beta_banner = Banner(
            "Not ready yet — this page is still under development. Frame grabbing works, "
            "but treat the workflow as a preview.",
            level="warning",
        )
        beta_banner.show()
        root.addWidget(beta_banner)

        card = Card()
        title = QLabel(f"{Icon.styled(Icon.TIMELINE, 14)}  Files")
        title.setObjectName("SectionLabel")
        card.add(title)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Folder with the ripped .mkv files…")
        browse = SubtleButton("Browse…", glyph=Icon.FOLDER_OPEN)
        browse.clicked.connect(self._on_browse)
        self._load_button = AccentButton("Load files", glyph=Icon.MOVIE)
        self._load_button.clicked.connect(self._on_load)
        row.addWidget(self._folder_edit, 1)
        row.addWidget(browse)
        row.addWidget(self._load_button)
        card.add_layout(row)

        info_row = QHBoxLayout()
        self._duration_spin = QSpinBox()
        self._duration_spin.setRange(1, 24 * 3600)
        self._duration_spin.setValue(42 * 60)
        self._duration_spin.setSuffix(" s")
        info_row.addWidget(QLabel("Approx. episode length"))
        info_row.addWidget(self._duration_spin)
        info_row.addWidget(CaptionLabel("(only for slider scaling — frames are grabbed from the real files)"))
        info_row.addStretch(1)
        card.add_layout(info_row)
        root.addWidget(card)

        self._scrubber = MultiFrameScrubber(self._context.tool_path("ffmpeg"))
        root.addWidget(self._scrubber, 1)

        self._hint = CaptionLabel(
            "Tip: run a workspace scan first — “Load from scan” pulls every matched file with its real duration."
        )
        root.addWidget(self._hint)

    # -- actions -----------------------------------------------------------------

    def _on_browse(self) -> None:
        start = self._folder_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select the ripped folder", start)
        if folder:
            self._folder_edit.setText(folder)

    def _on_load(self) -> None:
        from episodematch.core.models import MkvFile

        folder = Path(self._folder_edit.text().strip())
        if not folder.is_dir():
            self._hint.setText("That folder does not exist.")
            return
        files = sorted(folder.glob("*.mkv"))
        self._apply_files(files, duration_hint=float(self._duration_spin.value()))
        self._hint.setText(f"{len(files)} file(s) loaded from {folder.name}.")

    def load_from_scan(self, report) -> None:  # noqa: ANN001 - ScanReport
        """Populate from the last workspace scan (real durations via report cache)."""
        if report is None:
            return
        files: list[Path] = []
        max_duration = 0.0
        for candidate in report.candidates:
            files.append(candidate.file_path)
        for entry in report.assignment.entries:
            pass  # durations live on candidates only; report carries no probed MkvFile
        self._apply_files(files, duration_hint=max_duration or float(self._duration_spin.value()))
        if files:
            self._folder_edit.setText(str(files[0].parent))
            self._hint.setText(f"{len(files)} file(s) from the last scan loaded.")

    def _apply_files(self, files: list[Path], *, duration_hint: float) -> None:
        self._files = files
        self._duration = duration_hint or None
        self._duration_spin.setValue(int(duration_hint) if duration_hint else self._duration_spin.value())
        self._scrubber.set_files(files, duration_seconds=duration_hint)

    def shutdown(self) -> None:
        self._scrubber.shutdown()

    def is_busy(self) -> bool:
        """The verify view only plays media — it never touches the database."""
        return False
