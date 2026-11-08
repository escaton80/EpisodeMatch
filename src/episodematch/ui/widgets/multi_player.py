"""Synchronized multi-file frame scrubber (visual verification across a disc).

One master time slider drives a grid of per-file frame tiles; every move
(debounced) grabs the frame at that timestamp for *all* files simultaneously via
a single sequential ffmpeg worker, so a user can visually confirm an assignment
("do all these files really show the same scene at minute 12?").
"""

from __future__ import annotations

import logging
import queue
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from episodematch.ui.theme import Icon, Palette
from episodematch.ui.widgets.common import CaptionLabel, SubtleButton

logger = logging.getLogger(__name__)

_THUMB_WIDTH = 320
_DEBOUNCE_MS = 280


class FrameGrabber(QThread):
    """Sequential ffmpeg frame extractor with an internal job queue."""

    frame_ready = Signal(int, float, str)  # tile_index, timestamp, image_path
    grab_failed = Signal(int, float, str)  # tile_index, timestamp, error

    def __init__(self, ffmpeg_path: Path | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ffmpeg = ffmpeg_path
        self._jobs: "queue.Queue[_Job | None]" = queue.Queue()
        self._work_dir = Path.cwd() / "data" / "frames" / "scrubber"
        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self._ffmpeg is not None and Path(self._ffmpeg).is_file()

    def enqueue(self, tile_index: int, video_path: Path, timestamp: float) -> None:
        if not self.available:
            self.grab_failed.emit(tile_index, timestamp, "ffmpeg not found — run 'episodematch fetch-tools'")
            return
        self._jobs.put(_Job(tile_index=tile_index, video=video_path, timestamp=timestamp))

    def shutdown(self) -> None:
        self._jobs.put(None)

    def run(self) -> None:  # noqa: D102 - QThread override
        while True:
            job = self._jobs.get()
            if job is None:
                return
            out_path = self._work_dir / f"grab_{uuid.uuid4().hex[:10]}.png"
            command = [
                str(self._ffmpeg),
                "-y",
                "-v", "error",
                "-ss", f"{max(0.0, job.timestamp):.3f}",
                "-i", str(job.video),
                "-frames:v", "1",
                "-vf", f"scale={_THUMB_WIDTH}:-1",
                str(out_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=60,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.grab_failed.emit(job.tile_index, job.timestamp, str(exc))
                continue
            if completed.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
                self.frame_ready.emit(job.tile_index, job.timestamp, str(out_path))
            else:
                message = (completed.stderr or "frame extraction failed").strip().splitlines()
                self.grab_failed.emit(job.tile_index, job.timestamp, message[-1][:200] if message else "failed")
            # Small breather keeps the UI responsive on huge batches.
            time.sleep(0.02)


@dataclass
class _Job:
    tile_index: int
    video: Path
    timestamp: float


class FrameTile(QWidget):
    """One file's frame thumbnail with name and status caption."""

    def __init__(self, video_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self._timestamp = -1.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.image = QLabel()
        self.image.setFixedSize(_THUMB_WIDTH, int(_THUMB_WIDTH * 9 / 16))
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setObjectName("CardFlat")
        self.image.setText(Icon.styled(Icon.MOVIE, 26, Palette.TEXT_TERTIARY))
        name = CaptionLabel(video_path.name)
        name.setFixedWidth(_THUMB_WIDTH)
        self.status = CaptionLabel("")
        layout.addWidget(self.image)
        layout.addWidget(name)
        layout.addWidget(self.status)

    def set_pending(self) -> None:
        self.status.setText("grabbing frame…")
        self.status.setStyleSheet(f"color: {Palette.TEXT_TERTIARY};")

    def set_frame(self, timestamp: float, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image.setPixmap(
                pixmap.scaled(
                    self.image.width(),
                    self.image.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._timestamp = timestamp
        self.status.setText(f"frame at {timestamp:.1f}s")
        self.status.setStyleSheet(f"color: {Palette.SUCCESS};")
        Path(image_path).unlink(missing_ok=True)

    def set_failed(self, timestamp: float, error: str) -> None:
        self.status.setText(error)
        self.status.setStyleSheet(f"color: {Palette.ERROR};")


class MultiFrameScrubber(QWidget):
    """Master time slider + synchronized frame tiles for all disc files."""

    timestamp_changed = Signal(float)

    def __init__(self, ffmpeg_path: Path | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._files: list[Path] = []
        self._current_timestamp = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 10_000)
        self._slider.setSliderPosition(0)
        self._time_label = CaptionLabel("00:00 / --:--")
        self._time_label.setMinimumWidth(140)
        self._grab_button = SubtleButton(f"  {Icon.REFRESH}  Grab frames")
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time_label)
        controls.addWidget(self._grab_button)
        layout.addLayout(controls)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(14)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_host)
        layout.addWidget(scroll, 1)

        self._grabber = FrameGrabber(ffmpeg_path, self)
        self._grabber.frame_ready.connect(self._on_frame_ready)
        self._grabber.grab_failed.connect(self._on_grab_failed)
        self._grabber.start()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._grab_all)

        self._slider.valueChanged.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._debounce.start)
        self._grab_button.clicked.connect(lambda: self._grab_all())

    # -- public API -----------------------------------------------------------

    def set_files(self, files: list[Path], *, duration_seconds: float | None = None) -> None:
        """Replace the file grid (keeps the grabber thread alive)."""
        self._files = list(files)
        self._duration = duration_seconds or 0.0
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(1, int(self._duration * 10)))
        self._slider.blockSignals(False)
        self._update_time_label(self.current_timestamp())
        self._rebuild_grid()

    def current_timestamp(self) -> float:
        return self._slider.value() / 10.0

    def set_timestamp(self, seconds: float) -> None:
        """Jump to a media time (seconds) and grab all frames there."""
        clamped = max(0.0, seconds)
        self._slider.blockSignals(True)
        if self._duration > 0:
            clamped = min(clamped, self._duration)
        self._slider.setValue(int(clamped * 10))
        self._slider.blockSignals(False)
        self._update_time_label(clamped)
        self._grab_all()

    def shutdown(self) -> None:
        self._debounce.stop()
        self._grabber.shutdown()
        self._grabber.wait(2000)

    # -- internals ---------------------------------------------------------------

    def _rebuild_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        columns = 3
        for index, path in enumerate(self._files):
            tile = FrameTile(path)
            self._grid.addWidget(tile, index // columns, index % columns)
        if not self._files:
            empty = CaptionLabel("No files loaded — run a scan in the workspace first.")
            empty.setStyleSheet(f"color: {Palette.TEXT_TERTIARY}; padding: 24px;")
            self._grid.addWidget(empty, 0, 0)

    def _on_slider_moved(self, value: int) -> None:
        timestamp = value / 10.0
        self._update_time_label(timestamp)
        self.timestamp_changed.emit(timestamp)
        if not self._slider.isSliderDown():
            self._debounce.start()

    def _update_time_label(self, timestamp: float) -> None:
        total = self._duration or 0.0

        def fmt(seconds: float) -> str:
            minutes, secs = divmod(int(seconds), 60)
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

        self._time_label.setText(f"{fmt(timestamp)} / {fmt(total) if total else '--:--'}")

    def _grab_all(self) -> None:
        timestamp = self.current_timestamp()
        self._current_timestamp = timestamp
        for index in range(self._grid.count()):
            widget = self._grid.itemAt(index).widget()
            if isinstance(widget, FrameTile):
                widget.set_pending()
                self._grabber.enqueue(index, widget.video_path, timestamp)

    def _on_frame_ready(self, tile_index: int, timestamp: float, image_path: str) -> None:
        if abs(timestamp - self._current_timestamp) > 0.01:
            Path(image_path).unlink(missing_ok=True)
            return  # stale result from a previous position
        widget = self._grid.itemAt(tile_index).widget() if tile_index < self._grid.count() else None
        if isinstance(widget, FrameTile):
            widget.set_frame(timestamp, image_path)

    def _on_grab_failed(self, tile_index: int, timestamp: float, error: str) -> None:
        widget = self._grid.itemAt(tile_index).widget() if tile_index < self._grid.count() else None
        if isinstance(widget, FrameTile):
            widget.set_failed(timestamp, error)
