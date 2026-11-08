"""Shared custom widgets for the EpisodeMatch Fluent UI (all hand-written)."""

from __future__ import annotations

from typing import Callable, Sequence

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from episodematch.ui.theme import Icon, Palette


class Card(QFrame):
    """A rounded Fluent surface with optional vertical content layout."""

    def __init__(self, *, flat: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat" if flat else "Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(10)

    def add(self, widget: QWidget, *, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:  # noqa: ANN001 - QLayout
        self._layout.addLayout(layout)

    def add_stretch(self, stretch: int = 1) -> None:
        self._layout.addStretch(stretch)


class TitleLabel(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("TitleLabel")


class SectionLabel(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionLabel")


class SubtitleLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SubtitleLabel")
        self.setWordWrap(True)


class CaptionLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("CaptionLabel")
        self.setWordWrap(True)


def set_style_class(widget: QWidget, *classes: str) -> None:
    """Set the ``styleClass`` dynamic property (used by the QSS selectors)."""
    widget.setProperty("styleClass", " ".join(classes))
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class AccentButton(QPushButton):
    def __init__(self, text: str, parent: QWidget | None = None, *, glyph: str | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("AccentButton")
        if glyph:
            self.setIcon(Icon.icon(glyph, 14, Palette.ACCENT_TEXT))


class SubtleButton(QPushButton):
    def __init__(self, text: str, parent: QWidget | None = None, *, glyph: str | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SubtleButton")
        if glyph:
            self.setIcon(Icon.icon(glyph, 14, Palette.TEXT_SECONDARY))


class DangerButton(QPushButton):
    def __init__(self, text: str, parent: QWidget | None = None, *, glyph: str | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("DangerButton")
        if glyph:
            self.setIcon(Icon.icon(glyph, 14, Palette.ERROR))


class IconButton(QPushButton):
    """A square Fluent-icon-only button (Segoe Fluent Icons glyph)."""

    def __init__(self, glyph: str, tooltip: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(Icon.icon(glyph, 16))
        self.setToolTip(tooltip)
        self.setFixedSize(34, 32)
        self.setObjectName("SubtleButton")


class NavButton(QPushButton):
    """Sidebar navigation button: icon + label, checkable."""

    def __init__(self, glyph: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(f"  {text}", parent)
        self.setObjectName("NavButton")
        self.setIcon(Icon.icon(glyph, 16, Palette.TEXT_SECONDARY))
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class FieldRow(QHBoxLayout):
    """Horizontal label + field row used across all settings/input views."""

    def __init__(self, label: str, field: QWidget, *, label_width: int = 170) -> None:
        super().__init__()
        caption = QLabel(label)
        caption.setMinimumWidth(label_width)
        caption.setMaximumWidth(label_width)
        caption.setStyleSheet(f"color: {Palette.TEXT_SECONDARY};")
        self.addWidget(caption)
        self.addWidget(field, 1)


class ConfidenceBar(QWidget):
    """Custom-painted rounded confidence bar with percent text (Fluent style)."""

    COLOR_HIGH = Palette.SUCCESS
    COLOR_MEDIUM = Palette.WARNING
    COLOR_LOW = Palette.ERROR

    def __init__(self, value: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self.setFixedHeight(22)
        self.setMinimumWidth(120)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self.update()

    def value(self) -> float:
        return self._value

    def _bar_color(self) -> QColor:
        if self._value >= 0.85:
            return QColor(self.COLOR_HIGH)
        if self._value >= 0.55:
            return QColor(self.COLOR_MEDIUM)
        return QColor(self.COLOR_LOW)

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self.rect().adjusted(0, 6, 0, -6)
        radius = track.height() / 2
        # Track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(Palette.CONTROL))
        painter.drawRoundedRect(track, radius, radius)
        # Fill
        fill_width = int(track.width() * self._value)
        if fill_width > 0:
            fill = track.adjusted(0, 0, -(track.width() - fill_width), 0)
            painter.setBrush(self._bar_color())
            painter.drawRoundedRect(fill, radius, radius)
        # Percent text
        painter.setPen(QPen(QColor(Palette.TEXT_SECONDARY)))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(round(self._value * 100))}%")


class StatusDot(QWidget):
    """Small colored status dot (ok / warn / fail / skip)."""

    SIZE = 10

    def __init__(self, color: str = Palette.SUCCESS, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(self.SIZE + 4, self.SIZE + 4)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(2, 2, self.SIZE, self.SIZE)


class Collapsible(QFrame):
    """Header row with chevron that expands/collapses a content widget."""

    toggled = Signal(bool)

    def __init__(self, header: str, content: QWidget, *, expanded: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat")
        self._content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle = QPushButton(("▾  " if expanded else "▸  ") + header)
        self._toggle.setObjectName("SubtleButton")
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        content.setVisible(expanded)
        layout.addWidget(content)
        self._animation = QPropertyAnimation(content, b"maximumHeight", self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _on_toggle(self, checked: bool) -> None:
        marker = "▾  " if checked else "▸  "
        header_text = self._toggle.text().split("  ", 1)[-1]
        self._toggle.setText(marker + header_text)
        self._content.setVisible(True)
        self._animation.stop()
        self._animation.setStartValue(0 if checked else self._content.sizeHint().height())
        self._animation.setEndValue(self._content.sizeHint().height() if checked else 0)
        self._animation.finished.connect(lambda: self._content.setVisible(checked))
        self._animation.start()
        self.toggled.emit(checked)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()


class Banner(QFrame):
    """Inline info/success/warning/error banner (Fluent style announcement bar)."""

    LEVEL_COLORS = {
        "info": Palette.ACCENT,
        "success": Palette.SUCCESS,
        "warning": Palette.WARNING,
        "error": Palette.ERROR,
    }

    def __init__(self, text: str = "", level: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._label = QLabel(text)
        self._label.setWordWrap(True)
        self._icon = QLabel()
        self._layout.addWidget(self._icon)
        self._layout.addWidget(self._label, 1)
        self.set_level(level, text or "")

    def set_level(self, level: str, text: str | None = None) -> None:
        color = self.LEVEL_COLORS.get(level, Palette.ACCENT)
        glyph = {
            "success": Icon.CHECKMARK,
            "warning": Icon.WARNING,
            "error": Icon.CANCEL,
        }.get(level, Icon.INFO)
        self._icon.setText(Icon.styled(glyph, 14, color))
        if text is not None:
            self._label.setText(text)
        self.setStyleSheet(f"QFrame#CardFlat {{ border-left: 3px solid {color}; }}")


def form_row_grid(rows: Sequence[tuple[str, QWidget]], *, label_width: int = 170) -> QVBoxLayout:
    """Build a vertical layout of :class:`FieldRow` entries."""
    container = QVBoxLayout()
    container.setSpacing(8)
    for label, field in rows:
        row = FieldRow(label, field, label_width=label_width)
        container.addLayout(row)
    return container


def hline() -> QFrame:
    """Thin horizontal separator line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {Palette.CARD_BORDER}; border: none; max-height: 1px;")
    return line


def bind_line_edit(line_edit: QLineEdit, callback: Callable[[str], None]) -> None:
    line_edit.textChanged.connect(callback)
