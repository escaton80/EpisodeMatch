"""Evidence card widget: shows exactly WHY an episode was recognized.

Each :class:`~episodematch.core.models.Evidence` becomes one card with a pillar
badge, a score bar, the human-readable summary and a collapsible details view.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from episodematch.core.models import Evidence
from episodematch.ui.theme import Icon, Palette
from episodematch.ui.widgets.common import CaptionLabel, ConfidenceBar, SubtitleLabel

_PILLAR_ICONS = {
    "metadata": Icon.INFO,
    "disc_db": Icon.MOVIE,
    "subtitle": Icon.SUBTITLE,
    "audio": Icon.MUSIC,
    "visual": Icon.EYE,
    "llm": Icon.ROBOT,
}

_PILLAR_TITLES = {
    "metadata": "Metadata",
    "disc_db": "Disc fingerprint (TheDiscDB)",
    "subtitle": "Subtitles & dialog",
    "audio": "Audio transcription (Whisper)",
    "visual": "Title cards & credits (OCR)",
    "llm": "LLM arbiter",
}


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def pillar_color(pillar: str) -> str:
    return Palette.PILLAR_COLORS.get(pillar.lower(), Palette.TEXT_TERTIARY)


class PillarBadge(QWidget):
    """Colored dot + pillar title row."""

    def __init__(self, pillar: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        color = pillar_color(pillar)
        glyph = _PILLAR_ICONS.get(pillar.lower(), Icon.INFO)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        dot = QLabel(Icon.styled(glyph, 13, color))
        dot.setFixedWidth(20)
        title = QLabel(_PILLAR_TITLES.get(pillar.lower(), pillar.replace("_", " ").title()))
        title.setStyleSheet(f"color: {Palette.TEXT}; font-weight: 600;")
        layout.addWidget(dot)
        layout.addWidget(title, 1)


class EvidenceCard(QFrame):
    """One evidence item as a Fluent card with collapsible details."""

    def __init__(self, evidence: Evidence, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._evidence = evidence
        self.setObjectName("EvidenceCard")
        color = pillar_color(evidence.pillar.value)
        self.setStyleSheet(
            f"QFrame#EvidenceCard {{ border-left: 3px solid {color}; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(10)
        badge = PillarBadge(evidence.pillar.value)
        header.addWidget(badge, 1)
        self._score_bar = ConfidenceBar(evidence.score)
        self._score_bar.setMinimumWidth(90)
        self._score_bar.setMaximumWidth(140)
        header.addWidget(self._score_bar)
        layout.addLayout(header)

        summary_text = evidence.summary or evidence.kind
        if evidence.timestamp_seconds is not None:
            summary_text = f"[{format_timestamp(evidence.timestamp_seconds)}]  {summary_text}"
        summary = SubtitleLabel(summary_text)
        summary.setStyleSheet(f"color: {Palette.TEXT_SECONDARY};")
        layout.addWidget(summary)

        if evidence.details:
            detail_lines = []
            for key, value in evidence.details.items():
                rendered = str(value)
                if isinstance(value, (list, tuple)):
                    rendered = "; ".join(str(item) for item in value[:6])
                detail_lines.append(f"· {key}: {rendered}")
            self._details = CaptionLabel("\n".join(detail_lines))
            self._details.setVisible(False)
            self._details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(self._details)
            self._more = QLabel("  ▸  details")
            self._more.setStyleSheet(f"color: {Palette.TEXT_TERTIARY}; font-size: 11px;")
            self._more.setCursor(Qt.CursorShape.PointingHandCursor)
            more_layout = QHBoxLayout()
            more_layout.addWidget(self._more)
            more_layout.addStretch(1)
            layout.addLayout(more_layout)

            def _toggle(_checked: bool = False, card: EvidenceCard = self) -> None:
                visible = not card._details.isVisible()  # noqa: SLF001 - intentional UI toggle
                card._details.setVisible(visible)
                marker = "▾" if visible else "▸"
                card._more.setText(f"  {marker}  details")

            self._more.mousePressEvent = _toggle  # type: ignore[assignment]


class EvidenceList(QWidget):
    """Vertical stack of evidence cards with an optional empty-state label."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._empty = CaptionLabel("Select a file to inspect its evidence.")
        self._empty.setStyleSheet(f"color: {Palette.TEXT_TERTIARY};")
        self._layout.addWidget(self._empty)

    def set_evidence(self, items: Sequence[Evidence], *, title: str | None = None) -> None:
        self.clear()
        if title:
            header = QLabel(title)
            header.setObjectName("SectionLabel")
            self._layout.addWidget(header)
        if not items:
            self._empty.setText("No evidence recorded for this file yet.")
            self._empty.show()
            self._layout.addWidget(self._empty)
            return
        self._empty.hide()
        for evidence in sorted(items, key=lambda item: item.score, reverse=True):
            self._layout.addWidget(EvidenceCard(evidence))
        self._layout.addStretch(1)

    def clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._empty:
                # Detach immediately so findChildren() reflects the cleared state
                # even before Qt processes the deferred deletion.
                widget.setParent(None)
                widget.deleteLater()
        self._empty.hide()
