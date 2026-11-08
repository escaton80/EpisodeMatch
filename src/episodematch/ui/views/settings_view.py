"""Settings view: tool status, API keys, naming scheme builder."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from episodematch.core.binaries import TOOL_NAMES
from episodematch.core.config import format_episode_name
from episodematch.core.exceptions import EpisodeMatchError
from episodematch.ui.context import AppContext
from episodematch.ui.theme import Icon, Palette
from episodematch.ui.widgets.common import (
    AccentButton,
    Banner,
    Card,
    CaptionLabel,
    DangerButton,
    StatusDot,
    SubtleButton,
    SubtitleLabel,
    TitleLabel,
    form_row_grid,
)
from episodematch.ui.workers import FetchToolsWorker, LlmTestWorker

_STATUS_COLORS = {
    "ok": Palette.SUCCESS,
    "missing": Palette.ERROR,
}


class SettingsView(QWidget):
    """Tool status (./bin/), API keys and the naming scheme builder."""

    config_saved = Signal()

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._fetch_worker: FetchToolsWorker | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        root = QVBoxLayout(host)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(14)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(TitleLabel("Settings"))
        header.addWidget(SubtitleLabel("External tools, API keys and the naming scheme — saved to data/config.json."))
        root.addLayout(header)

        root.addWidget(self._build_tools_card())
        root.addWidget(self._build_keys_card())
        root.addWidget(self._build_naming_card())
        root.addWidget(self._build_pillars_card())

        self._save_button = AccentButton("Save settings", glyph=Icon.SAVE)
        self._save_button.clicked.connect(self._on_save)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_row.addWidget(self._save_button)
        root.addLayout(save_row)

        self._banner = Banner("")
        self._banner.hide()
        root.addWidget(self._banner)
        root.addStretch(1)

        scroll.setWidget(host)
        outer.addWidget(scroll)

        self.refresh_tool_status()
        self._load_from_config()

    # -- cards --------------------------------------------------------------------

    def _build_tools_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.REPAIR, 14)}  External tools")
        title.setObjectName("SectionLabel")
        card.add(title)

        self._tool_rows: dict[str, tuple[StatusDot, QLabel, QLabel]] = {}
        for name in TOOL_NAMES:
            row = QHBoxLayout()
            row.setSpacing(10)
            dot = StatusDot()
            name_label = QLabel(name + ".exe")
            name_label.setMinimumWidth(140)
            version_label = QLabel("—")
            path_label = CaptionLabel("not found")
            row.addWidget(dot)
            row.addWidget(name_label)
            row.addWidget(version_label)
            row.addWidget(path_label, 1)
            card.add_layout(row)
            self._tool_rows[name] = (dot, version_label, path_label)

        actions = QHBoxLayout()
        self._fetch_button = AccentButton("Fetch missing tools", glyph=Icon.DOWNLOAD)
        self._fetch_button.clicked.connect(self._on_fetch_tools)
        rescan = SubtleButton("Re-scan", glyph=Icon.REFRESH)
        rescan.clicked.connect(self.refresh_tool_status)
        actions.addWidget(self._fetch_button)
        actions.addWidget(rescan)
        actions.addStretch(1)
        card.add_layout(actions)
        return card

    def _build_keys_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.SETTINGS, 14)}  API keys & LLM endpoint")
        title.setObjectName("SectionLabel")
        card.add(title)

        self._tmdb_edit = QLineEdit()
        self._tmdb_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._osdt_edit = QLineEdit()
        self._osdt_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._subdl_edit = QLineEdit()
        self._subdl_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._source_combo = QComboBox()
        self._source_combo.addItems(["opensubtitles", "subdl"])
        self._source_combo.setToolTip(
            "OpenSubtitles: precise API, free tier = 5 downloads / 24h.\n"
            "SubDL: free tier = 2,000 requests + 300 downloads per day (ZIP archives with SRTs)."
        )
        self._llm_key_edit = QLineEdit()
        self._llm_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_url_edit = QLineEdit()
        self._llm_model_edit = QLineEdit()
        card.add_layout(form_row_grid([
            ("TMDB API key", self._tmdb_edit),
            ("OpenSubtitles API key", self._osdt_edit),
            ("SubDL API key", self._subdl_edit),
            ("Subtitle source", self._source_combo),
            ("LLM API key", self._llm_key_edit),
            ("LLM base URL", self._llm_url_edit),
            ("LLM model", self._llm_model_edit),
        ]))

        llm_test_row = QHBoxLayout()
        self._llm_test_button = SubtleButton("Test LLM connection", glyph=Icon.PLAY)
        self._llm_test_button.clicked.connect(self._on_test_llm)
        llm_test_row.addStretch(1)
        llm_test_row.addWidget(self._llm_test_button)
        card.add_layout(llm_test_row)
        # Inline status right under the button — the global banner sits at the
        # bottom of a long scroll page and would be invisible while clicking here.
        self._llm_status = CaptionLabel("")
        self._llm_status.setWordWrap(True)
        self._llm_status.hide()
        card.add(self._llm_status)
        card.add(CaptionLabel(
            "Tests the values currently in the fields (no save needed). Works with any "
            "OpenAI-compatible endpoint — e.g. BigModel GLM or DeepSeek."
        ))
        card.add(CaptionLabel(
            "Keys are stored in data/config.json (gitignored). Environment variables EPISODEMATCH_* take precedence. "
            "TMDB/OSDT keys are free; the LLM default targets BigModel GLM (glm-4-flash)."
        ))
        return card

    def _build_naming_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.SAVE, 14)}  Naming scheme")
        title.setObjectName("SectionLabel")
        card.add(title)

        self._template_edit = QLineEdit()
        self._template_edit.textChanged.connect(self._update_naming_preview)
        self._season_folder = QPushButton("Season folder")
        self._season_folder.setCheckable(True)
        self._season_folder.setChecked(True)
        self._season_folder.toggled.connect(self._update_naming_preview)
        self._dots_button = QPushButton("Dots for spaces")
        self._dots_button.setCheckable(True)
        self._dots_button.toggled.connect(self._update_naming_preview)
        self._lowercase_button = QPushButton("Lowercase series")
        self._lowercase_button.setCheckable(True)
        self._lowercase_button.toggled.connect(self._update_naming_preview)
        card.add_layout(form_row_grid([("Episode template", self._template_edit)]))

        toggles = QHBoxLayout()
        toggles.addWidget(self._season_folder)
        toggles.addWidget(self._dots_button)
        toggles.addWidget(self._lowercase_button)
        toggles.addStretch(1)
        card.add_layout(toggles)

        preview_caption = CaptionLabel("Preview")
        self._preview_label = QLabel()
        self._preview_label.setStyleSheet(f"color: {Palette.ACCENT};")
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card.add(preview_caption)
        card.add(self._preview_label)
        card.add(CaptionLabel(
            "Placeholders: {series}, {season}, {episode}, {title}, zero-padded {season:02d}/{episode:02d}. "
            "Each path component is sanitized for Windows automatically."
        ))
        return card

    def _build_pillars_card(self) -> QWidget:
        card = Card()
        title = QLabel(f"{Icon.styled(Icon.MOVIE, 14)}  Defaults")
        title.setObjectName("SectionLabel")
        card.add(title)

        self._languages_edit = QLineEdit()
        self._preset_spin = QSpinBox()
        self._preset_spin.setRange(0, 13)
        self._crf_spin = QSpinBox()
        self._crf_spin.setRange(0, 63)
        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(50, 100)
        self._threshold_spin.setSuffix(" %")
        card.add_layout(form_row_grid([
            ("Preferred languages", self._languages_edit),
            ("AV1 preset", self._preset_spin),
            ("AV1 CRF", self._crf_spin),
            ("Auto-accept confidence", self._threshold_spin),
        ]))
        card.add(CaptionLabel("Languages: comma-separated ISO-639-2 codes, most preferred first (e.g. ger, eng)."))
        return card

    # -- behaviour --------------------------------------------------------------------

    def refresh_tool_status(self) -> None:
        try:
            status = self._context.tool_status(refresh=True)
        except EpisodeMatchError:
            return
        for name in TOOL_NAMES:
            dot, version_label, path_label = self._tool_rows[name]
            info = status.get(name)
            if info.is_available:
                dot.set_color(_STATUS_COLORS["ok"])
                version_label.setText(f"v{info.version or '?'}")
                path_label.setText(str(info.path))
            else:
                dot.set_color(_STATUS_COLORS["missing"])
                version_label.setText("—")
                path_label.setText("not found — fetch or install")

    def _load_from_config(self) -> None:
        config = self._context.config
        self._tmdb_edit.setText(config.tmdb_api_key)
        self._osdt_edit.setText(config.opensubtitles_api_key)
        self._subdl_edit.setText(config.subdl_api_key)
        self._source_combo.setCurrentText(config.subtitle_source)
        self._llm_key_edit.setText(config.llm_api_key)
        self._llm_url_edit.setText(config.llm_base_url)
        self._llm_model_edit.setText(config.llm_model)
        self._template_edit.setText(config.naming.episode_template)
        self._season_folder.setChecked(config.naming.season_folder)
        self._dots_button.setChecked(config.naming.replace_spaces_with_dots)
        self._lowercase_button.setChecked(config.naming.lowercase_series)
        self._languages_edit.setText(", ".join(config.preferred_languages))
        self._preset_spin.setValue(config.av1_preset)
        self._crf_spin.setValue(config.av1_crf)
        self._threshold_spin.setValue(int(config.matching.auto_accept_confidence * 100))
        self._update_naming_preview()

    def _update_naming_preview(self) -> None:
        try:
            preview = format_episode_name(
                self._template_edit.text() or "{series} - S{season:02d}E{episode:02d} - {title}",
                series="Example Show",
                season=2,
                episode=7,
                title="The Perfect Episode",
                extension="",
                season_folder=True,
            )
        except Exception:  # noqa: BLE001 - invalid templates must not crash the view
            self._preview_label.setText("(invalid template)")
            self._preview_label.setStyleSheet(f"color: {Palette.ERROR};")
            return
        parts = [self._apply_style(part) for part in preview.split("/")]
        self._preview_label.setText("/".join(parts))
        self._preview_label.setStyleSheet(f"color: {Palette.ACCENT};")

    def _apply_style(self, part: str) -> str:
        if self._dots_button.isChecked():
            part = part.replace(" ", ".")
        if self._lowercase_button.isChecked():
            part = part.lower()
        return part

    def _on_fetch_tools(self) -> None:
        self._fetch_button.setEnabled(False)
        self._show_banner("info", "Downloading missing tools into ./bin/ …")
        self._fetch_worker = FetchToolsWorker(self._context.config)
        self._fetch_worker.progress.connect(self._on_fetch_progress)
        self._fetch_worker.finished_ok.connect(self._on_fetch_done)
        self._fetch_worker.failed.connect(self._on_fetch_failed)
        self._fetch_worker.start()

    def _on_test_llm(self) -> None:
        """Test the LLM endpoint with the values CURRENTLY in the fields — the
        user can validate before saving anything."""
        from episodematch.matcher.llm_arbiter import LlmSettings

        settings = LlmSettings(
            api_key=self._llm_key_edit.text().strip(),
            base_url=self._llm_url_edit.text().strip() or "https://open.bigmodel.cn/api/paas/v4",
            model=self._llm_model_edit.text().strip() or "glm-4-flash",
        )
        self._llm_test_button.setEnabled(False)
        self._llm_test_button.setText("Testing…")
        self._llm_status.setText(f"Testing connection to {settings.base_url} ({settings.model})…")
        self._llm_status.setStyleSheet(f"color: {Palette.TEXT_TERTIARY};")
        self._llm_status.show()
        self._show_banner("info", f"Testing LLM connection to {settings.base_url} ({settings.model})…")
        self._llm_test_worker = LlmTestWorker(settings)
        self._llm_test_worker.finished_ok.connect(self._on_llm_test_done)
        self._llm_test_worker.failed.connect(self._on_llm_test_failed)
        self._llm_test_worker.start()

    def _on_llm_test_done(self, message: str) -> None:
        self._llm_test_button.setEnabled(True)
        self._llm_test_button.setText("Test LLM connection")
        self._llm_status.setText(message)
        self._llm_status.setStyleSheet(f"color: {Palette.SUCCESS};")
        self._llm_status.show()
        self._show_banner("success", message)

    def _on_llm_test_failed(self, message: str) -> None:
        self._llm_test_button.setEnabled(True)
        self._llm_test_button.setText("Test LLM connection")
        self._llm_status.setText(message)
        self._llm_status.setStyleSheet(f"color: {Palette.ERROR};")
        self._llm_status.show()
        self._show_banner("error", message)

    def _on_fetch_progress(self, label: str, done: int, total: int | None) -> None:
        if total:
            self._show_banner("info", f"Downloading {label}: {done / total:.0%}")

    def _on_fetch_done(self, installed: tuple) -> None:
        self._fetch_button.setEnabled(True)
        if installed:
            self._show_banner("success", f"Installed: {', '.join(installed)}")
        else:
            self._show_banner("info", "All required tools are already present.")
        self.refresh_tool_status()

    def _on_fetch_failed(self, message: str) -> None:
        self._fetch_button.setEnabled(True)
        self._show_banner("error", message)

    def _on_save(self) -> None:
        config = self._context.config
        try:
            config.tmdb_api_key = self._tmdb_edit.text().strip()
            config.opensubtitles_api_key = self._osdt_edit.text().strip()
            config.subdl_api_key = self._subdl_edit.text().strip()
            config.subtitle_source = self._source_combo.currentText().strip().lower() or "opensubtitles"
            config.llm_api_key = self._llm_key_edit.text().strip()
            config.llm_base_url = self._llm_url_edit.text().strip() or "https://open.bigmodel.cn/api/paas/v4"
            config.llm_model = self._llm_model_edit.text().strip() or "glm-4-flash"
            from episodematch.core.config import NamingTemplate

            config.naming = NamingTemplate(
                episode_template=self._template_edit.text().strip() or config.naming.episode_template,
                season_folder=self._season_folder.isChecked(),
                replace_spaces_with_dots=self._dots_button.isChecked(),
                lowercase_series=self._lowercase_button.isChecked(),
            )
            languages = [part.strip().lower() for part in self._languages_edit.text().split(",") if part.strip()]
            config.preferred_languages = languages or ["ger", "eng"]
            config.av1_preset = self._preset_spin.value()
            config.av1_crf = self._crf_spin.value()
            from episodematch.core.config import MatchingSettings

            config.matching = MatchingSettings(
                auto_accept_confidence=self._threshold_spin.value() / 100.0,
            )
            self._context.config_store.save(config)
        except EpisodeMatchError as exc:
            self._show_banner("error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - pydantic validation errors etc.
            self._show_banner("error", f"Invalid settings: {exc}")
            return
        self.refresh_tool_status()
        self._show_banner("success", "Settings saved.")
        self.config_saved.emit()

    def _show_banner(self, level: str, text: str) -> None:
        self._banner.set_level(level, text)
        self._banner.show()

    def shutdown(self) -> None:
        for worker in (self._fetch_worker, getattr(self, "_llm_test_worker", None)):
            if worker is not None and worker.isRunning():
                worker.wait(1500)
