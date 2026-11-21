"""Offscreen UI tests (QT_QPA_PLATFORM=offscreen): theme, widgets, views, window."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Marks the session for conftest.pytest_sessionfinish: after Qt tests, exit
# hard with the real status (PySide6 can segfault during interpreter teardown).
os.environ["EM_QT_TESTS"] = "1"

import pytest

pyside = pytest.importorskip("PySide6", reason="PySide6 not installed (pip install -e .[gui])")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QProgressBar, QLabel  # noqa: E402

from episodematch.core.config import AppConfig  # noqa: E402
from episodematch.core.database import CacheDatabase  # noqa: E402
from episodematch.core.models import (  # noqa: E402
    AssignmentEntry,
    Evidence,
    MatchCandidate,
    Pillar,
    SeasonAssignment,
    SeriesInfo,
)
from episodematch.matcher.pipeline import ScanReport  # noqa: E402
from episodematch.ui.context import AppContext  # noqa: E402
from episodematch.ui import main_window as main_window_module  # noqa: E402
from episodematch.ui.main_window import FluentWindow, create_app_context  # noqa: E402
from episodematch.ui.theme import Palette, build_qss  # noqa: E402
from episodematch.ui.views.postprocess_view import PostprocessView  # noqa: E402
from episodematch.ui.views.settings_view import SettingsView  # noqa: E402
from episodematch.ui.views.visual_verify_view import VisualVerifyView  # noqa: E402
from episodematch.ui.views.workspace_view import WorkspaceView  # noqa: E402
from episodematch.ui.widgets.common import Card, ConfidenceBar  # noqa: E402
from episodematch.ui.widgets.evidence_card import (  # noqa: E402
    EvidenceCard,
    EvidenceList,
    format_timestamp,
)
from episodematch.ui.widgets.multi_player import MultiFrameScrubber  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    # No teardown gymnastics here: the hard exit in conftest.pytest_unconfigure
    # runs after all summaries and prevents the PySide6 finalize segfault.
    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def ui_context(tmp_path: Path, config: AppConfig) -> AppContext:
    """AppContext pointing at tmp dirs (the `config` fixture clears env vars)."""
    store_config = AppConfig(bin_dir=tmp_path / "bin", data_dir=tmp_path / "data")
    from episodematch.core.config import ConfigStore

    return AppContext(config=store_config, config_store=ConfigStore(tmp_path / "data" / "config.json"))


@pytest.fixture()
def window(qapp: QApplication, ui_context: AppContext) -> FluentWindow:
    win = FluentWindow(ui_context)
    # show() is required: close() on a never-shown window is a no-op, which
    # would skip closeEvent (thread shutdown!) and randomly segfault at exit.
    win.show()
    qapp.processEvents()
    yield win
    for view in (win._workspace, win._verify, win._postprocess, win._settings):
        if hasattr(view, "shutdown"):
            view.shutdown()
    qapp.processEvents()
    win.close()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


class TestTheme:
    def test_qss_contains_core_selectors(self) -> None:
        qss = build_qss()
        assert len(qss) > 4000
        for selector in ("QFrame#Card", "QPushButton#AccentButton", "QLineEdit", "QScrollBar:vertical", "QFrame#NavRail"):
            assert selector in qss

    def test_palette_pillar_colors_complete(self) -> None:
        for pillar in ("disc_db", "subtitle", "audio", "visual", "llm"):
            assert pillar in Palette.PILLAR_COLORS

    def test_apply_theme_sets_stylesheet(self, qapp: QApplication) -> None:
        from episodematch.ui.theme import apply_theme

        length = apply_theme(qapp)
        assert length > 4000
        assert qapp.styleSheet()


# ---------------------------------------------------------------------------
# Common widgets
# ---------------------------------------------------------------------------


class TestCommonWidgets:
    def test_confidence_bar_clamps(self) -> None:
        bar = ConfidenceBar()
        bar.set_value(1.7)
        assert bar.value() == 1.0
        bar.set_value(-0.4)
        assert bar.value() == 0.0
        bar.set_value(0.9)
        assert bar.value() == 0.9

    def test_confidence_bar_color_tiers(self) -> None:
        bar = ConfidenceBar()
        bar.set_value(0.95)
        assert bar._bar_color().name().lower() == Palette.SUCCESS.lower()
        bar.set_value(0.6)
        assert bar._bar_color().name().lower() == Palette.WARNING.lower()
        bar.set_value(0.2)
        assert bar._bar_color().name().lower() == Palette.ERROR.lower()

    def test_card_add(self, qapp: QApplication) -> None:
        card = Card()
        label = card.add(QLabel("x"))
        assert label.parent() is card
        card.add_stretch()
        assert card.layout().count() >= 2


# ---------------------------------------------------------------------------
# Evidence widgets
# ---------------------------------------------------------------------------


def make_evidence(score: float = 0.9, *, kind: str = "fingerprint_match", pillar: Pillar = Pillar.DISC_DB) -> Evidence:
    return Evidence(
        pillar=pillar,
        kind=kind,
        score=score,
        summary="MPLS 00800.mpls matches disc entry -> S01E01 Pilot",
        details={"reasons": ["MPLS match", "duration ±0.4s"], "source": "thediscdb"},
        timestamp_seconds=42.5,
        episode_hint=1,
    )


class TestEvidenceWidgets:
    def test_format_timestamp(self) -> None:
        assert format_timestamp(None) == ""
        assert format_timestamp(42.5) == "00:42"
        assert format_timestamp(3725.0) == "1:02:05"

    def test_evidence_card_populates(self, qapp: QApplication) -> None:
        card = EvidenceCard(make_evidence())
        labels = card.findChildren(QLabel)
        texts = " | ".join(label.text() for label in labels)
        assert "Disc fingerprint" in texts
        assert "MPLS 00800.mpls" in texts
        assert "00:42" in texts

    def test_evidence_list_sorted_and_cleared(self, qapp: QApplication) -> None:
        widget = EvidenceList()
        widget.set_evidence([make_evidence(0.3), make_evidence(0.95), make_evidence(0.6, pillar=Pillar.VISUAL)])
        cards = widget.findChildren(EvidenceCard)
        assert len(cards) == 3
        widget.set_evidence([])
        assert len(widget.findChildren(EvidenceCard)) == 0


# ---------------------------------------------------------------------------
# Multi-frame scrubber
# ---------------------------------------------------------------------------


class TestMultiFrameScrubber:
    def test_set_files_builds_grid(self, qapp: QApplication) -> None:
        scrubber = MultiFrameScrubber(ffmpeg_path=None)
        files = [Path(f"D:/rips/title_t0{i}.mkv") for i in range(4)]
        scrubber.set_files(files, duration_seconds=2400.0)
        tiles = scrubber.findChildren(type(scrubber._grid.itemAt(0).widget()))
        assert len(tiles) == 4
        assert scrubber._slider.maximum() == 24000
        scrubber.shutdown()

    def test_set_timestamp_clamps(self, qapp: QApplication) -> None:
        scrubber = MultiFrameScrubber(ffmpeg_path=None)
        scrubber.set_files([Path("a.mkv")], duration_seconds=100.0)
        scrubber.set_timestamp(500.0)
        assert scrubber.current_timestamp() == 100.0
        scrubber.shutdown()

    def test_unavailable_ffmpeg_reports_failure(self, qapp: QApplication) -> None:
        scrubber = MultiFrameScrubber(ffmpeg_path=None)
        assert scrubber._grabber.available is False
        scrubber.set_files([Path("a.mkv")], duration_seconds=60.0)
        scrubber._grab_all()  # must not crash; tiles get the failure caption
        tile = scrubber._grid.itemAt(0).widget()
        assert "ffmpeg" in tile.status.text().lower()
        scrubber.shutdown()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def make_report() -> ScanReport:
    series = SeriesInfo(name="Test Show", tmdb_id=1)
    entry = AssignmentEntry(
        file_path=Path("D:/rips/title_t00.mkv"),
        season=1,
        episode=1,
        episode_title="Pilot",
        confidence=0.97,
        methods=["disc_db", "subtitle"],
        reasoning=["MPLS match"],
        evidence=[make_evidence()],
    )
    candidate = MatchCandidate(file_path=Path("D:/rips/title_t00.mkv"))
    candidate.add_evidence(make_evidence())
    assignment = SeasonAssignment(
        series=series,
        season=1,
        entries=[entry],
        unassigned=[],
    )
    return ScanReport(assignment=assignment, candidates=[candidate])


class TestWorkspaceView:
    def test_fill_results_and_inspector(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        report = make_report()
        ui_context.last_report = report
        view._fill_results(report)
        assert view._table.rowCount() == 1
        assert view._table.item(0, 1).text() == "S01E01"
        assert view._table.item(0, 2).text() == "Pilot"
        assert "97%" in view._table.item(0, 3).text()
        # Selection populates the evidence inspector.
        view._table.selectRow(0)
        view._on_selection_changed()
        assert "S01E01" in view._inspector_title.text()
        assert len(view._evidence_list.findChildren(EvidenceCard)) >= 1

    def test_scan_options_from_checkboxes(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view._pillar_checks["use_llm"].setChecked(True)
        view._depth_combo.setCurrentIndex(2)  # Deep
        options = view._build_scan_options()
        assert options.depth == "deep"
        assert options.use_llm is True
        assert options.use_disc_db is True

    def test_depth_combo_applies_pillar_presets(self, qapp: QApplication, ui_context: AppContext) -> None:
        """Three presets: Fast = DiscDB only, Medium = DiscDB + Subtitles,
        Deep = all five. The checkboxes must follow the selector — the pipeline
        gates audio/visual/LLM on these flags."""
        view = WorkspaceView(ui_context)

        # Initial state: Medium preset (previous default behavior).
        assert view._depth_combo.currentIndex() == 1
        assert view._pillar_checks["use_disc_db"].isChecked() is True
        assert view._pillar_checks["use_subtitles"].isChecked() is True
        assert view._pillar_checks["use_audio"].isChecked() is False
        assert view._pillar_checks["use_visual"].isChecked() is False
        assert view._pillar_checks["use_llm"].isChecked() is False

        view._depth_combo.setCurrentIndex(0)  # Fast: DiscDB only
        assert view._pillar_checks["use_disc_db"].isChecked() is True
        assert view._pillar_checks["use_subtitles"].isChecked() is False
        assert view._pillar_checks["use_audio"].isChecked() is False
        fast_options = view._build_scan_options()
        assert fast_options.depth == "fast" and fast_options.use_subtitles is False

        view._depth_combo.setCurrentIndex(1)  # Medium
        assert view._pillar_checks["use_subtitles"].isChecked() is True
        assert view._pillar_checks["use_audio"].isChecked() is False

        view._depth_combo.setCurrentIndex(2)  # Deep
        assert all(check.isChecked() for check in view._pillar_checks.values())
        assert view._build_scan_options().depth == "deep"

    def test_manual_override_survives_after_preset(self, qapp: QApplication, ui_context: AppContext) -> None:
        """After applying the Deep preset the user can untick single pillars —
        the built ScanOptions must honor that combination."""
        view = WorkspaceView(ui_context)
        view._depth_combo.setCurrentIndex(2)  # Deep: all on
        view._pillar_checks["use_llm"].setChecked(False)  # user says: deep but no LLM
        options = view._build_scan_options()
        assert options.depth == "deep"
        assert options.use_audio is True and options.use_visual is True
        assert options.use_llm is False

    def test_prefill(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view.prefill(folder=Path("D:/rips"), series="Test Show", season=3)
        assert view._folder_edit.text() == str(Path("D:/rips"))
        assert view._search_edit.text() == "Test Show"
        assert view._season_spin.value() == 3


class TestVisualVerifyView:
    def test_load_folder_populates_scrubber(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path) -> None:
        folder = tmp_path / "rips"
        folder.mkdir()
        for index in range(2):
            (folder / f"title_t0{index}.mkv").write_bytes(b"fake")
        view = VisualVerifyView(ui_context)
        view._folder_edit.setText(str(folder))
        view._on_load()
        assert len(view._files) == 2
        assert view._scrubber._grid.count() == 2
        view.shutdown()

    def test_load_from_scan(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = VisualVerifyView(ui_context)
        view.load_from_scan(make_report())
        assert len(view._files) == 1
        assert view._folder_edit.text().endswith("rips")
        view.shutdown()


class TestPostprocessView:
    def test_analyze_results_fill_table(self, qapp: QApplication, ui_context: AppContext) -> None:
        from episodematch.core.models import MkvFile, StreamInfo, StreamType
        from episodematch.postprocess.stream_cleaner import build_cleanup_plan

        view = PostprocessView(ui_context)
        mkv = MkvFile(
            path=Path("D:/fin/title_t00.mkv"),
            duration_seconds=2400.0,
            streams=[
                StreamInfo(index=1, stream_type=StreamType.AUDIO, language="ger"),
                StreamInfo(index=2, stream_type=StreamType.AUDIO, language="jpn"),
            ],
        )
        plan = build_cleanup_plan(mkv, preferred_languages=["ger"], keep_commentary=False)
        view._on_analyze_done([mkv], [plan])
        assert view._cleanup_table.rowCount() == 1
        assert view._cleanup_table.item(0, 0).text() == "title_t00.mkv"
        assert view._clean_button.isEnabled()

    def test_queue_rows_added(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path) -> None:
        folder = tmp_path / "done"
        folder.mkdir()
        (folder / "ep1.mkv").write_bytes(b"x")
        view = PostprocessView(ui_context)
        view._folder_edit.setText(str(folder))
        view._on_add_folder()
        assert view._queue_table.rowCount() == 1
        assert view._queue_table.item(0, 2).text() == "pending"
        assert isinstance(view._queue_table.cellWidget(0, 3), QProgressBar)

    def test_add_folder_skips_existing_outputs(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path) -> None:
        """Audit B3: files whose .av1.mkv output already exists are not queued
        again (re-encoding would silently overwrite the finished file)."""
        folder = tmp_path / "done"
        folder.mkdir()
        (folder / "ep1.mkv").write_bytes(b"x")
        (folder / "ep1.av1.mkv").write_bytes(b"finished encode")  # output exists
        (folder / "ep2.mkv").write_bytes(b"x")  # no output yet
        view = PostprocessView(ui_context)
        view._folder_edit.setText(str(folder))
        view._on_add_folder()
        assert view._queue_table.rowCount() == 1
        assert view._queue_table.item(0, 0).text().endswith("ep2.mkv")
        assert "1 file(s) queued" in view._banner.text() if view._banner.isVisible() else True


class TestSettingsView:
    def test_load_and_save_roundtrip(self, qapp: QApplication, ui_context: AppContext) -> None:
        ui_context.config.tmdb_api_key = "tmdb-key-123"
        view = SettingsView(ui_context)
        assert view._tmdb_edit.text() == "tmdb-key-123"
        view._llm_model_edit.setText("glm-4.5-air")
        view._on_save()
        assert ui_context.config.llm_model == "glm-4.5-air"
        assert ui_context.config_store.path.is_file()

    def test_naming_preview(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = SettingsView(ui_context)
        view._template_edit.setText("{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}")
        view._dots_button.setChecked(True)
        assert "Example.Show.-.S02E07.-.The.Perfect.Episode" in view._preview_label.text().replace(" ", ".")

    def test_tool_status_rows(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = SettingsView(ui_context)
        view.refresh_tool_status()
        # Without tools in the tmp bin dir every row shows "not found".
        for name in ("ffmpeg", "mkvmerge"):
            assert "not found" in view._tool_rows[name][2].text()

    def test_llm_connection_test_button(self, qapp: QApplication, ui_context: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Test button probes with the CURRENT form values and reports via
        the banner — without saving anything."""
        from PySide6.QtCore import QObject, Signal

        class FakeWorker(QObject):
            finished_ok = Signal(str)
            failed = Signal(str)

            def __init__(self, settings, *, client=None, parent=None) -> None:
                super().__init__(parent)
                self.settings = settings

            def start(self) -> None:
                self.finished_ok.emit(f"Connected — {self.settings.model} replied: 'OK'")

            def isRunning(self) -> bool:
                return False

        monkeypatch.setattr("episodematch.ui.views.settings_view.LlmTestWorker", FakeWorker)

        view = SettingsView(ui_context)
        view._llm_key_edit.setText("sk-live")
        view._llm_model_edit.setText("deepseek-chat")
        view._on_test_llm()

        assert view._llm_test_button.isEnabled()  # re-enabled after completion
        assert view._llm_test_button.text() == "Test LLM connection"  # restored
        assert view._llm_test_worker.settings.model == "deepseek-chat"
        assert view._llm_test_worker.settings.api_key == "sk-live"
        # INLINE status under the button (the bottom banner is off-screen there):
        assert view._llm_status.isHidden() is False
        assert "deepseek-chat" in view._llm_status.text()
        assert "OK" in view._llm_status.text()
        assert Palette.SUCCESS in view._llm_status.styleSheet()

    def test_llm_test_button_recovers_on_failure(self, qapp: QApplication, ui_context: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:
        from PySide6.QtCore import QObject, Signal

        class FailWorker(QObject):
            finished_ok = Signal(str)
            failed = Signal(str)

            def __init__(self, settings, *, client=None, parent=None) -> None:
                super().__init__(parent)

            def start(self) -> None:
                self.failed.emit("Connection failed: refused")

            def isRunning(self) -> bool:
                return False

        monkeypatch.setattr("episodematch.ui.views.settings_view.LlmTestWorker", FailWorker)
        view = SettingsView(ui_context)
        view._on_test_llm()
        assert view._llm_test_button.isEnabled()
        assert view._llm_test_button.text() == "Test LLM connection"
        assert "Connection failed" in view._llm_status.text()
        assert Palette.ERROR in view._llm_status.styleSheet()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class TestFluentWindow:
    def test_navigation(self, window: FluentWindow) -> None:
        assert window.current_page_index() == 0
        window.navigate(3)
        assert window.current_page_index() == 3
        window.navigate(1)
        assert window.current_page_index() == 1
        for index, button in window._nav_buttons.items():
            assert button.isChecked() == (index == 1)

    def test_scan_broadcast_updates_verify(self, window: FluentWindow) -> None:
        window._workspace._folder_edit.setText("D:/rips")
        report = make_report()
        window._workspace._on_scan_finished(report)
        assert len(window._verify._files) == 1
        assert window._postprocess._folder_edit.text().endswith("rips")

    def test_create_app_context_offline(self, tmp_path: Path, config: AppConfig) -> None:
        context = create_app_context(config)
        assert context.config.data_dir == config.resolved_data_dir

    def test_run_gui_smoke_boots_and_exits(self, qapp: QApplication, tmp_path: Path) -> None:
        """The packaged-EXE smoke path: full window boot, auto-close, exit 0."""
        from episodematch.core.config import ConfigStore
        from episodematch.ui.main_window import run_gui

        code = run_gui(
            AppConfig(bin_dir=tmp_path / "bin", data_dir=tmp_path / "data"),
            config_path=tmp_path / "data" / "config.json",
            smoke=True,
        )
        assert code == 0
        assert ConfigStore(tmp_path / "data" / "config.json").path == tmp_path / "data" / "config.json"

class _StubWorker(QThread):
    """A worker that ignores cancel() and finishes after `duration` seconds."""

    def __init__(self, duration: float = 0.3) -> None:
        super().__init__()
        self._duration = duration

    def cancel(self) -> None:
        pass  # deliberately ignores the cancellation handshake (worst case)

    def run(self) -> None:
        import time

        time.sleep(self._duration)


class TestShutdownHandshake:
    """Audit D2: the shared cache database must not close under live workers."""

    def _window_with_db(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path) -> FluentWindow:
        ui_context.database = CacheDatabase(tmp_path / "cache.sqlite3")
        win = FluentWindow(ui_context)
        win.show()
        qapp.processEvents()
        return win

    def test_views_busy_reflects_running_worker(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path) -> None:
        win = self._window_with_db(qapp, ui_context, tmp_path)
        try:
            assert win._views_busy() is False
            worker = _StubWorker(duration=0.2)
            win._workspace._scan_worker = worker
            worker.start()
            qapp.processEvents()
            assert win._views_busy() is True
            worker.wait(2000)
            assert win._views_busy() is False
        finally:
            win.close()

    def test_close_waits_until_views_idle_then_closes_db(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        win = self._window_with_db(qapp, ui_context, tmp_path)
        worker = _StubWorker(duration=0.2)  # ignores cancel(); finishes on its own
        win._workspace._scan_worker = worker
        worker.start()
        qapp.processEvents()
        monkeypatch.setattr(main_window_module, "_confirm_close_while_busy", lambda parent: "wait")
        try:
            win.close()
            qapp.processEvents()
            assert ui_context.database.is_closed  # closed only AFTER the worker ended
            assert not worker.isRunning()
        except Exception:
            worker.wait(3000)
            raise

    def test_close_anyway_closes_db_immediately(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        win = self._window_with_db(qapp, ui_context, tmp_path)
        worker = _StubWorker(duration=1.5)  # keeps running past the close
        win._workspace._scan_worker = worker
        worker.start()
        qapp.processEvents()
        monkeypatch.setattr(main_window_module, "_confirm_close_while_busy", lambda parent: "close")
        try:
            win.close()
            qapp.processEvents()
            assert ui_context.database.is_closed  # user chose to close anyway
        finally:
            worker.wait(4000)

    def test_idle_close_never_asks(self, qapp: QApplication, ui_context: AppContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        win = self._window_with_db(qapp, ui_context, tmp_path)
        monkeypatch.setattr(
            main_window_module,
            "_confirm_close_while_busy",
            lambda parent: (_ for _ in ()).throw(AssertionError("dialog must not open for an idle window")),
        )
        win.close()
        qapp.processEvents()
        assert ui_context.database.is_closed

class TestAppIcon:
    """The program logo (logo/) becomes the window/taskbar icon; frozen builds
    resolve it from the PyInstaller bundle (assets/)."""

    def test_dev_icon_found_in_repo_logo_dir(self) -> None:
        from episodematch.ui.theme import app_icon_path

        icon = app_icon_path()
        assert icon is not None and icon.is_file()
        assert "logo" in str(icon).lower()

    def test_fluent_window_uses_app_icon(self, qapp: QApplication, ui_context: AppContext) -> None:
        win = FluentWindow(ui_context)
        try:
            assert not win.windowIcon().isNull()
        finally:
            win.close()

    def test_frozen_resolves_bundled_assets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys as _sys

        from episodematch.ui.theme import app_icon_path

        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "episode_match.jpg").write_bytes(b"fake jpg")
        monkeypatch.setattr(_sys, "frozen", True, raising=False)
        monkeypatch.setattr(_sys, "_MEIPASS", str(tmp_path), raising=False)
        icon = app_icon_path()
        assert icon == assets / "episode_match.jpg"

class TestScanProgressDisplay:
    """The center progress bar must reflect ALL scan stages (not jump to 100 %
    after the probe) and the current activity must be clearly visible."""

    def _emit(self, view: WorkspaceView, events: list) -> list[int]:
        values = []
        for event in events:
            view._on_scan_progress(event)
            values.append(view._progress.value())
        return values

    def test_combined_percent_is_monotonic_and_stage_weighted(self, qapp: QApplication, ui_context: AppContext) -> None:
        from episodematch.core.models import PipelineStage, ProgressEvent

        view = WorkspaceView(ui_context)
        values = self._emit(
            view,
            [
                ProgressEvent(stage=PipelineStage.PROBE, message="Probed a.mkv", percent=100.0),
                ProgressEvent(stage=PipelineStage.SUBTITLES, message="Extracting dialog from b.mkv", percent=0.0),
                ProgressEvent(stage=PipelineStage.SUBTITLES, message="b.mkv: best match E01 (90%)", percent=50.0),
                ProgressEvent(stage=PipelineStage.SOLVE, message="Solving global assignment…"),
                ProgressEvent(stage=PipelineStage.SOLVE, message="Assigned 2/2 (avg 93%)", percent=100.0),
            ],
        )
        assert values == sorted(values)  # never jumps backwards
        assert values[0] == 10  # probe done -> 10 % of the whole scan
        assert values[2] == 40  # subtitles at 50 % -> 10+15+0.5*30
        assert values[-1] == 100
        assert "SOLVE" in view._progress_caption.text().upper()

    def test_events_without_percent_never_reset_the_bar(self, qapp: QApplication, ui_context: AppContext) -> None:
        from episodematch.core.models import PipelineStage, ProgressEvent

        view = WorkspaceView(ui_context)
        values = self._emit(
            view,
            [
                ProgressEvent(stage=PipelineStage.PROBE, message="Probed a.mkv", percent=100.0),
                ProgressEvent(stage=PipelineStage.SOLVE, message="TheDiscDB already conclusive"),
            ],
        )
        assert values == [10, 10]

    def test_pillar_report_shows_visible_activity(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view._on_pillar_report("title_t02.mkv", "subtitles", "ok", "561 lines")
        text = view._activity_caption.text()
        assert "title_t02.mkv" in text and "subtitles" in text

    def test_finished_caption_reports_summary(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view._on_scan_finished(make_report())
        caption = view._progress_caption.text()
        assert "Done" in caption

class TestDepthPromotion:
    """A manually checked deep-only pillar must not be silently disabled by
    the Fast/Medium presets — the scan promotes itself to Deep (observed: a
    Voyager DVD scan with only Title-card OCR checked finished 0/7 in seconds
    because Medium kept the visual pillar gated)."""

    def test_visual_checkbox_promotes_medium_to_deep(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view._depth_combo.setCurrentIndex(1)  # Medium
        view._pillar_checks["use_visual"].setChecked(True)
        options = view._build_scan_options()
        assert options.depth == "deep"
        assert options.use_visual is True

    def test_fast_preset_with_visual_checkbox_promotes_too(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view._depth_combo.setCurrentIndex(0)  # Fast
        view._pillar_checks["use_visual"].setChecked(True)
        options = view._build_scan_options()
        assert options.depth == "deep"

    def test_no_deep_pillars_stays_in_preset(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        view._depth_combo.setCurrentIndex(0)  # Fast, DiscDB only
        assert view._deep_pillars_requested() is False
        assert view._build_scan_options().depth == "fast"

    def test_deep_pillars_requested_reflects_checkboxes(self, qapp: QApplication, ui_context: AppContext) -> None:
        view = WorkspaceView(ui_context)
        assert view._deep_pillars_requested() is False  # Medium default
        view._pillar_checks["use_audio"].setChecked(True)
        assert view._deep_pillars_requested() is True
