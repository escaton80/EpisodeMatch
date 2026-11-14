"""The EpisodeMatch FluentWindow: dark Win11 shell with a navigation rail."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from episodematch import __app_name__, __version__
from episodematch.core.config import AppConfig, ConfigStore
from episodematch.core.database import CacheDatabase
from episodematch.core.exceptions import EpisodeMatchError
from episodematch.ui.context import AppContext
from episodematch.ui.theme import Icon, Palette, app_icon_path, apply_theme, apply_windows11_chrome
from episodematch.ui.views.postprocess_view import PostprocessView
from episodematch.ui.views.settings_view import SettingsView
from episodematch.ui.views.visual_verify_view import VisualVerifyView
from episodematch.ui.views.workspace_view import WorkspaceView
from episodematch.ui.widgets.common import CaptionLabel, NavButton

logger = logging.getLogger(__name__)

_PAGE_WORKSPACE = 0
_PAGE_VERIFY = 1
_PAGE_POSTPROCESS = 2
_PAGE_SETTINGS = 3


def create_app_context(config: AppConfig | None = None, *, config_path: Path | str | None = None) -> AppContext:
    """Build the shared AppContext (config load + SQLite cache)."""
    config = config or ConfigStore(config_path).load()
    database: CacheDatabase | None = None
    try:
        database = CacheDatabase(config.database_path)
    except EpisodeMatchError as exc:
        logger.warning("Cache database unavailable (%s) — continuing without cache.", exc)
    return AppContext(
        config=config,
        config_store=ConfigStore(config_path),
        database=database,
    )


class FluentWindow(QMainWindow):
    """Main application window: navigation rail + stacked Fluent pages."""

    def __init__(self, context: AppContext | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context or create_app_context()
        self.setWindowTitle(f"{__app_name__} {__version__}")
        icon_path = app_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_nav_rail())
        root.addWidget(self._build_pages(), 1)
        self.setCentralWidget(central)

        self._views = {
            _PAGE_WORKSPACE: self._workspace,
            _PAGE_VERIFY: self._verify,
            _PAGE_POSTPROCESS: self._postprocess,
            _PAGE_SETTINGS: self._settings,
        }
        self._nav_buttons[_PAGE_WORKSPACE].setChecked(True)
        self._stack.setCurrentIndex(_PAGE_WORKSPACE)

    # -- construction ---------------------------------------------------------------

    def _build_nav_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("NavRail")
        rail.setFixedWidth(236)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(4)

        brand = QLabel(f"  {Icon.styled(Icon.MOVIE, 18, Palette.ACCENT)}  {__app_name__}")
        brand.setObjectName("NavBrand")
        layout.addWidget(brand)
        version = CaptionLabel(f"  v{__version__} — five-pillar episode matcher")
        layout.addWidget(version)
        layout.addSpacing(16)

        self._nav_buttons: dict[int, NavButton] = {}
        for page_index, (glyph, text, beta) in enumerate(
            (
                (Icon.HOME, "Workspace", False),
                (Icon.EYE, "Visual Verify", True),
                (Icon.REPAIR, "Post-Processing", True),
                (Icon.SETTINGS, "Settings", False),
            )
        ):
            button_text = f"{text} (beta)" if beta else text
            button = NavButton(glyph, button_text)
            if beta:
                button.setToolTip("Still under development — treat this page as a preview.")
            button.clicked.connect(lambda _checked=False, idx=page_index: self.navigate(idx))
            layout.addWidget(button)
            self._nav_buttons[page_index] = button

        layout.addStretch(1)
        footer = CaptionLabel("  MIT licensed · PySide6")
        layout.addWidget(footer)
        return rail

    def _build_pages(self) -> QWidget:
        self._stack = QStackedWidget()
        self._workspace = WorkspaceView(self._context)
        self._verify = VisualVerifyView(self._context)
        self._postprocess = PostprocessView(self._context)
        self._settings = SettingsView(self._context)
        for page in (self._workspace, self._verify, self._postprocess, self._settings):
            self._stack.addWidget(page)

        # Cross-view wiring.
        self._workspace.scan_finished.connect(self._on_scan_finished)
        self._workspace.files_available.connect(self._verify_load_files)
        self._settings.config_saved.connect(self._on_config_saved)
        return self._stack

    # -- navigation ---------------------------------------------------------------------

    def navigate(self, page_index: int) -> None:
        for index, button in self._nav_buttons.items():
            button.setChecked(index == page_index)
        self._stack.setCurrentIndex(page_index)

    def current_page_index(self) -> int:
        return self._stack.currentIndex()

    # -- cross-view wiring ------------------------------------------------------------------

    def _on_scan_finished(self, report: object) -> None:  # noqa: ANN001 - ScanReport
        self._postprocess.prefill_folder(self._context.last_folder)

    def _verify_load_files(self, files: list) -> None:
        report = self._context.last_report
        if report is not None and files:
            self._verify.load_from_scan(report)

    def _on_config_saved(self) -> None:
        # Tool paths may have changed (bin dir) — refresh statuses everywhere.
        self._settings.refresh_tool_status()

    # -- lifecycle ------------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().showEvent(event)
        apply_windows11_chrome(self)

    def _views_busy(self) -> bool:
        """True while any view still runs a worker thread (audit D2)."""
        return any(
            getattr(view, "is_busy", lambda: False)()  # noqa: B008 - default probe
            for view in self._views.values()
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        for view in self._views.values():
            if hasattr(view, "shutdown"):
                view.shutdown()
        if self._views_busy():
            # Audit D2: never close the shared database under live workers —
            # let the user wait (recommended) or explicitly accept the risk.
            if _confirm_close_while_busy(self) == "wait":
                while self._views_busy():
                    QApplication.processEvents()
                    time.sleep(0.05)
        if self._context.database is not None:
            self._context.database.close()
        super().closeEvent(event)


def run_gui(
    config: AppConfig | None = None,
    *,
    config_path: Path | str | None = None,
    smoke: bool = False,
) -> int:
    """Create the QApplication, apply the theme and run the main loop.

    ``smoke=True`` closes the window automatically after a short delay — used by
    the packaged EXE build and CI to verify the GUI boots without a display.
    """
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    gui_icon = app_icon_path()
    if gui_icon is not None:
        app.setWindowIcon(QIcon(str(gui_icon)))
    window = FluentWindow(create_app_context(config, config_path=config_path))
    if smoke:
        from PySide6.QtCore import QTimer

        def _finish() -> None:
            window.close()
            app.quit()

        QTimer.singleShot(1200, _finish)
    window.show()
    exit_code = app.exec()
    return 0 if smoke else exit_code


def _confirm_close_while_busy(parent) -> str:  # noqa: ANN001 - QWidget
    """Ask whether to wait for running background tasks or close anyway.

    Returns ``"wait"`` (default) or ``"close"`` — module-level so tests can
    stub the decision.
    """
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setWindowTitle("Background tasks running")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText(
        "Background tasks are still running.\n\n"
        "Wait until they finish (recommended), or close anyway?"
    )
    wait_button = box.addButton("Wait for tasks", QMessageBox.ButtonRole.AcceptRole)
    close_button = box.addButton("Close anyway", QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(wait_button)
    box.exec()
    return "close" if box.clickedButton() is close_button else "wait"


__all__ = ["FluentWindow", "create_app_context", "run_gui", "_confirm_close_while_busy"]
