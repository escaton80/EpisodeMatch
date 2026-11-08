"""Windows 11 dark Fluent theme: palette, complete QSS stylesheet and DWM helpers.

Everything here is hand-written (MIT, PySide6/LGPL only) — no third-party widget
libraries. The stylesheet targets Windows 11 visuals: layered surfaces, 8 px card
radii, accent fills, 2 px input focus affordances and slim rounded scrollbars.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

# ---------------------------------------------------------------------------
# Palette (Windows 11 dark Fluent defaults)
# ---------------------------------------------------------------------------


class Palette:
    """Semantic color tokens; all hex without alpha unless noted."""

    # Layers
    WINDOW_BG = "#202020"
    LAYER = "#272727"
    CARD = "#2D2D2D"
    CARD_HOVER = "#323232"
    CARD_BORDER = "#383838"
    CONTROL = "#3A3A3A"
    CONTROL_HOVER = "#414141"
    CONTROL_PRESSED = "#323232"
    CONTROL_DISABLED = "#2D2D2D"

    # Content
    TEXT = "#FFFFFF"
    TEXT_SECONDARY = "#C9C9C9"  # Fluent 78% white
    TEXT_TERTIARY = "#9A9A9A"  # 55% white
    TEXT_DISABLED = "#6B6B6B"

    # Accent
    ACCENT = "#60CDFF"
    ACCENT_HOVER = "#77D3FF"
    ACCENT_PRESSED = "#0078D4"
    ACCENT_TEXT = "#001B3A"
    ACCENT_SOFT = "#0F3B5C"  # subtle accent fill for selections

    # States
    SUCCESS = "#6CCB5F"
    WARNING = "#FCE100"
    ERROR = "#FF99A4"
    SKIPPED = "#9A9A9A"

    # Pillar colors (used by badges and evidence cards)
    PILLAR_COLORS = {
        "metadata": "#9A9A9A",
        "disc_db": "#60CDFF",
        "subtitle": "#6CCB5F",
        "audio": "#FCE100",
        "visual": "#FFB900",
        "llm": "#C794FF",
    }

    # Font stack: Segoe UI Variable ships with Win11, Segoe UI is the fallback.
    FONT_STACK = '"Segoe UI Variable Text", "Segoe UI", "Noto Sans", sans-serif'
    DISPLAY_STACK = '"Segoe UI Variable Display", "Segoe UI", "Noto Sans", sans-serif'
    ICON_STACK = '"Segoe Fluent Icons", "Segoe MDL2 Assets"'

    RADIUS = "8px"
    RADIUS_SMALL = "4px"


# ---------------------------------------------------------------------------
# Icon glyphs (Segoe Fluent Icons / MDL2 — both ship with Windows 10/11)
# ---------------------------------------------------------------------------


class Icon:
    """Unicode codepoints from the Segoe Fluent Icons / MDL2 Assets fonts."""

    HOME = "\uE80F"
    SEARCH = "\uE721"
    MOVIE = "\uE8B2"
    REPAIR = "\uE90F"
    SETTINGS = "\uE713"
    PLAY = "\uE768"
    PAUSE = "\uE769"
    FOLDER_OPEN = "\uED25"
    REFRESH = "\uE72C"
    CHECKMARK = "\uE73E"
    CANCEL = "\uE711"
    WARNING = "\uE7BA"
    INFO = "\uE946"
    DOWNLOAD = "\uE896"
    SAVE = "\uE74E"
    UNDO = "\uE7A7"
    CHEVRON_RIGHT = "\uE76C"
    CHEVRON_DOWN = "\uE70D"
    TIMELINE = "\uE9D9"
    SUBTITLE = "\uE7F3"  # CC glyphs fallback: E7F3
    MUSIC = "\uE8D6"
    EYE = "\uE7B3"
    ROBOT = "\uE99A"

    @classmethod
    def styled(cls, glyph: str, size: int = 14, color: str | None = None) -> str:
        """Inline HTML span rendering a Fluent icon glyph (for QLabel rich text)."""
        span_color = f" color:{color};" if color else ""
        return (
            "<span style=\"font-family:'Segoe Fluent Icons','Segoe MDL2 Assets';"
            f"font-size:{size}px;{span_color}\">{glyph}</span>"
        )

    _PIXMAP_CACHE: dict[tuple[str, int, str], QPixmap] = {}

    @classmethod
    def pixmap(cls, glyph: str, size: int = 16, color: str | None = None) -> QPixmap:
        """Render a Fluent glyph into a QPixmap (works for buttons and labels)."""
        key = (glyph, size, color or "")
        cached = cls._PIXMAP_CACHE.get(key)
        if cached is not None:
            return cached
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = QFont()
        font.setFamilies(["Segoe Fluent Icons", "Segoe MDL2 Assets"])
        font.setPixelSize(int(size * 0.92))
        painter.setFont(font)
        painter.setPen(QColor(color or Palette.TEXT_SECONDARY))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
        painter.end()
        cls._PIXMAP_CACHE[key] = pixmap
        return pixmap

    @classmethod
    def icon(cls, glyph: str, size: int = 16, color: str | None = None) -> QIcon:
        return QIcon(cls.pixmap(glyph, size, color))


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------


def build_qss() -> str:
    """Build the complete application stylesheet."""
    p = Palette()
    return f"""
* {{
    outline: none;
}}
QWidget {{
    background: transparent;
    color: {p.TEXT};
    font-family: {p.FONT_STACK};
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {p.WINDOW_BG};
}}
QWidget:disabled {{
    color: {p.TEXT_DISABLED};
}}

/* --- Typography helpers ------------------------------------------------- */
QLabel {{
    background: transparent;
}}
QLabel#TitleLabel {{
    font-family: {p.DISPLAY_STACK};
    font-size: 26px;
    font-weight: 600;
}}
QLabel#SubtitleLabel {{
    font-size: 13px;
    color: {p.TEXT_SECONDARY};
}}
QLabel#SectionLabel {{
    font-family: {p.DISPLAY_STACK};
    font-size: 16px;
    font-weight: 600;
}}
QLabel#CaptionLabel {{
    font-size: 11px;
    color: {p.TEXT_TERTIARY};
}}
QLabel[styleClass="mono"] {{
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
}}
QLabel[styleClass="success"] {{ color: {p.SUCCESS}; }}
QLabel[styleClass="error"] {{ color: {p.ERROR}; }}
QLabel[styleClass="warning"] {{ color: {p.WARNING}; }}
QLabel[styleClass="secondary"] {{ color: {p.TEXT_SECONDARY}; }}
QLabel[styleClass="tertiary"] {{ color: {p.TEXT_TERTIARY}; }}

/* --- Cards ---------------------------------------------------------------- */
QFrame#Card {{
    background: {p.CARD};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS};
}}
QFrame#Card:hover {{
    background: {p.CARD_HOVER};
}}
QFrame#CardFlat {{
    background: {p.LAYER};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS};
}}

/* --- Buttons ---------------------------------------------------------------- */
QPushButton {{
    background: {p.CONTROL};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS_SMALL};
    padding: 6px 14px;
    color: {p.TEXT};
    min-height: 24px;
}}
QPushButton:hover {{
    background: {p.CONTROL_HOVER};
}}
QPushButton:pressed {{
    background: {p.CONTROL_PRESSED};
    color: {p.TEXT_SECONDARY};
}}
QPushButton:focus {{
    border: 1px solid {p.ACCENT};
}}
QPushButton:disabled {{
    background: {p.CONTROL_DISABLED};
    color: {p.TEXT_DISABLED};
    border-color: {p.CARD_BORDER};
}}
QPushButton#AccentButton {{
    background: {p.ACCENT};
    color: {p.ACCENT_TEXT};
    font-weight: 600;
    border: none;
    border-radius: {p.RADIUS_SMALL};
}}
QPushButton#AccentButton:hover {{
    background: {p.ACCENT_HOVER};
}}
QPushButton#AccentButton:pressed {{
    background: {p.ACCENT_PRESSED};
    color: #FFFFFF;
}}
QPushButton#AccentButton:disabled {{
    background: {p.ACCENT_SOFT};
    color: {p.TEXT_DISABLED};
}}
QPushButton#SubtleButton {{
    background: transparent;
    border: none;
    color: {p.TEXT_SECONDARY};
    padding: 5px 10px;
}}
QPushButton#SubtleButton:hover {{
    background: {p.CONTROL};
    color: {p.TEXT};
}}
QPushButton#DangerButton {{
    background: transparent;
    border: 1px solid {p.ERROR};
    color: {p.ERROR};
}}
QPushButton#DangerButton:hover {{
    background: rgba(255, 153, 164, 0.12);
}}

/* --- Inputs ----------------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {p.CONTROL};
    border: 1px solid {p.CARD_BORDER};
    border-bottom: 1px solid {p.TEXT_TERTIARY};
    border-radius: {p.RADIUS_SMALL};
    padding: 5px 10px;
    selection-background-color: {p.ACCENT_PRESSED};
    selection-color: #FFFFFF;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    background: {p.CONTROL_HOVER};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {p.CARD_BORDER};
    border-bottom: 2px solid {p.ACCENT};
    background: {p.CONTROL_PRESSED};
}}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background: {p.CONTROL_DISABLED};
    color: {p.TEXT_DISABLED};
}}
QComboBox::drop-down {{
    border: none;
    width: 26px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {p.TEXT_SECONDARY};
    margin-right: 10px;
}}
QComboBox QAbstractItemView {{
    background: {p.LAYER};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS_SMALL};
    selection-background-color: {p.ACCENT_SOFT};
    selection-color: {p.TEXT};
    outline: none;
    padding: 4px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 18px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {p.TEXT_SECONDARY};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {p.TEXT_SECONDARY};
}}

/* --- Check / radio --------------------------------------------------------- */
QCheckBox, QRadioButton {{
    spacing: 8px;
    color: {p.TEXT};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {p.TEXT_TERTIARY};
    background: {p.CONTROL};
}}
QCheckBox::indicator {{
    border-radius: {p.RADIUS_SMALL};
}}
QRadioButton::indicator {{
    border-radius: 9px;
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    background: {p.CONTROL_HOVER};
}}
QCheckBox::indicator:checked {{
    background: {p.ACCENT};
    border-color: {p.ACCENT};
    image: none;
}}
QRadioButton::indicator:checked {{
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.9, fx:0.5, fy:0.5,
        stop:0 {p.ACCENT} 6px, stop:1 {p.CONTROL} 7px);
    border-color: {p.ACCENT};
}}
QCheckBox::indicator:checked::after {{
    /* checkmark approximated by inner fill */
}}

/* --- Item views ------------------------------------------------------------- */
QTableWidget, QTableView, QTreeWidget, QTreeView, QListWidget, QListView {{
    background: {p.LAYER};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS};
    alternate-background-color: {p.LAYER};
    selection-background-color: {p.ACCENT_SOFT};
    selection-color: {p.TEXT};
    outline: none;
    gridline-color: transparent;
}}
QHeaderView::section {{
    background: {p.LAYER};
    color: {p.TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {p.CARD_BORDER};
    padding: 7px 10px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: {p.LAYER};
    border: none;
}}
QListWidget::item, QListView::item {{
    padding: 6px 8px;
    border-radius: {p.RADIUS_SMALL};
}}
QListWidget::item:hover, QListView::item:hover {{
    background: {p.CONTROL};
}}
QListWidget::item:selected, QListView::item:selected {{
    background: {p.ACCENT_SOFT};
    color: {p.TEXT};
}}

/* --- Bars & sliders ----------------------------------------------------------- */
QProgressBar {{
    background: {p.CONTROL};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {p.ACCENT};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {p.CONTROL};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {p.ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {p.ACCENT};
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
    border: 3px solid {p.WINDOW_BG};
}}
QSlider::handle:horizontal:hover {{
    background: {p.ACCENT_HOVER};
}}

/* --- Scrollbars (slim Fluent style) --------------------------------------------- */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {p.CONTROL_HOVER};
    border-radius: 4px;
    min-height: 32px;
    margin: 0 3px;
}}
QScrollBar::handle:vertical:hover {{
    background: {p.TEXT_TERTIARY};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {p.CONTROL_HOVER};
    border-radius: 4px;
    min-width: 32px;
    margin: 3px 0;
}}
QScrollBar::handle:horizontal:hover {{
    background: {p.TEXT_TERTIARY};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* --- Tabs, menus, tooltips -------------------------------------------------------- */
QTabBar::tab {{
    background: transparent;
    color: {p.TEXT_SECONDARY};
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {p.TEXT};
    border-bottom: 2px solid {p.ACCENT};
}}
QTabWidget::pane {{
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS};
}}
QMenu {{
    background: {p.LAYER};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS_SMALL};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px;
    border-radius: {p.RADIUS_SMALL};
}}
QMenu::item:selected {{
    background: {p.ACCENT_SOFT};
}}
QToolTip {{
    background: {p.LAYER};
    color: {p.TEXT};
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS_SMALL};
    padding: 6px 8px;
}}
QStatusBar {{
    background: {p.LAYER};
    color: {p.TEXT_SECONDARY};
    border-top: 1px solid {p.CARD_BORDER};
}}
QSplitter::handle {{
    background: transparent;
    width: 6px;
}}
QGroupBox {{
    border: 1px solid {p.CARD_BORDER};
    border-radius: {p.RADIUS};
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {p.TEXT_SECONDARY};
}}


/* --- Navigation rail -------------------------------------------------------------- */
QFrame#NavRail {{
    background: {p.LAYER};
    border-right: 1px solid {p.CARD_BORDER};
}}
QPushButton#NavButton {{
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    color: {p.TEXT_SECONDARY};
    font-size: 13px;
}}
QPushButton#NavButton:hover {{
    background: {p.CONTROL};
    color: {p.TEXT};
}}
QPushButton#NavButton:checked {{
    background: {p.CONTROL_HOVER};
    color: {p.TEXT};
    font-weight: 600;
}}
QPushButton#NavButton:checked:true {{
    background: {p.CONTROL_HOVER};
}}
QLabel#NavBrand {{
    font-family: {p.DISPLAY_STACK};
    font-size: 17px;
    font-weight: 600;
    color: {p.TEXT};
    background: transparent;
}}

/* --- Evidence cards ------------------------------------------------------------------ */
QFrame#EvidenceCard {{
    background: {p.CARD};
    border: 1px solid {p.CARD_BORDER};
    border-left: 3px solid {p.ACCENT};
    border-radius: 6px;
}}
"""


# ---------------------------------------------------------------------------
# Windows 11 DWM helpers (dark title bar + Mica backdrop), via ctypes
# ---------------------------------------------------------------------------


def apply_windows11_chrome(window) -> bool:  # noqa: ANN001 - QWidget
    """Enable the immersive dark title bar and the Mica backdrop (Win11 22H2+).

    Returns True when any backdrop effect was applied. Safe no-op on older
    systems — the dark QSS already provides the full visual.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        hwnd = int(window.winId())
        dwm = ctypes.windll.dwmapi

        # DWMWA_USE_IMMERSIVE_DARK_MODE (20 on modern builds, 19 on early ones).
        dark = ctypes.c_int(1)
        if dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)) != 0:
            dwm.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(dark), ctypes.sizeof(dark))

        # DWMWA_SYSTEMBACKDROP_TYPE = 38; DWMSBT_MAINWINDOW (Mica) = 2.
        backdrop = ctypes.c_int(2)
        result = dwm.DwmSetWindowAttribute(hwnd, 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
        if result != 0:
            # Older builds: try DWMWA_MICA_EFFECT (undocumented, 1029).
            backdrop = ctypes.c_int(1)
            dwm.DwmSetWindowAttribute(hwnd, 1029, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
            return False
        return True
    except Exception:  # noqa: BLE001 - cosmetic only, never fatal
        return False


def apply_theme(app) -> str:  # noqa: ANN001 - QApplication
    """Apply the global stylesheet to a QApplication; returns the QSS length."""
    qss = build_qss()
    app.setStyleSheet(qss)
    return len(qss)
