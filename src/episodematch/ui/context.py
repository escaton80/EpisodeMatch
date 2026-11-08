"""Application context shared by all views (config, database, tool resolver)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from episodematch.core.config import AppConfig, ConfigStore
from episodematch.core.database import CacheDatabase
from episodematch.matcher.pipeline import ScanReport

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Dependency + state hub passed to every view."""

    config: AppConfig
    config_store: ConfigStore
    database: Optional[CacheDatabase] = None
    # Last completed scan (shared between workspace and postprocess views).
    last_report: Optional[ScanReport] = None
    # Directory of the last scan (pre-fill for other views).
    last_folder: Optional[Path] = None

    # -- paths ---------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.config.resolved_data_dir

    # -- tools -----------------------------------------------------------------

    def tool_resolver(self):
        from episodematch.core.binaries import ToolResolver

        if not hasattr(self, "_resolver") or getattr(self, "_resolver_bin_dir", None) != self.config.resolved_bin_dir:
            self._resolver = ToolResolver(self.config.resolved_bin_dir)  # type: ignore[attr-defined]
            self._resolver_bin_dir = self.config.resolved_bin_dir  # type: ignore[attr-defined]
        return self._resolver  # type: ignore[attr-defined]

    def tool_path(self, name: str) -> Optional[Path]:
        """Path of a tool or None (never raises)."""
        info = self.tool_resolver().status().get(name)
        return info.path if info.is_available else None

    def tool_status(self, *, refresh: bool = False):
        return self.tool_resolver().status(refresh=refresh)

    # -- assignment helpers -------------------------------------------------------

    @property
    def assignment(self):
        return self.last_report.assignment if self.last_report else None
