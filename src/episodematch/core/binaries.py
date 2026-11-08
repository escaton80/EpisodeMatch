"""Locate and validate external tools (ffmpeg, ffprobe, MKVToolNix).

Search order per tool:

1. the local project ``./bin/`` folder,
2. the system ``PATH`` (via :func:`shutil.which`),
3. ``C:\\Program Files\\MKVToolNix\\`` (and the x86 variant).

Versions are probed by executing the binary with ``--version`` and parsing the first
output line. All results are returned as :class:`episodematch.core.models.ToolInfo`.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from episodematch.core.exceptions import BinaryNotFoundError
from episodematch.core.models import ToolInfo, ToolSource, ToolStatus

logger = logging.getLogger(__name__)

#: Tools EpisodeMatch uses, with the executables that satisfy them.
#: ``seconv`` (SubtitleEdit's headless converter) powers the PGS fast path.
TOOL_NAMES: tuple[str, ...] = ("ffmpeg", "ffprobe", "mkvmerge", "mkvextract", "mkvpropedit", "seconv")

#: Additional install locations probed per tool (relative to well-known roots).
_MKVTOOLNIX_PROGRAM_DIRS = (
    r"C:\Program Files\MKVToolNix",
    r"C:\Program Files (x86)\MKVToolNix",
)

_VERSION_TIMEOUT = 15.0

_VERSION_PATTERNS = (
    re.compile(r"ffmpeg version\s+(\S+)"),
    re.compile(r"ffprobe version\s+(\S+)"),
    re.compile(r"mkvmerge v(\S+)"),
    re.compile(r"mkvextract v(\S+)"),
    re.compile(r"mkvpropedit v(\S+)"),
    re.compile(r"seconv\s+v?(\S+)"),
    re.compile(r"version\s+(\S+)"),
)


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def _exe_name(tool: str) -> str:
    return f"{tool}.exe" if _is_windows() else tool


def _candidate_paths(tool: str, bin_dir: Path) -> list[tuple[Path, ToolSource]]:
    """Return all candidate (path, source) pairs for a tool, in priority order."""
    exe = _exe_name(tool)
    candidates: list[tuple[Path, ToolSource]] = [(bin_dir / exe, ToolSource.LOCAL_BIN)]

    which_hit = shutil.which(tool)
    if which_hit:
        candidates.append((Path(which_hit), ToolSource.SYSTEM_PATH))

    for root in _MKVTOOLNIX_PROGRAM_DIRS:
        root_path = Path(root)
        direct = root_path / exe
        if direct != candidates[0][0]:
            candidates.append((direct, ToolSource.PROGRAM_FILES))
        # MKVToolNix installs may live in a versioned subfolder.
        if root_path.is_dir():
            for child in sorted(root_path.glob("*/"), reverse=True):
                probe = child / exe
                if probe.is_file():
                    candidates.append((probe, ToolSource.PROGRAM_FILES))
    return candidates


def probe_version(executable: Path, *, timeout: float = _VERSION_TIMEOUT) -> str | None:
    """Run ``<tool> --version`` and extract a version token from its output."""
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Version probe failed for %s: %s", executable, exc)
        return None
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    for pattern in _VERSION_PATTERNS:
        match = pattern.search(output)
        if match:
            return match.group(1)
    first_line = output.strip().splitlines()
    return first_line[0][:80] if first_line else None


def locate_tool(tool: str, *, bin_dir: Path = Path("bin"), probe: bool = True) -> ToolInfo:
    """Locate one tool. Returns a ToolInfo with source NOT_FOUND when unavailable."""
    bin_dir = Path(bin_dir)
    for candidate, source in _candidate_paths(tool, bin_dir):
        if candidate.is_file():
            version = probe_version(candidate) if probe else None
            logger.debug("Found %s at %s (%s, version %s)", tool, candidate, source.value, version)
            return ToolInfo(name=tool, path=candidate, version=version, source=source)
    return ToolInfo(name=tool, path=None, version=None, source=ToolSource.NOT_FOUND)


def locate_subtitle_edit(bin_dir: Path) -> Path | None:
    """Locate the optional SubtitleEdit CLI.

    Search order: ``<bin_dir>/SubtitleEdit.exe``, ``<bin_dir>/SubtitleEdit/SubtitleEdit.exe``
    (the common portable-layout subfolder), then the system PATH. Returns None when
    unavailable — callers fall back to the native Python PGS pipeline.
    """
    exe = _exe_name("SubtitleEdit")
    for candidate in (Path(bin_dir) / exe, Path(bin_dir) / "SubtitleEdit" / exe):
        if candidate.is_file():
            return candidate
    which_hit = shutil.which("SubtitleEdit")
    return Path(which_hit) if which_hit else None


def locate_seconv(bin_dir: Path) -> Path | None:
    """Locate ``seconv.exe`` — SubtitleEdit's headless console converter.

    Unlike the GUI ``SubtitleEdit.exe /convert`` (which opens the OCR dialog for
    image-based subtitles), seconv converts fully unattended. Search order:
    ``<bin_dir>/seconv.exe``, ``<bin_dir>/SubtitleEdit/seconv.exe``,
    ``<bin_dir>/SubtitleEdit/seconv/seconv.exe``, system PATH.
    """
    exe = _exe_name("seconv")
    candidates = (
        Path(bin_dir) / exe,
        Path(bin_dir) / "SubtitleEdit" / exe,
        Path(bin_dir) / "SubtitleEdit" / "seconv" / exe,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    which_hit = shutil.which("seconv")
    return Path(which_hit) if which_hit else None


def locate_all_tools(*, bin_dir: Path = Path("bin"), probe: bool = True) -> ToolStatus:
    """Locate every known tool and return the full status snapshot.

    ``seconv`` uses :func:`locate_seconv` because it may also live inside the
    SubtitleEdit subfolder (``bin/SubtitleEdit/seconv.exe``).
    """
    tools: dict[str, ToolInfo] = {}
    for name in TOOL_NAMES:
        if name == "seconv":
            path = locate_seconv(bin_dir)
            version = probe_version(path) if path is not None and probe else None
            source = ToolSource.NOT_FOUND
            if path is not None:
                try:
                    source = (
                        ToolSource.LOCAL_BIN
                        if path.resolve().is_relative_to(Path(bin_dir).resolve())
                        else ToolSource.SYSTEM_PATH
                    )
                except OSError:
                    source = ToolSource.LOCAL_BIN
            tools[name] = ToolInfo(name=name, path=path, version=version, source=source)
        else:
            tools[name] = locate_tool(name, bin_dir=bin_dir, probe=probe)
    return ToolStatus(tools=tools)


def require_tool(status: ToolStatus, name: str) -> Path:
    """Return the path of a required tool or raise :class:`BinaryNotFoundError`."""
    info = status.get(name)
    if not info.is_available or info.path is None:
        raise BinaryNotFoundError(
            f"Required tool '{name}.exe' was not found in ./bin/, PATH or Program Files.",
            hint="Run 'episodematch fetch-tools' to download it into ./bin/ automatically.",
        )
    return info.path


class ToolResolver:
    """Caching tool resolver bound to a bin directory.

    The pipeline resolves tools repeatedly; this wrapper caches results until
    :meth:`invalidate` is called (e.g. after a successful download).
    """

    def __init__(self, bin_dir: Path, *, probe: bool = True) -> None:
        self._bin_dir = Path(bin_dir)
        self._probe = probe
        self._status: ToolStatus | None = None

    @property
    def bin_dir(self) -> Path:
        return self._bin_dir

    def status(self, *, refresh: bool = False) -> ToolStatus:
        if self._status is None or refresh:
            self._status = locate_all_tools(bin_dir=self._bin_dir, probe=self._probe)
        return self._status

    def tool_path(self, name: str) -> Path:
        return require_tool(self.status(), name)

    def find(self, name: str) -> ToolInfo:
        return self.status().get(name)

    def invalidate(self) -> None:
        self._status = None
