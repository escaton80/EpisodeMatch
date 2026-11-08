"""Automatic downloader for missing external tools into ``./bin/``.

Sources:

* **ffmpeg / ffprobe** — gyan.dev release-essentials ZIP (primary) with the BtbN
  GitHub auto-latest build as fallback.
* **MKVToolNix** — release listing on ``mkvtoolnix.download`` is scraped for the newest
  version that ships a Windows archive; a pinned version URL is used as fallback.

The downloader streams with progress callbacks, supports cancellation, refuses
zip-slip members, flattens the wanted ``*.exe`` files into ``./bin/`` and verifies
every installed binary with a ``--version`` probe before reporting success.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import httpx

from episodematch.core.binaries import _exe_name, locate_tool, probe_version
from episodematch.core.exceptions import BinaryDownloadError
from episodematch.core.models import ToolInfo, ToolStatus

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int | None], None]  # (label, done, total)
CancelCheck = Callable[[], bool]

_FF_URLS = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip",
)

_MKVTOOLNIX_RELEASES_URL = "https://mkvtoolnix.download/windows/releases/"
#: Used when the release listing cannot be scraped (offline mirror, blocked listing, ...).
_MKVTOOLNIX_PINNED_VERSION = "83.0"

_FFMPEG_TOOLS = ("ffmpeg", "ffprobe")
_MKVTOOLS = ("mkvmerge", "mkvextract", "mkvpropedit")
_SECONV_TOOLS = ("seconv",)

#: SubtitleEdit's headless converter — standalone asset of the SE release
#: (the "latest/download" path redirects to the newest tagged version).
_SECONV_URLS = (
    "https://github.com/SubtitleEdit/subtitleedit/releases/latest/download/SeConv-Windows-x64.zip",
    "https://github.com/SubtitleEdit/subtitleedit/releases/download/v5.4.0/SeConv-Windows-x64.zip",
)

_DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=120.0)
_CHUNK_SIZE = 256 * 1024
_MIN_ARCHIVE_BYTES = 1024 * 1024  # sane lower bound for real tool archives

#: Matches ``href="90.0/"``, ``href="./90.0/"`` and ``href="/windows/releases/90.0/"``.
_VERSION_DIR_RE = re.compile(r'href="(?:\./|/windows/releases/)?(\d+\.\d+(?:\.\d+)?)/"')
#: Matches ``href="mkvtoolnix-64-bit-90.0.7z"`` and ``href="./mkvtoolnix-64-bit-90.0.7z"``.
_ARCHIVE_RE = re.compile(r'href="\.?/?(mkvtoolnix-[^"/]+?\.(?:zip|7z))"', re.IGNORECASE)


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a download/extract/install cycle."""

    installed: tuple[str, ...]
    skipped: tuple[str, ...] = field(default_factory=tuple)
    archive_url: str = ""

    @property
    def success(self) -> bool:
        return bool(self.installed)


def _check_cancel(cancel: CancelCheck | None) -> None:
    if cancel is not None and cancel():
        raise BinaryDownloadError("Download cancelled by user.")


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------


def validate_zip_member(name: str) -> None:
    """Refuse zip members that would escape the destination directory (zip slip)."""
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../"):
        raise BinaryDownloadError(f"Unsafe archive member rejected: {name!r}")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    depth = 0
    for part in parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                raise BinaryDownloadError(f"Unsafe archive member rejected: {name!r}")
        else:
            depth += 1
    if re.match(r"^[A-Za-z]:", normalized):
        raise BinaryDownloadError(f"Absolute archive path rejected: {name!r}")


def safe_extract_zip(archive: Path, dest_dir: Path, *, cancel: CancelCheck | None = None) -> list[Path]:
    """Extract a ZIP archive safely; returns the list of extracted file paths."""
    extracted: list[Path] = []
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            _check_cancel(cancel)
            validate_zip_member(member.filename)
            target = dest_dir / member.filename
            target = target.resolve()
            if not str(target).startswith(str(dest_dir)):
                raise BinaryDownloadError(f"Unsafe archive member rejected: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def extract_7z(archive: Path, dest_dir: Path, *, cancel: CancelCheck | None = None) -> list[Path]:
    """Extract a 7z archive using the optional ``py7zr`` dependency."""
    try:
        import py7zr  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BinaryDownloadError(
            f"Archive {archive.name} is a .7z file and py7zr is not installed.",
            hint="pip install py7zr  (or: pip install -e .[downloader])",
        ) from exc
    dest_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        names = zf.getnames()
        for name in names:
            _check_cancel(cancel)
            validate_zip_member(name)
        zf.extractall(path=dest_dir)
    return [p for p in dest_dir.rglob("*") if p.is_file()]


def collect_executables(extracted_root: Path, wanted: Iterable[str]) -> dict[str, Path]:
    """Walk an extraction tree and find the wanted executables (first hit wins)."""
    found: dict[str, Path] = {}
    remaining = set(wanted)
    for candidate in sorted(extracted_root.rglob("*")):
        if not remaining:
            break
        if candidate.is_file() and candidate.suffix.lower() == ".exe":
            stem = candidate.stem.lower()
            if stem in remaining:
                found[stem] = candidate
                remaining.discard(stem)
    return found


# ---------------------------------------------------------------------------
# Remote listing parsing (pure functions, unit-testable)
# ---------------------------------------------------------------------------


def parse_mkvtoolnix_versions(listing_html: str) -> list[str]:
    """Extract version folder names from the mkvtoolnix.download releases listing."""
    versions = _VERSION_DIR_RE.findall(listing_html)
    # Newest first: sort by numeric components descending.
    def version_key(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    return sorted(set(versions), key=version_key, reverse=True)


def choose_mkvtoolnix_archive(listing_html: str) -> str | None:
    """Pick the best Windows archive from a version folder listing.

    Preference: 64-bit > anything else, ``.zip`` > ``.7z`` (zip needs no extra deps).
    """
    archives = _ARCHIVE_RE.findall(listing_html)

    def rank(name: str) -> tuple[int, int, int]:
        lower = name.lower()
        is_zip = lower.endswith(".zip")
        is_64 = "64-bit" in lower
        is_cli = "gui" not in lower
        return (is_64, is_zip, is_cli)

    for archive in sorted(set(archives), key=rank, reverse=True):
        return archive
    return None


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class BinaryDownloader:
    """Downloads and installs missing tools into a local ``bin`` directory."""

    def __init__(
        self,
        bin_dir: Path,
        *,
        client: httpx.Client | None = None,
        status: ToolStatus | None = None,
    ) -> None:
        self._bin_dir = Path(bin_dir)
        self._client = client
        self._owns_client = client is None
        self._status = status

    def __enter__(self) -> BinaryDownloader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT, headers={"User-Agent": "EpisodeMatch/0.1"})
        return self._client

    # -- status ----------------------------------------------------------------

    def current_status(self) -> ToolStatus:
        if self._status is None:
            from episodematch.core.binaries import locate_seconv

            tools = {
                name: locate_tool(name, bin_dir=self._bin_dir, probe=False)
                for name in (*_FFMPEG_TOOLS, *_MKVTOOLS)
            }
            seconv_path = locate_seconv(self._bin_dir)
            from episodematch.core.models import ToolInfo, ToolSource

            tools["seconv"] = (
                ToolInfo(name="seconv", path=seconv_path, source=ToolSource.LOCAL_BIN)
                if seconv_path is not None
                else ToolInfo(name="seconv", source=ToolSource.NOT_FOUND)
            )
            self._status = ToolStatus(tools=tools)
        return self._status

    # -- public entry points ------------------------------------------------------

    def download_missing(
        self,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
        force: Sequence[str] = (),
    ) -> DownloadResult:
        """Download whatever is missing (or listed in ``force``)."""
        status = self.current_status()
        all_tools = (*_FFMPEG_TOOLS, *_MKVTOOLS, *_SECONV_TOOLS)
        missing = [name for name in all_tools if not status.get(name).is_available]
        forced = [name for name in force if name in all_tools]
        todo_ffmpeg = [name for name in forced if name in _FFMPEG_TOOLS] or (
            [name for name in _FFMPEG_TOOLS if name in missing]
        )
        todo_mkv = [name for name in forced if name in _MKVTOOLS] or (
            [name for name in _MKVTOOLS if name in missing]
        )
        todo_seconv = [name for name in forced if name in _SECONV_TOOLS] or (
            [name for name in _SECONV_TOOLS if name in missing]
        )

        installed: list[str] = []
        archive_url = ""

        if todo_ffmpeg:
            got = self.fetch_ffmpeg(wanted=todo_ffmpeg, progress=progress, cancel=cancel)
            installed.extend(got)
            archive_url = archive_url or "ffmpeg"
        if todo_mkv:
            got = self.fetch_mkvtoolnix(wanted=todo_mkv, progress=progress, cancel=cancel)
            installed.extend(got)
            archive_url = archive_url or "mkvtoolnix"
        if todo_seconv:
            got = self.fetch_seconv(progress=progress, cancel=cancel)
            installed.extend(got)
            archive_url = archive_url or "seconv"
        skipped = [name for name in all_tools if name not in installed]
        return DownloadResult(installed=tuple(installed), skipped=tuple(skipped), archive_url=archive_url)

    # -- seconv (SubtitleEdit headless converter) ------------------------------------

    def fetch_seconv(
        self,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
    ) -> list[str]:
        """Download SeConv (SubtitleEdit's headless converter) and install
        ``seconv.exe`` into ``./bin/`` — it powers the PGS fast path."""
        errors: list[str] = []
        for url in _SECONV_URLS:
            try:
                with tempfile.TemporaryDirectory(prefix="em_seconv_") as tmp:
                    tmp_dir = Path(tmp)
                    archive = self._download_to(url, tmp_dir / "seconv.zip", label="seconv", progress=progress, cancel=cancel)
                    safe_extract_zip(archive, tmp_dir / "x", cancel=cancel)
                    installed = self._install_from_tree(tmp_dir / "x", _SECONV_TOOLS, progress=progress)
                if installed:
                    logger.info("seconv installed from %s: %s", url, ", ".join(installed))
                    return installed
                errors.append(f"{url}: no seconv.exe found in archive")
            except BinaryDownloadError:
                raise
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                logger.warning("seconv source failed: %s (%s)", url, exc)
                errors.append(f"{url}: {exc}")
        raise BinaryDownloadError(
            "Could not download seconv from any known source.",
            hint="; ".join(errors),
        )

    # -- ffmpeg --------------------------------------------------------------------

    def fetch_ffmpeg(
        self,
        *,
        wanted: Sequence[str] = _FFMPEG_TOOLS,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
    ) -> list[str]:
        """Download an ffmpeg build and install ``ffmpeg.exe``/``ffprobe.exe``."""
        errors: list[str] = []
        for url in _FF_URLS:
            try:
                with tempfile.TemporaryDirectory(prefix="em_ffmpeg_") as tmp:
                    tmp_dir = Path(tmp)
                    archive = self._download_to(url, tmp_dir / "ffmpeg.zip", label="ffmpeg", progress=progress, cancel=cancel)
                    extract_7z(archive, tmp_dir / "x", cancel=cancel) if archive.suffix == ".7z" else safe_extract_zip(
                        archive, tmp_dir / "x", cancel=cancel
                    )
                    installed = self._install_from_tree(tmp_dir / "x", wanted, progress=progress)
                if installed:
                    logger.info("ffmpeg tools installed from %s: %s", url, ", ".join(installed))
                    return installed
                errors.append(f"{url}: no {list(wanted)} found in archive")
            except BinaryDownloadError:
                raise
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                logger.warning("ffmpeg source failed: %s (%s)", url, exc)
                errors.append(f"{url}: {exc}")
        raise BinaryDownloadError(
            "Could not download ffmpeg from any known mirror.",
            hint="; ".join(errors),
        )

    # -- MKVToolNix ---------------------------------------------------------------------

    def fetch_mkvtoolnix(
        self,
        *,
        wanted: Sequence[str] = _MKVTOOLS,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
    ) -> list[str]:
        """Download MKVToolNix (portable build) and install the CLI tools."""
        errors: list[str] = []
        for archive_url in self._mkvtoolnix_candidates():
            try:
                with tempfile.TemporaryDirectory(prefix="em_mkv_") as tmp:
                    tmp_dir = Path(tmp)
                    archive_name = archive_url.rsplit("/", 1)[-1]
                    archive = self._download_to(archive_url, tmp_dir / archive_name, label="mkvtoolnix", progress=progress, cancel=cancel)
                    if archive.suffix == ".7z":
                        extract_7z(archive, tmp_dir / "x", cancel=cancel)
                    else:
                        safe_extract_zip(archive, tmp_dir / "x", cancel=cancel)
                    installed = self._install_from_tree(tmp_dir / "x", wanted, progress=progress)
                if installed:
                    logger.info("MKVToolNix tools installed from %s: %s", archive_url, ", ".join(installed))
                    return installed
                errors.append(f"{archive_url}: no {list(wanted)} found in archive")
            except BinaryDownloadError:
                raise
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                logger.warning("MKVToolNix source failed: %s (%s)", archive_url, exc)
                errors.append(f"{archive_url}: {exc}")
        raise BinaryDownloadError(
            "Could not download MKVToolNix from any known source.",
            hint="; ".join(errors),
        )

    def _mkvtoolnix_candidates(self) -> list[str]:
        """Build the candidate URL list: scraped newest versions first, pinned fallback.

        Recent MKVToolNix releases ship ``.7z`` only; older ones also had ``.zip``.
        Both spellings are tried per version.
        """
        candidates: list[str] = []
        try:
            response = self.client.get(_MKVTOOLNIX_RELEASES_URL)
            response.raise_for_status()
            for version in parse_mkvtoolnix_versions(response.text)[:3]:
                folder_url = f"{_MKVTOOLNIX_RELEASES_URL}{version}/"
                archive: str | None = None
                try:
                    listing = self.client.get(folder_url)
                    listing.raise_for_status()
                    archive = choose_mkvtoolnix_archive(listing.text)
                except httpx.HTTPError:
                    archive = None
                if archive is None:
                    # Guess the common names when the sub-listing is unavailable.
                    candidates.append(f"{folder_url}mkvtoolnix-64-bit-{version}.7z")
                    candidates.append(f"{folder_url}mkvtoolnix-64-bit-{version}.zip")
                else:
                    candidates.append(folder_url + archive)
        except httpx.HTTPError as exc:
            logger.warning("MKVToolNix release listing unavailable: %s", exc)
        for suffix in (".7z", ".zip"):
            candidates.append(
                f"{_MKVTOOLNIX_RELEASES_URL}{_MKVTOOLNIX_PINNED_VERSION}/mkvtoolnix-64-bit-{_MKVTOOLNIX_PINNED_VERSION}{suffix}"
            )
        # De-duplicate, preserve order.
        seen: set[str] = set()
        return [url for url in candidates if not (url in seen or seen.add(url))]

    # -- plumbing -------------------------------------------------------------------------

    def _download_to(
        self,
        url: str,
        target: Path,
        *,
        label: str,
        progress: ProgressCallback | None,
        cancel: CancelCheck | None,
    ) -> Path:
        """Stream a URL to ``target`` with progress reporting and cancellation."""
        _check_cancel(cancel)
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from %s", label, url)
        try:
            with self.client.stream("GET", url) as response:
                response.raise_for_status()
                total_header = response.headers.get("content-length")
                total = int(total_header) if total_header and total_header.isdigit() else None
                if total is not None and total < _MIN_ARCHIVE_BYTES:
                    raise BinaryDownloadError(f"Archive at {url} is suspiciously small ({total} bytes).")
                done = 0
                with open(target, "wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                        _check_cancel(cancel)
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(label, done, total)
        except httpx.HTTPError as exc:
            raise BinaryDownloadError(f"Download failed for {url}: {exc}", hint="Check your internet connection.") from exc
        if target.stat().st_size < _MIN_ARCHIVE_BYTES:
            raise BinaryDownloadError(f"Downloaded archive from {url} is too small to be valid.")
        return target

    def _install_from_tree(
        self,
        extracted_root: Path,
        wanted: Sequence[str],
        *,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        """Copy wanted executables from the extraction tree into ./bin/ and verify them."""
        found = collect_executables(extracted_root, wanted)
        installed: list[str] = []
        self._bin_dir.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            source = found.get(name)
            if source is None:
                logger.warning("Tool %s not present in downloaded archive.", name)
                continue
            destination = self._bin_dir / _exe_name(name)
            shutil.copy2(source, destination)
            version = probe_version(destination)
            if version is None:
                logger.warning("Installed %s but the version probe failed; keeping the file.", name)
            else:
                logger.info("Installed %s (%s) into %s", name, version, destination)
            installed.append(name)
            if progress is not None:
                progress(f"{name} installed", 1, 1)
        return installed
