"""Tests for binaries locator, downloader internals, config/env handling and the CLI."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import httpx
import pytest

from episodematch.core.bin_downloader import (
    BinaryDownloader,
    choose_mkvtoolnix_archive,
    collect_executables,
    parse_mkvtoolnix_versions,
    safe_extract_zip,
    validate_zip_member,
)
from episodematch.core.binaries import ToolResolver, locate_tool, probe_version
from episodematch.core.config import AppConfig, apply_env_overrides
from episodematch.core.exceptions import BinaryDownloadError, BinaryNotFoundError, ConfigError, EpisodeMatchError
from episodematch.core.database import CacheDatabase
from episodematch.main import main

# ---------------------------------------------------------------------------
# Zip safety
# ---------------------------------------------------------------------------


class TestZipSafety:
    def test_valid_names_pass(self) -> None:
        validate_zip_member("ffmpeg/bin/ffmpeg.exe")
        validate_zip_member("plain.txt")

    def test_parent_traversal_rejected(self) -> None:
        with pytest.raises(BinaryDownloadError):
            validate_zip_member("../evil.exe")
        with pytest.raises(BinaryDownloadError):
            validate_zip_member("a/../../evil.exe")

    def test_absolute_paths_rejected(self) -> None:
        with pytest.raises(BinaryDownloadError):
            validate_zip_member("C:/Windows/evil.exe")
        with pytest.raises(BinaryDownloadError):
            validate_zip_member("/abs/evil.exe")

    def test_safe_extract_zip(self, tmp_path: Path) -> None:
        archive = tmp_path / "tools.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ffmpeg-7/bin/ffmpeg.exe", b"MZ fake")
            zf.writestr("ffmpeg-7/bin/ffprobe.exe", b"MZ fake")
            zf.writestr("readme.txt", b"hello")
        dest = tmp_path / "extracted"
        extracted = safe_extract_zip(archive, dest)
        assert (dest / "ffmpeg-7" / "bin" / "ffmpeg.exe").read_bytes() == b"MZ fake"
        assert len(extracted) == 3

    def test_safe_extract_rejects_zip_slip(self, tmp_path: Path) -> None:
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../evil.exe", b"bad")
        with pytest.raises(BinaryDownloadError):
            safe_extract_zip(archive, tmp_path / "out")

    def test_collect_executables_flattens(self, tmp_path: Path) -> None:
        nested = tmp_path / "x" / "bin"
        nested.mkdir(parents=True)
        (nested / "ffmpeg.exe").write_bytes(b"a")
        (nested / "ffprobe.exe").write_bytes(b"b")
        (nested / "unrelated.exe").write_bytes(b"c")
        found = collect_executables(tmp_path, ["ffmpeg", "ffprobe"])
        assert set(found) == {"ffmpeg", "ffprobe"}
        assert found["ffmpeg"].name == "ffmpeg.exe"


# ---------------------------------------------------------------------------
# MKVToolNix release listing parsing
# ---------------------------------------------------------------------------


class TestMkvToolNixListing:
    # Real-world autoindex format (as of mkvtoolnix.download): "./99.0/" style hrefs.
    LISTING = """
    <a href="./82.0/">82.0/</a>
    <a href="./84.0/">84.0/</a>
    <a href="./83.0.1/">83.0.1/</a>
    <a href="86.0/">86.0/</a>
    <a href="./notes/">notes/</a>
    """

    def test_versions_sorted_newest_first(self) -> None:
        versions = parse_mkvtoolnix_versions(self.LISTING)
        assert versions[0] == "86.0"
        assert versions == sorted(versions, key=lambda v: [int(p) for p in v.split(".")], reverse=True)

    def test_relative_hrefs_also_matched(self) -> None:
        versions = parse_mkvtoolnix_versions('<a href="12.0/">12.0/</a><a href="/windows/releases/13.0/">')
        assert "12.0" in versions and "13.0" in versions

    def test_archive_preference(self) -> None:
        listing = (
            '<a href="./mkvtoolnix-32-bit-84.0.7z">32</a>'
            '<a href="./mkvtoolnix-64-bit-84.0.7z">7z</a>'
            '<a href="mkvtoolnix-64-bit-84.0.zip">zip</a>'
            '<a href="./mkvtoolnix-64-bit-gui-84.0.7z">gui</a>'
        )
        assert choose_mkvtoolnix_archive(listing) == "mkvtoolnix-64-bit-84.0.zip"

    def test_real_world_7z_only_listing(self) -> None:
        listing = (
            '<a href="./mkvtoolnix-64-bit-90.0-setup.exe">setup</a>'
            '<a href="./mkvtoolnix-64-bit-90.0.7z">7z</a>'
        )
        assert choose_mkvtoolnix_archive(listing) == "mkvtoolnix-64-bit-90.0.7z"

    def test_no_archives(self) -> None:
        assert choose_mkvtoolnix_archive("<a href='./82.0/'>x</a>") is None


# ---------------------------------------------------------------------------
# Tool locator
# ---------------------------------------------------------------------------


class TestLocateTool:
    def test_prefers_local_bin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ffmpeg.exe").write_bytes(b"MZ")
        monkeypatch.setattr("episodematch.core.binaries.probe_version", lambda *_a, **_k: "7.0")
        info = locate_tool("ffmpeg", bin_dir=bin_dir)
        assert info.is_available
        assert info.source.value == "local_bin"
        assert info.version == "7.0"

    def test_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        info = locate_tool("mkvmerge", bin_dir=tmp_path / "empty")
        assert not info.is_available
        assert info.source.value == "not_found"

    def test_system_path_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = tmp_path / "elsewhere"
        fake.mkdir()
        (fake / "ffmpeg.exe").write_bytes(b"MZ")
        monkeypatch.setattr("shutil.which", lambda name: str(fake / "ffmpeg.exe") if name == "ffmpeg" else None)
        monkeypatch.setattr("episodematch.core.binaries.probe_version", lambda *_a, **_k: "6.9")
        info = locate_tool("ffmpeg", bin_dir=tmp_path / "bin")
        assert info.source.value == "system_path"

    def test_resolver_caches_and_invalidates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        resolver = ToolResolver(tmp_path / "bin", probe=False)
        assert resolver.find("ffmpeg").is_available is False
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "ffmpeg.exe").write_bytes(b"MZ")
        assert resolver.find("ffmpeg").is_available is False  # cached
        resolver.invalidate()
        assert resolver.find("ffmpeg").is_available is True

    def test_seconv_in_tool_status_via_subtitle_edit_folder(self, tmp_path: Path) -> None:
        """seconv may live in bin/SubtitleEdit/ — locate_all_tools must find it
        there (it powers the PGS fast path and shows in Settings)."""
        from episodematch.core.binaries import locate_all_tools

        se_dir = tmp_path / "bin" / "SubtitleEdit"
        se_dir.mkdir(parents=True)
        (se_dir / "seconv.exe").write_bytes(b"MZ")
        status = locate_all_tools(bin_dir=tmp_path / "bin", probe=False)
        info = status.get("seconv")
        assert info.is_available and info.path == se_dir / "seconv.exe"
        assert info.source.value == "local_bin"

    def test_seconv_missing_reports_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        from episodematch.core.binaries import locate_all_tools

        status = locate_all_tools(bin_dir=tmp_path / "bin", probe=False)
        assert not status.get("seconv").is_available

    def test_require_tool_raises_with_hint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        resolver = ToolResolver(tmp_path / "bin", probe=False)
        with pytest.raises(BinaryNotFoundError) as excinfo:
            resolver.tool_path("ffprobe")
        assert "fetch-tools" in str(excinfo.value)


class TestVersionProbe:
    def test_parses_ffmpeg_version(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        exe = tmp_path / "ffmpeg.exe"
        exe.write_bytes(b"MZ")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="ffmpeg version 7.0.2 Copyright\nbuilt with gcc", stderr=""),
        )
        assert probe_version(exe) == "7.0.2"

    def test_parses_mkvmerge_version(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        exe = tmp_path / "mkvmerge.exe"
        exe.write_bytes(b"MZ")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="mkvmerge v84.0 ('Name') 64-bit", stderr=""),
        )
        assert probe_version(exe) == "84.0"

    def test_unrunnable_binary_returns_none(self, tmp_path: Path) -> None:
        assert probe_version(tmp_path / "does_not_exist.exe") is None


# ---------------------------------------------------------------------------
# Downloader end-to-end with a mocked transport
# ---------------------------------------------------------------------------


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


class TestBinaryDownloaderMocked:
    @pytest.fixture(autouse=True)
    def _tiny_archive_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Test archives are tiny; relax the real-world 1 MiB sanity threshold.
        monkeypatch.setattr("episodematch.core.bin_downloader._MIN_ARCHIVE_BYTES", 64)

    def test_fetch_ffmpeg_installs_and_verifies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive = _zip_bytes(
            {
                "ffmpeg-7.0-essentials_build/bin/ffmpeg.exe": b"MZ ffmpeg",
                "ffmpeg-7.0-essentials_build/bin/ffprobe.exe": b"MZ ffprobe",
            }
        )
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return httpx.Response(200, content=archive, headers={"content-length": str(len(archive))})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr("episodematch.core.bin_downloader.probe_version", lambda path, **k: "7.0")
        downloader = BinaryDownloader(tmp_path / "bin", client=httpx.Client(transport=transport))
        result = downloader.fetch_ffmpeg()
        assert set(result) == {"ffmpeg", "ffprobe"}
        assert (tmp_path / "bin" / "ffmpeg.exe").is_file()
        assert (tmp_path / "bin" / "ffprobe.exe").is_file()

    def test_progress_callback_called(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        archive = _zip_bytes({"build/bin/ffmpeg.exe": b"MZ ffmpeg"})
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=archive, headers={"content-length": str(len(archive))}))
        monkeypatch.setattr("episodematch.core.bin_downloader.probe_version", lambda path, **k: "7.0")
        progress_events: list[tuple[str, int, int | None]] = []
        downloader = BinaryDownloader(tmp_path / "bin", client=httpx.Client(transport=transport))
        downloader.fetch_ffmpeg(progress=lambda label, done, total: progress_events.append((label, done, total)))
        # Streaming progress must report at least the full archive size at some point.
        assert any(done >= len(archive) for _label, done, _total in progress_events)

    def test_http_error_raises_binary_download_error(self, tmp_path: Path) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(404, content=b"nope"))
        downloader = BinaryDownloader(tmp_path / "bin", client=httpx.Client(transport=transport))
        with pytest.raises(BinaryDownloadError):
            downloader.fetch_ffmpeg()

    def test_cancel_stops_download(self, tmp_path: Path) -> None:
        archive = _zip_bytes({"build/bin/ffmpeg.exe": b"MZ" * 1000})
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=archive, headers={"content-length": str(len(archive))}))
        downloader = BinaryDownloader(tmp_path / "bin", client=httpx.Client(transport=transport))
        with pytest.raises(BinaryDownloadError) as excinfo:
            downloader.fetch_ffmpeg(cancel=lambda: True)
        assert "cancelled" in str(excinfo.value)

    def test_mkvtoolnix_candidates_include_pinned_fallback(self, tmp_path: Path) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b'<a href="84.0/">84.0/</a>',
                headers={"content-type": "text/html"},
            )
        )
        downloader = BinaryDownloader(tmp_path / "bin", client=httpx.Client(transport=transport))
        candidates = downloader._mkvtoolnix_candidates()
        assert candidates, "expected at least the pinned fallback"
        assert "mkvtoolnix-64-bit" in candidates[0]
        assert candidates[-1].endswith(".zip")

    def test_fetch_seconv_installs_into_bin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SeConv ships as a standalone Windows asset on the SubtitleEdit release —
        the downloader must install it into ./bin/ (PGS fast path)."""
        archive = _zip_bytes({"seconv.exe": b"MZ seconv"})
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=archive, headers={"content-length": str(len(archive))})
        )
        monkeypatch.setattr("episodematch.core.bin_downloader.probe_version", lambda path, **k: "5.4.0")
        downloader = BinaryDownloader(tmp_path / "bin", client=httpx.Client(transport=transport))
        installed = downloader.fetch_seconv()
        assert installed == ["seconv"]
        assert (tmp_path / "bin" / "seconv.exe").is_file()

    def test_download_missing_includes_seconv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """download_missing must treat a missing seconv like any other missing tool."""
        from episodematch.core.bin_downloader import BinaryDownloader, DownloadResult

        archive = _zip_bytes({"seconv.exe": b"MZ seconv"})
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=archive, headers={"content-length": str(len(archive))})
        )
        monkeypatch.setattr("episodematch.core.bin_downloader.probe_version", lambda path, **k: "5.4.0")

        # ffmpeg/mkvtoolnix present, only seconv missing:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name in ("ffmpeg", "ffprobe", "mkvmerge", "mkvextract", "mkvpropedit"):
            (bin_dir / f"{name}.exe").write_bytes(b"MZ")

        downloader = BinaryDownloader(bin_dir, client=httpx.Client(transport=transport))
        result: DownloadResult = downloader.download_missing()
        assert "seconv" in result.installed
        assert "ffmpeg" in result.skipped  # present -> not re-downloaded
        assert (bin_dir / "seconv.exe").is_file()


# ---------------------------------------------------------------------------
# Config: env overrides + persistence
# ---------------------------------------------------------------------------


class TestConfig:
    def test_env_overrides_llm_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EPISODEMATCH_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("EPISODEMATCH_LLM_BASE_URL", "https://api.example.com/v4/")
        monkeypatch.setenv("EPISODEMATCH_LLM_MODEL", "glm-4.5-air")
        monkeypatch.setenv("EPISODEMATCH_TMDB_API_KEY", "tmdb123")
        config = apply_env_overrides(AppConfig())
        assert config.llm_api_key == "sk-test"
        assert config.llm_base_url == "https://api.example.com/v4"
        assert config.llm_model == "glm-4.5-air"
        assert config.tmdb_api_key == "tmdb123"

    def test_defaults(self) -> None:
        config = AppConfig()
        assert config.llm_base_url == "https://open.bigmodel.cn/api/paas/v4"
        assert config.llm_model == "glm-4-flash"
        assert config.naming.episode_template.startswith("{series}/Season")

    def test_invalid_llm_url_rejected(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AppConfig(llm_base_url="not-a-url")

    def test_store_roundtrip(self, tmp_path: Path) -> None:
        from episodematch.core.config import ConfigStore

        path = tmp_path / "data" / "config.json"
        store = ConfigStore(path)
        config = AppConfig(tmdb_api_key="abc123", av1_preset=8)
        written = store.save(config)
        assert written == path
        loaded = store.load()
        assert loaded.tmdb_api_key == "abc123"
        assert loaded.av1_preset == 8

    def test_store_invalid_json_raises(self, tmp_path: Path) -> None:
        from episodematch.core.config import ConfigStore

        path = tmp_path / "config.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(ConfigError):
            ConfigStore(path).load()

    def test_masked_keys(self) -> None:
        config = AppConfig(tmdb_api_key="abcdefghijk")
        masked = config.masked()
        assert "abcdefghijk" not in masked["tmdb_api_key"]
        assert masked["llm_api_key"] == ""

    def test_ocr_backend_setting_default_auto(self) -> None:
        """PILLAR4_PLAN batch 4 (RapidOCR evaluation): the OCR backend for the
        visual pillar is configurable — "auto" keeps the WinRT-first chain."""
        config = AppConfig()
        assert config.extraction.ocr_backend == "auto"

    def test_masked_keys_cover_subdl(self) -> None:
        """Audit F1: the SubDL API key must be masked like the other keys."""
        config = AppConfig(
            subdl_api_key="abcdefghijk",
            tmdb_api_key="tmdb-long-key-value",
            opensubtitles_api_key="osdt-long-key",
            llm_api_key="llm-long-key-1234567890",
        )
        masked = config.masked()
        assert "abcdefghijk" not in json.dumps(masked)
        assert "abcdefghijk" not in str(masked.get("subdl_api_key"))


class TestPathResolution:

    def test_find_project_root_finds_checkout(self) -> None:
        from episodematch.core.config import find_project_root

        root = find_project_root(Path("src") / "episodematch" / "core" / "config.py")
        assert root is not None and (root / "pyproject.toml").is_file()

    def test_dev_anchor_outranks_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Development (venv/checkout): the project root's bin/data win even when
        the current directory contains same-named folders — accidental CWDs must
        not fork configuration and caches."""
        from episodematch.core.config import AppConfig, find_project_root

        (tmp_path / "bin").mkdir()
        monkeypatch.chdir(tmp_path)
        root = find_project_root()
        assert root is not None
        resolved = AppConfig().resolve_path("bin")
        assert resolved == root / "bin"

    def test_frozen_exe_anchors_at_executable_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audit F2: frozen builds anchor bin/data at the EXE directory, not the
        launch CWD — a shortcut started from anywhere must not fork the tools."""
        import sys

        from episodematch.core.config import AppConfig

        exe_dir = tmp_path / "EpisodeMatch"
        exe_dir.mkdir()
        (exe_dir / "bin").mkdir()
        foreign_cwd = tmp_path / "elsewhere"
        foreign_cwd.mkdir()
        (foreign_cwd / "bin").mkdir()  # a same-named bin/ in the CWD must NOT win
        monkeypatch.chdir(foreign_cwd)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe_dir / "EpisodeMatch.exe"))
        assert AppConfig().resolve_path("bin") == exe_dir / "bin"

    def test_frozen_exe_missing_bin_anchors_at_exe_dir_anyway(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from episodematch.core.config import AppConfig

        exe_dir = tmp_path / "EpisodeMatch"
        exe_dir.mkdir()  # no bin/ next to the exe yet
        foreign_cwd = tmp_path / "elsewhere"
        foreign_cwd.mkdir()
        (foreign_cwd / "bin").mkdir()  # CWD fallback would silently fork here
        monkeypatch.chdir(foreign_cwd)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe_dir / "EpisodeMatch.exe"))
        assert AppConfig().resolve_path("bin") == exe_dir / "bin"

    def test_dev_anchor_without_venv_still_finds_checkout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CI runners install the interpreter into a hosted cache (no pyproject
        marker around sys.prefix) while the editable install still runs from
        the checkout — the anchor must fall back to the module's own repo root."""
        import sys

        from episodematch.core import config as config_module

        fake_prefix = tmp_path / "hosted" / "python312"  # no pyproject marker anywhere above
        fake_prefix.mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", str(fake_prefix))
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        root = config_module._dev_anchor()
        assert root is not None
        assert (root / "pyproject.toml").is_file()

    def test_resolve_path_falls_back_to_project_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Started from a foreign CWD (double-clicked shim), bin/data resolve to
        the checkout instead of being created in the wrong place."""
        from episodematch.core.config import AppConfig, find_project_root

        monkeypatch.chdir(tmp_path)  # tmp has no bin/
        config = AppConfig()
        root = find_project_root()
        assert root is not None
        assert config.resolve_path("bin") == root / "bin"
        assert config.resolved_bin_dir == root / "bin"
        assert config.database_path == root / "data" / "episodematch.db"

    def test_config_store_default_follows_root_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from episodematch.core.config import ConfigStore, find_project_root

        monkeypatch.chdir(tmp_path)
        root = find_project_root()
        assert root is not None
        assert ConfigStore().path == root / "data" / "config.json"
        # ... unless an explicit path is given (tests use this constantly)
        assert ConfigStore(tmp_path / "custom.json").path == tmp_path / "custom.json"

    def test_absolute_paths_pass_through(self) -> None:
        from episodematch.core.config import AppConfig

        assert AppConfig().resolve_path(Path("X:/anywhere/bin")) == Path("X:/anywhere/bin")


class TestCacheDatabase:
    def test_json_cache_roundtrip_with_ttl(self, tmp_path: Path) -> None:
        with CacheDatabase(tmp_path / "test.db") as db:
            db.put_json("key", {"a": 1}, ttl_seconds=60)
            assert db.get_json("key") == {"a": 1}
            db.put_json("key", {"a": 2}, ttl_seconds=-1)  # already expired
            assert db.get_json("key") is None

    def test_model_cache(self, tmp_path: Path) -> None:
        from episodematch.core.models import ToolInfo

        with CacheDatabase(tmp_path / "test.db") as db:
            db.put_model("tool", ToolInfo(name="ffmpeg", version="7"))
            loaded = db.get_model("tool", ToolInfo)
            assert loaded is not None and loaded.version == "7"

    def test_scan_cache(self, tmp_path: Path) -> None:
        with CacheDatabase(tmp_path / "test.db") as db:
            db.put_mkv_scan(Path("x.mkv"), 100, 1.0, {"duration": 5.0})
            assert db.get_mkv_scan(Path("x.mkv"), 100, 1.0) == {"duration": 5.0}
            assert db.get_mkv_scan(Path("x.mkv"), 101, 1.0) is None

    def test_disc_cache_search(self, tmp_path: Path) -> None:
        with CacheDatabase(tmp_path / "test.db") as db:
            db.put_disc("k1", "Breaking Bad", {"upc": "1"})
            hits = db.find_discs_by_series("breaking")
            assert hits == [{"upc": "1"}]

    def test_purge_expired(self, tmp_path: Path) -> None:
        with CacheDatabase(tmp_path / "test.db") as db:
            db.put_json("gone", [1], ttl_seconds=-10)
            db.put_json("stay", [2], ttl_seconds=60)
            assert db.purge_expired() == 1
            assert db.get_json("stay") == [2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_version_command(self, capsys: pytest.CaptureFixture) -> None:
        assert main(["version"]) == 0
        assert "EpisodeMatch" in capsys.readouterr().out

    def test_check_env_runs_without_tools(self, config: AppConfig, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("episodematch.main.ConfigStore.load", lambda self: config)
        monkeypatch.setattr("shutil.which", lambda name: None)
        # Force the "no OCR backend" branch even on machines with real OCR installed.
        monkeypatch.setattr("episodematch.extractors.ocr_engine._BACKEND_FACTORIES", ())
        assert main(["check-env"]) == 0
        output = capsys.readouterr().out
        assert "ffmpeg" in output and "not found" in output
        # With the backend factories cleared, the OCR line must show unavailable.
        assert "✗ ocr" in output

    def test_match_missing_folder_errors(self, config: AppConfig, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("episodematch.main.ConfigStore.load", lambda self: config)
        assert main(["match", str(config.resolved_data_dir / "nope"), "--series", "X", "--season", "1"]) == 1

    def test_match_folder_without_mkvs_errors(self, config: AppConfig, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("episodematch.main.ConfigStore.load", lambda self: config)
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["match", str(empty), "--series", "X", "--season", "1"]) == 1

    def test_json_report_shape(self, config: AppConfig, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: probe-less fast scan is impossible without tools -> the CLI
        must exit cleanly with an error message (JSON mode still exits 1)."""
        monkeypatch.setattr("episodematch.main.ConfigStore.load", lambda self: config)
        folder = tmp_path / "rips"
        folder.mkdir()
        (folder / "title_t00.mkv").write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
        code = main(["match", str(folder), "--series", "X", "--season", "1", "--json", "--no-discdb"])
        assert code == 1  # ffprobe missing -> clean failure, no traceback

class TestReadOnlyDatabase:
    """Audit F6: read_only=True must actually enforce read-only access."""

    def test_read_only_connection_cannot_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.sqlite3"
        writer = CacheDatabase(db_path)
        writer.put_json("k", {"a": 1})
        writer.close()

        ro = CacheDatabase(db_path, read_only=True)
        try:
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.put_json("k2", {"b": 2})
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.delete("k")
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.purge_expired()
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.put_mkv_scan(Path("x.mkv"), 1, 1, {})
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.put_disc("cache-key", "series", {})
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.put_osdb_subtitle("sub", {}, None)
            with pytest.raises(EpisodeMatchError, match="read-only"):
                ro.vacuum()
            # reads keep working and the data is intact
            assert ro.get_json("k") == {"a": 1}
        finally:
            ro.close()

    def test_read_only_sqlite_layer_rejects_writes(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "cache.sqlite3"
        CacheDatabase(db_path).close()
        ro = CacheDatabase(db_path, read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                ro._conn.execute("CREATE TABLE should_fail (x INT)")
        finally:
            ro.close()

class TestDebugVisualCommand:
    """Pillar-4 batch 1: episodematch debug-visual exposes raw OCR per frame,
    optional potential hits against the TMDB guide, and --json output."""

    @staticmethod
    def _patched_env(tmp_path: Path, config: AppConfig, monkeypatch: pytest.MonkeyPatch, hits: list) -> Path:
        from episodematch.core.models import MkvFile

        mkv = tmp_path / "title_t00.mkv"
        mkv.write_bytes(b"x")

        class FakeScanner:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                self.kwargs = kwargs
                self.last_timings = {"scene": 1.2, "extract": 0.8, "ocr": 2.5}

            def debug_scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                return hits

        class FakeProbe:
            def probe(self, path, **kwargs):  # noqa: ANN001, ANN003
                return MkvFile(path=Path(path), size_bytes=1, duration_seconds=2400.0)

        monkeypatch.setattr("episodematch.main.FrameScanner", FakeScanner)
        monkeypatch.setattr("episodematch.main.FfprobeReader", lambda *args, **kwargs: FakeProbe())
        monkeypatch.setattr("episodematch.main.ConfigStore.load", lambda self: config)
        return mkv

    def test_debug_visual_prints_frames(self, tmp_path: Path, config: AppConfig, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        from episodematch.core.models import TitleCardHit, TitleCardKind

        mkv = self._patched_env(
            tmp_path,
            config,
            monkeypatch,
            [
                TitleCardHit(timestamp_seconds=61.0, kind=TitleCardKind.CREDITS, raw_lines=["Directed by Michelle MacLaren"], directors=["Michelle MacLaren"], ocr_confidence=0.9),
                TitleCardHit(timestamp_seconds=240.0, kind=TitleCardKind.UNKNOWN, raw_lines=[], ocr_confidence=None),
            ],
        )
        from episodematch.main import main

        assert main(["debug-visual", str(mkv)]) == 0
        out = capsys.readouterr().out
        assert "01:01" in out  # MM:SS timestamp rendering
        assert "Directed by Michelle MacLaren" in out
        assert "unknown" in out.lower()

    def test_debug_visual_json_output(self, tmp_path: Path, config: AppConfig, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        import json as _json

        from episodematch.core.models import TitleCardHit, TitleCardKind

        mkv = self._patched_env(
            tmp_path,
            config,
            monkeypatch,
            [
                TitleCardHit(timestamp_seconds=61.0, kind=TitleCardKind.CREDITS, raw_lines=["Directed by Michelle MacLaren"], directors=["Michelle MacLaren"], ocr_confidence=0.9),
                TitleCardHit(timestamp_seconds=240.0, kind=TitleCardKind.UNKNOWN, raw_lines=[], ocr_confidence=None),
            ],
        )
        from episodematch.main import main

        assert main(["debug-visual", str(mkv), "--json"]) == 0
        payload = _json.loads(capsys.readouterr().out)
        assert len(payload["frames"]) == 2
        assert payload["frames"][0]["timestamp_seconds"] == 61.0
        assert payload["frames"][0]["kind"] == "credits"
        assert payload["frames"][1]["raw_lines"] == []

    def test_debug_visual_with_series_shows_potential_hits(self, tmp_path: Path, config: AppConfig, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        from episodematch.core.models import (
            EpisodeInfo,
            GuestStar,
            SeriesInfo,
            SeasonGuide,
            TitleCardHit,
            TitleCardKind,
        )

        mkv = self._patched_env(
            tmp_path,
            config,
            monkeypatch,
            [
                TitleCardHit(
                    timestamp_seconds=61.0,
                    kind=TitleCardKind.CREDITS,
                    raw_lines=["Directed by Michelle MacLaren"],
                    directors=["Michelle MacLaren"],
                    guest_stars=["Bryan Cranston"],
                    title_candidates=["Pilot"],
                    ocr_confidence=0.9,
                ),
                TitleCardHit(timestamp_seconds=240.0, kind=TitleCardKind.UNKNOWN, raw_lines=[]),
            ],
        )
        series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id="tt1", first_air_year=2020)
        guide = SeasonGuide(
            series=series,
            season=1,
            episodes=[
                EpisodeInfo(
                    season=1,
                    episode=1,
                    title="Pilot",
                    plot="p",
                    directors=["Michelle MacLaren"],
                    guest_stars=[GuestStar(name="Bryan Cranston")],
                ),
                EpisodeInfo(season=1, episode=2, title="Second", plot="p"),
            ],
        )

        class FakeTmdb:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

            def find_series(self, query: str, **kwargs):  # noqa: ANN001, ANN202
                return series

            def season_guide(self, series_info, season):  # noqa: ANN001, ANN202
                return guide

        monkeypatch.setattr("episodematch.main.TmdbClient", FakeTmdb)
        from episodematch.main import main

        assert main(["debug-visual", str(mkv), "--series", "Test Show", "--season", "1"]) == 0
        out = capsys.readouterr().out
        assert "S01E01" in out  # potential hits reference the matching episode
        assert "Michelle MacLaren" in out
        assert "Pilot" in out

    def test_debug_visual_shows_series_cast_hints(self, tmp_path: Path, config: AppConfig, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """A series-cast name on a card ("PICARDO" last-name-only) is shown as
        a series-confirmation hint — no episode score."""
        from episodematch.core.models import (
            GuestStar,
            MkvFile,
            SeriesInfo,
            SeasonGuide,
            EpisodeInfo,
            TitleCardHit,
            TitleCardKind,
        )

        mkv = tmp_path / "title_t00.mkv"
        mkv.write_bytes(b"x")
        hits = [
            TitleCardHit(
                timestamp_seconds=212.0,
                kind=TitleCardKind.CREDITS,
                raw_lines=["PICARDO as The Doctor"],
                guest_stars=["PICARDO"],
                ocr_confidence=0.9,
            ),
            TitleCardHit(timestamp_seconds=240.0, kind=TitleCardKind.UNKNOWN, raw_lines=[]),
        ]

        class FakeScanner:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                self.last_timings = {"scene": 1.0, "extract": 1.0, "ocr": 1.0}

            def debug_scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                return hits

        class FakeProbe:
            def probe(self, path, **kwargs):  # noqa: ANN001, ANN003
                return MkvFile(path=Path(path), size_bytes=1, duration_seconds=2400.0)

        monkeypatch.setattr("episodematch.main.FrameScanner", FakeScanner)
        monkeypatch.setattr("episodematch.main.FfprobeReader", lambda *args, **kwargs: FakeProbe())

        series = SeriesInfo(name="Voyager", tmdb_id=46533, imdb_id="tt1", first_air_year=2020)
        guide = SeasonGuide(
            series=series,
            season=2,
            episodes=[EpisodeInfo(season=2, episode=1, title="The 37's", plot="p")],
            cast=["Robert Picardo", "Kate Mulgrew"],
        )

        class FakeTmdb:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

            def find_series(self, query: str, **kwargs):  # noqa: ANN001, ANN202
                return series

            def season_guide(self, series_info, season):  # noqa: ANN001, ANN202
                return guide

        monkeypatch.setattr("episodematch.main.TmdbClient", FakeTmdb)
        monkeypatch.setattr("episodematch.main.ConfigStore.load", lambda self: config)
        from episodematch.main import main

        assert main(["debug-visual", str(mkv), "--series", "Voyager", "--season", "2"]) == 0
        out = capsys.readouterr().out
        assert "series cast" in out and "Robert Picardo" in out
        assert "confirms the series" in out
