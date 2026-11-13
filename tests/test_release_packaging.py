"""Tests for the release packaging license guard (scripts/build_exe.py)."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_exe.py"
_spec = importlib.util.spec_from_file_location("build_exe", MODULE_PATH)
build_exe = importlib.util.module_from_spec(_spec)
sys.modules["build_exe"] = build_exe
_spec.loader.exec_module(build_exe)


class TestForbiddenToolGuard:
    def test_detects_gpl_tools_in_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "_internal").mkdir()
        (tmp_path / "EpisodeMatch.exe").write_bytes(b"MZ")
        (tmp_path / "_internal" / "ffmpeg.exe").write_bytes(b"GPL")
        (tmp_path / "_internal" / "seconv.exe").write_bytes(b"GPL")
        found = build_exe.find_forbidden_tools(tmp_path)
        assert [p.name for p in found] == ["ffmpeg.exe", "seconv.exe"]

    def test_clean_bundle_passes(self, tmp_path: Path) -> None:
        (tmp_path / "_internal").mkdir()
        (tmp_path / "EpisodeMatch.exe").write_bytes(b"MZ")
        (tmp_path / "_internal" / "python312.dll").write_bytes(b"dll")
        assert build_exe.find_forbidden_tools(tmp_path) == []

    def test_name_is_case_insensitive(self, tmp_path: Path) -> None:
        (tmp_path / "FFMPEG.EXE").write_bytes(b"GPL")
        assert len(build_exe.find_forbidden_tools(tmp_path)) == 1

    def test_similar_names_are_allowed(self, tmp_path: Path) -> None:
        # e.g. "my_ffmpeg_tool.exe" is NOT one of the exact forbidden names
        (tmp_path / "my_ffmpeg_tool.exe").write_bytes(b"x")
        (tmp_path / "ffmpeg_wrapper.dll").write_bytes(b"x")
        assert build_exe.find_forbidden_tools(tmp_path) == []


class TestUnknownTopLevelExecutableGuard:
    """Audit F3: the guard must also flag executables that are not one of the
    six known GPL tool names — any stray EXE in a bundle is suspicious."""

    def test_unknown_top_level_exe_is_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "EpisodeMatch.exe").write_bytes(b"MZ")
        (tmp_path / "stray_tool.exe").write_bytes(b"MZ")
        found = build_exe.find_forbidden_tools(tmp_path, allowed_top_level={"EpisodeMatch.exe"})
        assert [p.name for p in found] == ["stray_tool.exe"]

    def test_bundle_entry_exe_is_not_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "episodematch.exe").write_bytes(b"MZ")
        (tmp_path / "_internal").mkdir()
        assert build_exe.find_forbidden_tools(tmp_path, allowed_top_level={"episodematch.exe"}) == []

    def test_without_allowed_set_behaves_like_before(self, tmp_path: Path) -> None:
        # legacy callers (no allowed_top_level) keep the exact-name-only behavior
        (tmp_path / "stray_tool.exe").write_bytes(b"MZ")
        assert build_exe.find_forbidden_tools(tmp_path) == []


class TestBuildReleaseZip:
    def _fake_bundles(self, tmp_path: Path) -> tuple[Path, Path]:
        cli = tmp_path / "dist" / "cli" / "episodematch"
        gui = tmp_path / "dist" / "gui" / "EpisodeMatch"
        (cli / "_internal").mkdir(parents=True)
        (gui / "_internal").mkdir(parents=True)
        (cli / "episodematch.exe").write_bytes(b"MZ cli")
        (cli / "_internal" / "python312.dll").write_bytes(b"dll")
        (gui / "EpisodeMatch.exe").write_bytes(b"MZ gui")
        (gui / "_internal" / "Qt6Core.dll").write_bytes(b"dll")
        return cli, gui

    def test_zip_contents_and_license_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cli, gui = self._fake_bundles(tmp_path)
        monkeypatch.setattr(build_exe, "PROJECT_ROOT", tmp_path)
        # minimal packaging files in the fake project root
        (tmp_path / "packaging").mkdir()
        (tmp_path / "packaging" / "README-FIRST.txt").write_text("beta note", encoding="utf-8")
        (tmp_path / "packaging" / "THIRD_PARTY_LICENSES.txt").write_text("LGPL-3.0-only: PySide6", encoding="utf-8")
        licenses = tmp_path / "packaging" / "LICENSES"
        licenses.mkdir()
        (licenses / "LGPL-3.0-only.txt").write_text("GNU LESSER GENERAL PUBLIC LICENSE", encoding="utf-8")

        (tmp_path / "LICENSE").write_text("MIT License\nCopyright ...", encoding="utf-8")

        zip_paths = build_exe.build_release_zip()
        assert [p.name for p in zip_paths] == ["EpisodeMatch-CLI-v0.11.0-win64.zip", "EpisodeMatch-GUI-v0.11.0-win64.zip"]
        by_name = {p.name: p for p in zip_paths}
        with zipfile.ZipFile(by_name["EpisodeMatch-CLI-v0.11.0-win64.zip"]) as zf:
            names = zf.namelist()
            assert "EpisodeMatch-CLI-v0.11.0-win64/cli/episodematch/episodematch.exe" in names
            assert "EpisodeMatch-CLI-v0.11.0-win64/LICENSE" in names  # audit F3
            assert "EpisodeMatch-CLI-v0.11.0-win64/README-FIRST.txt" in names
            assert "EpisodeMatch-CLI-v0.11.0-win64/THIRD_PARTY_LICENSES.txt" in names
            assert "EpisodeMatch-CLI-v0.11.0-win64/LICENSES/LGPL-3.0-only.txt" in names
            assert "EpisodeMatch-CLI-v0.11.0-win64/bin/.gitkeep" in names
            assert not [n for n in names if n.lower().endswith("ffmpeg.exe")]
            assert not [n for n in names if "config.json" in n]
        with zipfile.ZipFile(by_name["EpisodeMatch-GUI-v0.11.0-win64.zip"]) as zf:
            names = zf.namelist()
            assert "EpisodeMatch-GUI-v0.11.0-win64/gui/EpisodeMatch/EpisodeMatch.exe" in names
            assert "EpisodeMatch-GUI-v0.11.0-win64/LICENSE" in names
            assert "EpisodeMatch-GUI-v0.11.0-win64/data/.gitkeep" in names
            assert not [n for n in names if "cli/episodematch" in n]  # GUI zip ships no CLI

    def test_zip_build_fails_on_gpl_tool_in_bundle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cli, gui = self._fake_bundles(tmp_path)
        (gui / "_internal" / "ffmpeg.exe").write_bytes(b"GPL")
        monkeypatch.setattr(build_exe, "PROJECT_ROOT", tmp_path)
        (tmp_path / "packaging").mkdir()
        (tmp_path / "packaging" / "README-FIRST.txt").write_text("x", encoding="utf-8")
        (tmp_path / "packaging" / "THIRD_PARTY_LICENSES.txt").write_text("x", encoding="utf-8")
        licenses = tmp_path / "packaging" / "LICENSES"
        licenses.mkdir()
        (licenses / "LGPL-3.0-only.txt").write_text("x", encoding="utf-8")
        (tmp_path / "LICENSE").write_text("MIT", encoding="utf-8")
        with pytest.raises(SystemExit, match="Licensing guard"):
            build_exe.build_release_zip()


class TestZipGuardWiring:
    def test_zip_build_fails_on_unknown_top_level_exe(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cli = tmp_path / "dist" / "cli" / "episodematch"
        gui = tmp_path / "dist" / "gui" / "EpisodeMatch"
        (cli / "_internal").mkdir(parents=True)
        (gui / "_internal").mkdir(parents=True)
        (cli / "episodematch.exe").write_bytes(b"MZ cli")
        (gui / "EpisodeMatch.exe").write_bytes(b"MZ gui")
        (gui / "SomeUnrelatedTool.exe").write_bytes(b"MZ")
        monkeypatch.setattr(build_exe, "PROJECT_ROOT", tmp_path)
        (tmp_path / "packaging").mkdir()
        (tmp_path / "packaging" / "README-FIRST.txt").write_text("x", encoding="utf-8")
        (tmp_path / "packaging" / "THIRD_PARTY_LICENSES.txt").write_text("x", encoding="utf-8")
        (tmp_path / "LICENSE").write_text("MIT", encoding="utf-8")
        with pytest.raises(SystemExit, match="Licensing guard"):
            build_exe.build_release_zip()


class TestThirdPartyLicensesFile:
    def test_repo_licenses_file_mentions_pyside6_lgpl(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "packaging" / "THIRD_PARTY_LICENSES.txt").read_text(encoding="utf-8")
        assert "PySide6" in text and "LGPL-3.0-only" in text
        assert "NOT PART OF THIS ARCHIVE" in text  # GPL tools disclaimer
        assert (root / "packaging" / "LICENSES" / "LGPL-3.0-only.txt").is_file()
