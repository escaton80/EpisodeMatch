"""Tests for the renamer: templates, sanitizing, collisions, dry-run, undo journal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from episodematch.core.config import format_episode_name, sanitize_filename
from episodematch.core.exceptions import RenameError
from episodematch.core.models import AssignmentEntry
from episodematch.postprocess.renamer import (  # noqa: F401 - repoint_entries used below
    RenamePlan,
    RenamePlanItem,
    Renamer,
    build_rename_plan,
)


def entry(folder: Path, file_name: str, season: int, episode: int, title: str) -> AssignmentEntry:
    return AssignmentEntry(
        file_path=folder / file_name,
        season=season,
        episode=episode,
        episode_title=title,
        confidence=0.9,
    )


@pytest.fixture()
def rips(tmp_path: Path) -> Path:
    folder = tmp_path / "rips"
    folder.mkdir()
    for name in ("title_t00.mkv", "title_t01.mkv"):
        (folder / name).write_bytes(b"\x1a\x45\xdf\xa3fake")
    return folder


class TestSanitizeFilename:
    def test_illegal_characters_replaced(self) -> None:
        assert sanitize_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"

    def test_control_characters_replaced(self) -> None:
        assert sanitize_filename("a\x00b\x1fc") == "a_b_c"

    def test_windows_reserved_names(self) -> None:
        assert sanitize_filename("CON") == "_CON"
        assert sanitize_filename("nul.txt") == "_nul.txt"
        assert sanitize_filename("com1") == "_com1"

    def test_trailing_dots_and_spaces(self) -> None:
        assert sanitize_filename("name... ") == "name"

    def test_empty_becomes_underscore(self) -> None:
        assert sanitize_filename("   ") == "_"

    def test_max_length(self) -> None:
        assert len(sanitize_filename("x" * 500, max_length=100)) == 100


class TestFormatEpisodeName:
    def test_default_template(self) -> None:
        result = format_episode_name(
            "{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}",
            series="Breaking Bad",
            season=1,
            episode=3,
            title="...And the Bag's in the River",
            extension="",
        )
        parts = result.split("/")
        assert parts[0] == "Breaking Bad"
        assert parts[1] == "Season 01"
        # Leading dots of a *component* are stripped; inner dots survive (Windows-legal).
        assert parts[2] == "Breaking Bad - S01E03 - ...And the Bag's in the River"

    def test_leading_dots_of_component_stripped(self) -> None:
        assert sanitize_filename(".hidden") == "hidden"
        assert sanitize_filename("...dots") == "dots"

    def test_inner_dots_survive(self) -> None:
        assert sanitize_filename("Mr. Robot") == "Mr. Robot"

    def test_extension_handling(self) -> None:
        result = format_episode_name(
            "{series} - S{season:02d}E{episode:02d} - {title}",
            series="X",
            season=1,
            episode=1,
            title="T",
        )
        assert result == "X - S01E01 - T.mkv"

    def test_padding_and_extension(self) -> None:
        result = format_episode_name("{series} - S{season:02d}E{episode:03d} - {title}", series="X", season=2, episode=7, title="T", extension=".mp4")
        assert result == "X - S02E007 - T.mp4"

    def test_unknown_placeholder_raises(self) -> None:
        from episodematch.core.exceptions import ConfigError

        with pytest.raises(ConfigError):
            format_episode_name("{bogus}", series="s", season=1, episode=1, title="t")

    def test_nested_folders_sanitized(self) -> None:
        result = format_episode_name("{series}/Season {season:02d}/{series} - {title}", series="Show: The Series", season=1, episode=1, title="What? Why?")
        assert ":" not in result and "?" not in result


class TestBuildRenamePlan:
    def test_plan_with_season_folder(self, rips: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot"), entry(rips, "title_t01.mkv", 1, 2, "Cat's in the Bag...")],
            series="Breaking Bad",
            template="{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=True,
        )
        assert len(plan.items) == 2
        first = plan.items[0]
        assert first.destination.parent.name == "Season 01"
        assert first.destination.name == "Breaking Bad - S01E01 - Pilot.mkv"
        assert not plan.is_empty

    def test_plan_without_season_folder(self, rips: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot")],
            series="Breaking Bad",
            template="{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}",
            season_folder=False,
            dry_run=True,
        )
        # The "Season 01" component is dropped; the series folder from the template stays.
        assert plan.items[0].destination.parent == rips / "Breaking Bad"
        assert plan.items[0].destination.name == "Breaking Bad - S01E01 - Pilot.mkv"

    def test_missing_title_gets_placeholder(self, rips: Path) -> None:
        item = AssignmentEntry(file_path=rips / "title_t00.mkv", season=1, episode=5, episode_title=None, confidence=0.5)
        plan = build_rename_plan([item], series="Show", template="{series} - S{season:02d}E{episode:02d} - {title}", dry_run=True)
        assert "Episode 5" in plan.items[0].destination.name

    def test_collision_with_existing_file(self, rips: Path) -> None:
        (rips / "Show - S01E01 - Pilot.mkv").write_bytes(b"x")
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot")],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=True,
        )
        assert plan.items[0].destination.name == "Show - S01E01 - Pilot (2).mkv"

    def test_conflicts_within_plan_detected(self, rips: Path) -> None:
        item = RenamePlanItem(source=rips / "a.mkv", destination=rips / "same.mkv")
        plan = RenamePlan(items=[item, item.model_copy()], dry_run=True)
        assert len(plan.conflicts_within_plan()) == 1


class TestRenamerApply:
    def test_dry_run_changes_nothing(self, rips: Path, tmp_path: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot")],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=True,
        )
        renamer = Renamer(tmp_path / "undo")
        applied = renamer.apply(plan)
        assert len(applied) == 1
        assert (rips / "title_t00.mkv").is_file()
        assert not (rips / "Show - S01E01 - Pilot.mkv").exists()
        assert renamer.list_journals() == []

    def test_apply_renames_and_journals(self, rips: Path, tmp_path: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot"), entry(rips, "title_t01.mkv", 1, 2, "Second")],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=False,
        )
        renamer = Renamer(tmp_path / "undo")
        applied = renamer.apply(plan)
        assert len(applied) == 2
        assert (rips / "Show - S01E01 - Pilot.mkv").is_file()
        assert (rips / "Show - S01E02 - Second.mkv").is_file()
        assert not (rips / "title_t00.mkv").exists()
        journals = renamer.list_journals()
        assert len(journals) == 1
        payload = json.loads(journals[0].read_text(encoding="utf-8"))
        assert len(payload["operations"]) == 2

    def test_validate_rejects_missing_source(self, rips: Path, tmp_path: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot")],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=False,
        )
        (rips / "title_t00.mkv").unlink()
        renamer = Renamer(tmp_path / "undo")
        with pytest.raises(RenameError):
            renamer.apply(plan)

    def test_validate_rejects_existing_destination(self, rips: Path, tmp_path: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot")],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=False,
        )
        (rips / "Show - S01E01 - Pilot.mkv").write_bytes(b"existing")
        renamer = Renamer(tmp_path / "undo")
        with pytest.raises(RenameError):
            renamer.apply(plan)


class TestUndo:
    def test_undo_latest_restores_state(self, rips: Path, tmp_path: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot"), entry(rips, "title_t01.mkv", 1, 2, "Second")],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=False,
        )
        renamer = Renamer(tmp_path / "undo")
        renamer.apply(plan)
        restored = renamer.undo_latest()
        assert len(restored) == 2
        assert (rips / "title_t00.mkv").is_file()
        assert (rips / "title_t01.mkv").is_file()
        assert not (rips / "Show - S01E01 - Pilot.mkv").exists()
        assert renamer.list_journals() == []

    def test_undo_with_season_folder_prunes_empty_dir(self, rips: Path, tmp_path: Path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot")],
            series="Show",
            template="{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}",
            dry_run=False,
        )
        renamer = Renamer(tmp_path / "undo")
        renamer.apply(plan)
        season_dir = rips / "Show" / "Season 01"
        assert (season_dir / "Show - S01E01 - Pilot.mkv").is_file()
        renamer.undo_latest()
        assert (rips / "title_t00.mkv").is_file()
        assert not season_dir.exists()  # empty season folder pruned

    def test_undo_without_journal_raises(self, tmp_path: Path) -> None:
        renamer = Renamer(tmp_path / "undo")
        with pytest.raises(RenameError):
            renamer.undo_latest()

    def test_undo_roundtrip_twice(self, rips: Path, tmp_path: Path) -> None:
        renamer = Renamer(tmp_path / "undo")
        template = "{series} - S{season:02d}E{episode:02d} - {title}"
        for episode, title in ((1, "Pilot"), (2, "Second")):
            plan = build_rename_plan(
                [entry(rips, f"title_t0{episode - 1}.mkv", 1, episode, title)],
                series="Show",
                template=template,
                dry_run=False,
            )
            renamer.apply(plan)
        renamer.undo_latest()
        renamer.undo_latest()
        assert (rips / "title_t00.mkv").is_file()
        assert (rips / "title_t01.mkv").is_file()


class TestPartialFailureSafety:
    """Audit B1: a mid-batch failure must leave a journal for the renames that
    already happened, and every planned temp target must be validated up-front."""

    def test_validate_rejects_stale_temp_targets(self, rips, tmp_path) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t01.mkv", 1, 2, "Second")],
            series="Show", template="{series} - S{season:02d}E{episode:02d} - {title}", dry_run=False,
        )
        stale = plan.items[0].destination.with_name(
            f".{plan.items[0].destination.stem}.emtmp{plan.items[0].destination.suffix}"
        )
        stale.write_bytes(b"stale")
        renamer = Renamer(tmp_path / "undo")
        with pytest.raises(RenameError, match="Temporary rename target"):
            renamer.apply(plan)
        assert (rips / "title_t01.mkv").is_file()  # nothing renamed
        assert renamer.list_journals() == []  # fail-fast before any rename

    def test_mid_batch_failure_keeps_incremental_journal(self, rips, tmp_path, monkeypatch) -> None:
        plan = build_rename_plan(
            [entry(rips, "title_t00.mkv", 1, 1, "Pilot"), entry(rips, "title_t01.mkv", 1, 2, "Second")],
            series="Show", template="{series} - S{season:02d}E{episode:02d} - {title}", dry_run=False,
        )
        renamer = Renamer(tmp_path / "undo")

        real_rename = Path.rename
        calls = {"n": 0}

        def flaky(self, target, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:  # item 2: source -> temp
                raise OSError(39, "simulated mid-batch failure")
            return real_rename(self, target, **kwargs)

        monkeypatch.setattr(Path, "rename", flaky)

        with pytest.raises(RenameError):
            renamer.apply(plan)

        journals = renamer.list_journals()
        assert len(journals) == 1  # incremental journal survives the failure
        payload = json.loads(journals[0].read_text(encoding="utf-8"))
        assert len(payload["operations"]) == 1  # item 1 is undoable
        renamer.undo_latest()
        assert (rips / "title_t00.mkv").is_file()  # restored
        assert not (rips / "Show - S01E01 - Pilot.mkv").exists()

class TestRepointEntries:
    """Audit B4: after applying renames, the solved entries must point at the
    NEW paths, so a following `--tag` (mkvpropedit) tags the renamed files."""

    def test_repoint_updates_file_paths(self, tmp_path: Path) -> None:
        from episodematch.postprocess.renamer import RenamePlanItem, repoint_entries

        entries = [
            entry(tmp_path, "title_t00.mkv", 1, 1, "Pilot"),
            entry(tmp_path, "title_t01.mkv", 1, 2, "Second"),
        ]
        applied = [
            RenamePlanItem(source=tmp_path / "title_t00.mkv", destination=tmp_path / "S01E01 - Pilot.mkv"),
            RenamePlanItem(source=tmp_path / "title_t01.mkv", destination=tmp_path / "S01E02 - Second.mkv"),
        ]
        repointed = repoint_entries(entries, applied)
        assert entries[0].file_path == tmp_path / "S01E01 - Pilot.mkv"
        assert entries[1].file_path == tmp_path / "S01E02 - Second.mkv"
        assert repointed == 2

    def test_repoint_ignores_unrelated_entries(self, tmp_path: Path) -> None:
        from episodematch.postprocess.renamer import RenamePlanItem, repoint_entries

        entries = [entry(tmp_path, "title_t00.mkv", 1, 1, "Pilot")]
        applied = [
            RenamePlanItem(source=tmp_path / "other.mkv", destination=tmp_path / "Other.mkv"),
        ]
        repointed = repoint_entries(entries, applied)
        assert entries[0].file_path == tmp_path / "title_t00.mkv"
        assert repointed == 0

    def test_repoint_skips_noop_renames(self, tmp_path: Path) -> None:
        from episodematch.postprocess.renamer import RenamePlanItem, repoint_entries

        entries = [entry(tmp_path, "same.mkv", 1, 1, "Same")]
        applied = [
            RenamePlanItem(source=tmp_path / "same.mkv", destination=tmp_path / "same.mkv"),
        ]
        repointed = repoint_entries(entries, applied)
        assert entries[0].file_path == tmp_path / "same.mkv"
        assert repointed == 0

class TestMultiEpisodeNaming:
    """Review 1.1: an entry with episode_end renders Jellyfin's multi-episode
    form with the default template — automatically, without a custom template."""

    def test_range_entry_renders_e01_e02(self, tmp_path: Path) -> None:
        entry = AssignmentEntry(
            file_path=tmp_path / "rips" / "title_t00.mkv",
            season=1,
            episode=1,
            episode_end=2,
            episode_title="London",
            confidence=0.9,
        )
        plan = build_rename_plan(
            [entry],
            series="Show",
            template="{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}",
        )
        assert plan.items[0].destination.name == "Show - S01E01-E02 - London.mkv"

    def test_single_episode_entry_unaffected(self, tmp_path: Path) -> None:
        entry = AssignmentEntry(
            file_path=tmp_path / "rips" / "title_t00.mkv",
            season=1,
            episode=1,
            episode_title="Pilot",
            confidence=0.9,
        )
        plan = build_rename_plan(
            [entry],
            series="Show",
            template="{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}",
        )
        assert plan.items[0].destination.name == "Show - S01E01 - Pilot.mkv"

class TestCrossFilesystemRename:
    """Review 7.1: renaming across filesystems (NAS share, second drive) fails
    with EXDEV — the renamer stages a verified copy in the destination
    directory instead of failing the batch. Same-filesystem renames keep the
    fast two-step path."""

    def _plan(self, tmp_path: Path) -> tuple[Path, Path]:
        rips = tmp_path / "rips"
        rips.mkdir()
        source = rips / "title_t00.mkv"
        source.write_bytes(b"x" * 10_000)
        entry = AssignmentEntry(
            file_path=source,
            season=1,
            episode=1,
            episode_title="Pilot",
            confidence=0.9,
        )
        plan = build_rename_plan(
            [entry],
            series="Show",
            template="{series} - S{season:02d}E{episode:02d} - {title}",
            season_folder=False,
        )
        destination = plan.items[0].destination
        return source, destination

    def test_exdev_fallback_stages_verified_copy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import errno
        from pathlib import Path as _Path

        source, destination = self._plan(tmp_path)
        renamer = Renamer(tmp_path / "undo")
        real_rename = _Path.rename

        def fake_rename(self: _Path, target: _Path):
            # Only the FIRST leg (source -> temp in the destination directory)
            # is simulated cross-filesystem; the second leg stays real.
            if self == source and str(target).startswith(str(destination.parent)):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_rename(self, target)

        monkeypatch.setattr(_Path, "rename", fake_rename)
        applied = renamer.apply(self.plan_of(source, destination))
        assert applied and applied[0].destination.exists()
        assert applied[0].destination.stat().st_size == 10_000  # verified copy
        assert not source.exists()  # the original is deleted last
        journals = renamer.list_journals()
        assert journals and len(json.loads(journals[0].read_text(encoding="utf-8"))["operations"]) == 1

    def test_size_mismatch_aborts_and_cleans_temp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import errno
        import shutil as _shutil
        from pathlib import Path as _Path

        source, destination = self._plan(tmp_path)
        renamer = Renamer(tmp_path / "undo")
        real_rename = _Path.rename

        def fake_rename(self: _Path, target: _Path):
            if self == source and str(target).startswith(str(destination.parent)):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_rename(self, target)

        monkeypatch.setattr(_Path, "rename", fake_rename)

        def broken_copy2(src, dst, **kwargs):
            _shutil.copyfile(src, dst)
            Path(dst).write_bytes(b"truncated")  # wrong size on purpose

        monkeypatch.setattr("episodematch.postprocess.renamer.shutil.copy2", broken_copy2)
        with pytest.raises(RenameError, match="size"):
            renamer.apply(self.plan_of(source, destination))
        assert source.exists()  # untouched: the copy failed before any delete
        temps = list(destination.parent.glob(".*emtmp*"))
        assert temps == []  # the bad staged copy was cleaned up

    def plan_of(self, source: Path, destination: Path):
        from episodematch.postprocess.renamer import RenamePlan, RenamePlanItem

        return RenamePlan(items=[RenamePlanItem(source=source, destination=destination)])
