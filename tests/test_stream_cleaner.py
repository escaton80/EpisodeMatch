"""Tests for StreamCleaner._swap backup rotation (audit B2)."""

from __future__ import annotations

from pathlib import Path

from episodematch.postprocess.stream_cleaner import StreamCleaner


class TestSwapBackupRotation:
    def test_swap_preserves_previous_backup(self, tmp_path: Path) -> None:
        original = tmp_path / "episode.mkv"
        original.write_bytes(b"current state")
        backup = tmp_path / "episode.mkv.bak"
        backup.write_bytes(b"previous backup")
        remuxed = tmp_path / "episode.mkv.clean.mkv"
        remuxed.write_bytes(b"clean state")

        StreamCleaner._swap(original, remuxed)

        assert original.read_bytes() == b"clean state"
        assert backup.read_bytes() == b"current state"
        assert (tmp_path / "episode.mkv.bak.old").read_bytes() == b"previous backup"

    def test_second_swap_rotates_without_destroying_backups(self, tmp_path: Path) -> None:
        original = tmp_path / "episode.mkv"
        original.write_bytes(b"state A")
        remuxed1 = tmp_path / "r1.mkv"
        remuxed1.write_bytes(b"state B")

        StreamCleaner._swap(original, remuxed1)
        # after swap 1: original = B, .bak = A
        assert original.read_bytes() == b"state B"
        assert (tmp_path / "episode.mkv.bak").read_bytes() == b"state A"

        remuxed2 = tmp_path / "r2.mkv"
        remuxed2.write_bytes(b"state C")
        StreamCleaner._swap(original, remuxed2)

        # after swap 2: original = C, .bak = B, .bak.old = A -- nothing destroyed
        assert original.read_bytes() == b"state C"
        assert (tmp_path / "episode.mkv.bak").read_bytes() == b"state B"
        assert (tmp_path / "episode.mkv.bak.old").read_bytes() == b"state A"
