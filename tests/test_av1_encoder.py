"""Tests for Av1EncoderQueue.add_job overwrite guard (audit B3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from episodematch.core.exceptions import EncodingError
from episodematch.postprocess.av1_encoder import Av1EncoderQueue, JobStatus


class TestAddJobOverwriteGuard:
    def _encoder(self, tmp_path: Path) -> Av1EncoderQueue:
        return Av1EncoderQueue(
            tmp_path / "ffmpeg.exe",
            state_path=tmp_path / "queue.json",
        )

    def test_add_job_refuses_existing_destination(self, tmp_path: Path) -> None:
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"x")
        existing = tmp_path / "title_t00.av1.mkv"
        existing.write_bytes(b"previous encode")
        encoder = self._encoder(tmp_path)

        with pytest.raises(EncodingError, match="already exists"):
            encoder.add_job(source, existing)
        assert encoder.jobs == []  # nothing queued

    def test_add_job_overwrite_flag_allows_existing_destination(self, tmp_path: Path) -> None:
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"x")
        existing = tmp_path / "title_t00.av1.mkv"
        existing.write_bytes(b"previous encode")
        encoder = self._encoder(tmp_path)

        job = encoder.add_job(source, existing, overwrite=True)
        assert job.destination == existing

    def test_add_job_ok_when_destination_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"x")
        encoder = self._encoder(tmp_path)

        job = encoder.add_job(source)
        assert job.destination == tmp_path / "title_t00.av1.mkv"
        assert not job.destination.exists()

class TestStderrDrain:
    """Audit D1: ffmpeg blocks on a full stderr pipe while the encoder only
    drains stdout — the job must drain stderr concurrently, not after wait()."""

    def test_flooded_stderr_does_not_deadlock(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import io
        import threading

        import episodematch.postprocess.av1_encoder as av1_module

        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"x")
        events: list[str] = []

        class FakeStderr:
            def __init__(self, eof: threading.Event) -> None:
                self._eof = eof
                self.data = "E " * 40000  # ~80 KB, enough to fill a real pipe

            def read(self, *args: object) -> str:
                events.append("stderr_read")
                self._eof.set()
                return self.data

        class FakeProcess:
            def __init__(self, tmp: Path) -> None:
                self._tmp = tmp
                self._stderr_eof = threading.Event()
                self.stdout = io.StringIO("out_time_us=1000000\n" * 50)
                self.stderr = FakeStderr(self._stderr_eof)
                self.returncode = 0

            def wait(self, timeout=None) -> int:
                # Emulates ffmpeg blocking on a full stderr buffer: wait()
                # cannot complete until the stderr pipe is consumed.
                events.append("wait_entered")
                self._stderr_eof.wait(timeout=10)
                # Emulate ffmpeg having written the encoded output (temp name
                # follows destination.stem + ".encoding" + suffix).
                (self._tmp / "title_t00.av1.encoding.mkv").write_bytes(b"encoded")
                events.append("wait_returned")
                return 0

            def kill(self) -> None: ...

            def terminate(self) -> None: ...

        fake_process = FakeProcess(tmp_path)
        monkeypatch.setattr(
            av1_module.subprocess, "Popen", lambda *args, **kwargs: fake_process
        )
        encoder = Av1EncoderQueue(tmp_path / "ffmpeg.exe", state_path=tmp_path / "queue.json")
        job = encoder.add_job(source)

        encoder.run()

        assert job.status == JobStatus.DONE
        # stderr must be drained while/concurrently-with wait(), not only after
        # wait() returned (that ordering deadlocks a real pipe).
        assert events.index("stderr_read") < events.index("wait_returned")
