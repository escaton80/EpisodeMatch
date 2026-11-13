"""Optional batch AV1 (SVT-AV1) encoding queue via ffmpeg.

Features:

* sequential queue with persistent resume state (``data/av1-queue.json``),
* ffmpeg ``-progress pipe:1`` parsing for live progress,
* per-job failure isolation (one broken file does not kill the queue),
* cooperative cancellation,
* audio/subtitle streams are copied (video-only re-encode).
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from episodematch.core.exceptions import EncodingError

logger = logging.getLogger(__name__)

_TIMEOUT = 6 * 3600  # long movies on slow CPUs


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EncoderJob(BaseModel):
    """One file in the encoding queue."""

    source: Path
    destination: Path
    preset: int = 6
    crf: int = 30
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


ProgressCallback = Callable[[EncoderJob, float], None]


def parse_progress_output(line: str, *, total_duration_seconds: float | None) -> float | None:
    """Parse one ``-progress`` key=value line; returns progress ratio 0..1 or None."""
    line = line.strip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    if key == "out_time_ms":
        try:
            microseconds = int(value)
        except ValueError:
            return None
        # ffmpeg's out_time_ms is actually in microseconds (known quirk).
        seconds = microseconds / 1_000_000.0
        if total_duration_seconds and total_duration_seconds > 0:
            return max(0.0, min(1.0, seconds / total_duration_seconds))
        return None
    if key == "progress":
        return 1.0 if value == "end" else None
    return None


class Av1EncoderQueue:
    """Runs :class:`EncoderJob` items sequentially with ffmpeg SVT-AV1."""

    def __init__(
        self,
        ffmpeg_path: Path,
        *,
        state_path: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._ffmpeg = Path(ffmpeg_path)
        self._state_path = state_path or Path.cwd() / "data" / "av1-queue.json"
        self._progress_cb = progress or (lambda job, ratio: None)
        self._jobs: list[EncoderJob] = []
        self._cancel_event = threading.Event()
        self._load_state()

    # -- state ----------------------------------------------------------------

    def _load_state(self) -> None:
        if not self._state_path.is_file():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._jobs = [
                EncoderJob(
                    source=Path(item["source"]),
                    destination=Path(item["destination"]),
                    preset=item.get("preset", 6),
                    crf=item.get("crf", 30),
                    status=JobStatus(item.get("status", JobStatus.PENDING.value)),
                    progress=float(item.get("progress", 0.0)),
                    error=item.get("error"),
                )
                for item in payload.get("jobs", [])
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not load encoder queue state: %s", exc)
            self._jobs = []

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "jobs": [
                {
                    "source": str(job.source),
                    "destination": str(job.destination),
                    "preset": job.preset,
                    "crf": job.crf,
                    "status": job.status.value,
                    "progress": job.progress,
                    "error": job.error,
                }
                for job in self._jobs
            ]
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- queue management ----------------------------------------------------------

    def add_job(
        self,
        source: Path,
        destination: Path | None = None,
        *,
        preset: int = 6,
        crf: int = 30,
        overwrite: bool = False,
    ) -> EncoderJob:
        """Queue one file (skips duplicates; refuses existing outputs unless ``overwrite``).

        Audit B3: the encoder would silently replace a finished ``*.av1.mkv`` at
        the final ``replace()`` step — losing hours of encoding time if the user
        queued the same folder twice. Fail fast instead.
        """
        source = Path(source)
        if not source.is_file():
            raise EncodingError(f"Source not found: {source}")
        destination = Path(destination) if destination else source.with_name(source.stem + ".av1.mkv")
        if destination.exists() and not overwrite:
            raise EncodingError(
                f"Destination already exists (use overwrite=True to re-encode): {destination}"
            )
        for job in self._jobs:
            if job.source == source and job.destination == destination:
                return job
        job = EncoderJob(source=source, destination=destination, preset=preset, crf=crf)
        self._jobs.append(job)
        self._save_state()
        return job

    @property
    def jobs(self) -> list[EncoderJob]:
        return list(self._jobs)

    def pending_jobs(self) -> list[EncoderJob]:
        return [job for job in self._jobs if job.status in (JobStatus.PENDING, JobStatus.RUNNING)]

    def clear_finished(self) -> None:
        self._jobs = [job for job in self._jobs if job.status not in (JobStatus.DONE, JobStatus.CANCELLED)]
        self._save_state()

    def cancel(self) -> None:
        self._cancel_event.set()

    # -- execution ---------------------------------------------------------------------

    def run(self) -> list[EncoderJob]:
        """Process the queue sequentially; returns the updated job list."""
        for job in self._jobs:
            if job.status != JobStatus.PENDING:
                continue
            if self._cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                self._save_state()
                continue
            self._run_job(job)
        self._save_state()
        return list(self._jobs)

    def _run_job(self, job: EncoderJob) -> None:
        job.status = JobStatus.RUNNING
        job.progress = 0.0
        job.error = None
        job.started_at = time.time()
        self._save_state()
        destination = job.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_target = destination.with_name(destination.stem + ".encoding" + destination.suffix)
        command = [
            str(self._ffmpeg),
            "-y",
            "-i", str(job.source),
            "-c:v", "libsvtav1",
            "-preset", str(job.preset),
            "-crf", str(job.crf),
            "-pix_fmt", "yuv420p",
            "-g", "240",
            "-svtav1-params", "tune=0:film-grain=0",
            "-c:a", "copy",
            "-c:s", "copy",
            "-map", "0",
            "-progress", "pipe:1",
            "-nostats",
            str(temp_target),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            job.status = JobStatus.FAILED
            job.error = f"Cannot execute ffmpeg: {exc}"
            self._save_state()
            return
        assert process.stdout is not None
        # Audit D1: stderr is drained in a background thread while the stdout
        # progress loop runs — otherwise a full stderr pipe (ffmpeg banner +
        # error spam) blocks ffmpeg forever and wait() below never returns.
        stderr_chunks: list[str] = []

        def _drain_stderr() -> None:
            try:
                if process.stderr is not None:
                    stderr_chunks.append(process.stderr.read())
            except (OSError, ValueError):
                pass

        stderr_thread = threading.Thread(
            target=_drain_stderr, daemon=True, name="av1-stderr-drain"
        )
        if process.stderr is not None:
            stderr_thread.start()
        for line in process.stdout:
            if self._cancel_event.is_set():
                process.terminate()
                job.status = JobStatus.CANCELLED
                self._save_state()
                return
            ratio = parse_progress_output(line, total_duration_seconds=None)
            if ratio is not None:
                job.progress = ratio
                self._progress_cb(job, ratio)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            job.status = JobStatus.FAILED
            job.error = "ffmpeg did not finish within 60 s and was killed"
            job.finished_at = time.time()
            temp_target.unlink(missing_ok=True)
            self._save_state()
            return
        if stderr_thread.is_alive():
            stderr_thread.join(timeout=5)
        stderr_tail = "".join(stderr_chunks)[-500:]
        if process.returncode != 0:
            job.status = JobStatus.FAILED
            job.error = f"ffmpeg exit {process.returncode}: {stderr_tail}"
            temp_target.unlink(missing_ok=True)
        elif self._cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            temp_target.unlink(missing_ok=True)
        else:
            temp_target.replace(destination)
            job.status = JobStatus.DONE
            job.progress = 1.0
        job.finished_at = time.time()
        self._save_state()
