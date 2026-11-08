"""Audio snipping + local Whisper transcription (pillar 3).

Two short windows are extracted per file with ffmpeg (16 kHz mono WAV):

1. right after the intro (default 02:30–04:30), and
2. shortly before the end (default 5 minutes before EOF, 2 minutes long).

Transcription runs fully local via ``faster-whisper`` (lazy import: the extra
``[audio]`` dependency is only needed when this pillar is actually used).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from episodematch.core.exceptions import ExtractionError, WhisperUnavailableError
from episodematch.core.models import SubtitleLine, Transcript

logger = logging.getLogger(__name__)

_AUDIO_TIMEOUT = 600


@dataclass(frozen=True)
class AudioWindow:
    """One extracted audio window."""

    start_seconds: float
    end_seconds: float
    wav_path: Path

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def compute_audio_windows(
    duration_seconds: float | None,
    *,
    window_1_start: float = 150.0,
    window_1_end: float = 270.0,
    minutes_before_end: float = 300.0,
    window_2_length: float = 120.0,
) -> list[tuple[float, float]]:
    """Compute the two (start, end) audio windows, clamped to the file duration.

    When the duration is unknown, only window 1 is returned at its nominal position.
    """
    windows: list[tuple[float, float]] = []
    first = (max(0.0, window_1_start), max(0.0, window_1_end))
    if first[1] > first[0]:
        windows.append(first)
    if duration_seconds and duration_seconds > 60.0:
        end = max(0.0, duration_seconds - 2.0)  # stop 2s before EOF (credits fade)
        start = max(0.0, end - minutes_before_end)
        candidate_end = min(end, start + window_2_length)
        if candidate_end - start >= 30.0 and (not windows or start > windows[-1][1]):
            windows.append((start, candidate_end))
    return windows


class AudioSnipper:
    """Extracts audio windows via ffmpeg and transcribes them with faster-whisper."""

    def __init__(
        self,
        *,
        ffmpeg_path: Path,
        work_dir: Path | None = None,
        model_size: str = "small",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        keep_temp_files: bool = False,
    ) -> None:
        self._ffmpeg = Path(ffmpeg_path)
        self._work_dir = Path(work_dir) if work_dir else Path.cwd() / "data" / "audio"
        self._model_size = model_size
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._keep_temp_files = keep_temp_files
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._whisper_model: object | None = None

    # -- ffmpeg extraction ------------------------------------------------------

    def extract_window(self, mkv_path: Path, start: float, end: float) -> AudioWindow:
        """Cut one audio window to a 16 kHz mono WAV via ffmpeg."""
        length = end - start
        if length <= 0:
            raise ExtractionError(f"Invalid audio window: {start:.2f}-{end:.2f}")
        out_path = self._work_dir / f"{mkv_path.stem}_{int(start):05d}.wav"
        command = [
            str(self._ffmpeg),
            "-y",
            "-v", "error",
            "-ss", f"{start:.3f}",
            "-i", str(mkv_path),
            "-t", f"{length:.3f}",
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(out_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_AUDIO_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ExtractionError(f"Cannot execute ffmpeg at {self._ffmpeg}: {exc}") from exc
        if completed.returncode != 0 or not out_path.is_file():
            raise ExtractionError(
                f"Audio extraction failed for {mkv_path.name} at {start:.0f}s: {(completed.stderr or '')[:300]}"
            )
        return AudioWindow(start_seconds=start, end_seconds=end, wav_path=out_path)

    def extract_windows(self, mkv_path: Path, duration_seconds: float | None) -> list[AudioWindow]:
        """Extract all configured windows of one file."""
        windows: list[AudioWindow] = []
        for start, end in compute_audio_windows(duration_seconds):
            try:
                windows.append(self.extract_window(mkv_path, start, end))
            except ExtractionError as exc:
                logger.warning("Audio window skipped for %s: %s", mkv_path.name, exc)
        return windows

    # -- whisper transcription ----------------------------------------------------

    def _load_model(self) -> object:
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415 - lazy heavy import
        except ImportError as exc:
            raise WhisperUnavailableError(
                "faster-whisper is not installed",
                hint="pip install -e .[audio]",
            ) from exc
        logger.info("Loading faster-whisper model %r (%s/%s)", self._model_size, self._device, self._compute_type)
        self._whisper_model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._whisper_model

    def transcribe_window(self, window: AudioWindow) -> tuple[list[SubtitleLine], str | None]:
        """Transcribe one window; timestamps are mapped back to absolute media time.

        Returns the subtitle lines plus the detected language (if Whisper reports one).
        """
        model = self._load_model()
        try:
            segments, info = model.transcribe(  # type: ignore[attr-defined]
                str(window.wav_path),
                language=self._language,
                beam_size=5,
                vad_filter=True,
            )
        except Exception as exc:  # noqa: BLE001 - whisper runtime failures
            raise WhisperUnavailableError(f"Whisper transcription failed: {exc}") from exc
        detected = getattr(info, "language", None)
        logger.debug(
            "Whisper window %.0fs-%.0fs: language=%s prob=%.2f",
            window.start_seconds,
            window.end_seconds,
            detected,
            getattr(info, "language_probability", 0.0),
        )
        lines: list[SubtitleLine] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            lines.append(
                SubtitleLine(
                    start_seconds=window.start_seconds + float(segment.start),
                    end_seconds=window.start_seconds + float(segment.end),
                    text=text,
                    confidence=float(segment.avg_logprob) if getattr(segment, "avg_logprob", None) is not None else None,
                )
            )
        return lines, (str(detected) if detected else None)

    def transcribe_file(
        self,
        mkv_path: Path,
        duration_seconds: float | None,
    ) -> Transcript:
        """Full pillar-3 flow: extract windows, transcribe, merge into one transcript."""
        windows = self.extract_windows(mkv_path, duration_seconds)
        if not windows:
            return Transcript(source="whisper", lines=[])
        lines: list[SubtitleLine] = []
        detected_language: str | None = None
        for window in windows:
            try:
                window_lines, window_language = self.transcribe_window(window)
            except WhisperUnavailableError:
                self._cleanup(windows)
                raise
            except ExtractionError as exc:
                logger.warning("Window transcription failed: %s", exc)
                continue
            lines.extend(window_lines)
            if detected_language is None:
                detected_language = window_language
        self._cleanup(windows)
        lines.sort(key=lambda line: line.start_seconds)
        return Transcript(source="whisper", language=detected_language, lines=lines)

    def _cleanup(self, windows: list[AudioWindow]) -> None:
        if self._keep_temp_files:
            return
        for window in windows:
            try:
                window.wav_path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - best effort
                logger.debug("Could not delete %s: %s", window.wav_path, exc)
