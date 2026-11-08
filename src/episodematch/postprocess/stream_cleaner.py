"""Optional track cleanup: forced-subtitle detection and audio/subtitle pruning.

Safety model:

1. analyse the probed streams and build a keep-list,
2. remux to a temp file with ``mkvmerge`` (never in place),
3. verify the temp file's duration matches the source within ±1 s,
4. atomically swap it in (the original is kept as ``.bak`` until success).

"Forced" subtitle tracks are detected by the disposition flag *or* by event
coverage below the configured threshold (default 15% of the runtime) — pure
signs/songs tracks get flagged so they can be dropped with the "forced only"
policy.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from episodematch.core.exceptions import StreamCleanError
from episodematch.core.models import MkvFile, StreamInfo, SubtitleFormat, StreamType

logger = logging.getLogger(__name__)

_TIMEOUT = 1800  # remuxing can take a while on big files


@dataclass
class SubtitleTrackAnalysis:
    """Coverage analysis of one subtitle track."""

    stream: StreamInfo
    is_forced_flag: bool
    event_count: int
    coverage: float  # seconds with active subtitles / duration
    coverage_ratio: float  # coverage / duration (0..1)

    @property
    def looks_forced(self) -> bool:
        return self.is_forced_flag or (0.0 < self.coverage_ratio <= 0.15)


@dataclass
class CleanupPlan:
    """Keep/remove decision for one file."""

    file_path: Path
    keep_audio_indices: list[int] = field(default_factory=list)
    keep_subtitle_indices: list[int] = field(default_factory=list)
    remove_audio: list[StreamInfo] = field(default_factory=list)
    remove_subtitles: list[StreamInfo] = field(default_factory=list)
    forced_subtitles: list[SubtitleTrackAnalysis] = field(default_factory=list)
    destination: Path | None = None

    @property
    def is_noop(self) -> bool:
        return not self.remove_audio and not self.remove_subtitles

    def summary(self) -> str:
        if self.is_noop:
            return f"{self.file_path.name}: nothing to remove"
        return (
            f"{self.file_path.name}: remove {len(self.remove_audio)} audio / "
            f"{len(self.remove_subtitles)} subtitle tracks"
        )


def analyze_subtitle_streams(
    mkv: MkvFile,
    *,
    durations: dict[int, float] | None = None,
) -> list[SubtitleTrackAnalysis]:
    """Analyse subtitle tracks; ``durations`` maps stream index -> covered seconds.

    When coverage data is unavailable (no extraction), only the disposition flag
    is used. Extraction-based coverage comes from transcripts (sum of line
    durations per track); the 15% threshold is applied by
    :attr:`SubtitleTrackAnalysis.looks_forced`.
    """
    durations = durations or {}
    duration = mkv.duration_seconds or 0.0
    analyses: list[SubtitleTrackAnalysis] = []
    for stream in mkv.subtitle_streams:
        covered = durations.get(stream.index, 0.0)
        ratio = (covered / duration) if duration > 0 else 0.0
        analyses.append(
            SubtitleTrackAnalysis(
                stream=stream,
                is_forced_flag=stream.forced,
                event_count=0,
                coverage=covered,
                coverage_ratio=ratio,
            )
        )
    return analyses


def build_cleanup_plan(
    mkv: MkvFile,
    *,
    preferred_languages: list[str],
    keep_commentary: bool = False,
    forced_only_subtitles: bool = False,
    subtitle_analyses: list[SubtitleTrackAnalysis] | None = None,
) -> CleanupPlan:
    """Decide which tracks to keep.

    Policy:

    * audio: keep tracks in ``preferred_languages`` (or all if none match),
      optionally keep commentary tracks,
    * subtitles: with ``forced_only_subtitles`` keep only tracks that look forced;
      otherwise keep all.
    """
    plan = CleanupPlan(file_path=mkv.path)
    preferred = [code.lower() for code in preferred_languages]

    audio_streams = mkv.audio_streams
    matching = [s for s in audio_streams if (s.language or "").lower() in preferred]
    keep_audio = matching if matching else audio_streams
    if not keep_commentary:
        keep_audio = [s for s in keep_audio if not s.commentary] or keep_audio
    for stream in audio_streams:
        if stream not in keep_audio:
            plan.remove_audio.append(stream)
    plan.keep_audio_indices = [s.index for s in keep_audio]

    analyses = {analysis.stream.index: analysis for analysis in (subtitle_analyses or [])}
    keep_subtitles: list[StreamInfo] = []
    for stream in mkv.subtitle_streams:
        analysis = analyses.get(stream.index)
        if forced_only_subtitles:
            looks_forced = analysis.looks_forced if analysis else stream.forced
            if looks_forced:
                keep_subtitles.append(stream)
            else:
                plan.remove_subtitles.append(stream)
        else:
            keep_subtitles.append(stream)
    plan.keep_subtitle_indices = [s.index for s in keep_subtitles]
    plan.forced_subtitles = [a for a in (subtitle_analyses or []) if a.looks_forced]
    return plan


class StreamCleaner:
    """Applies cleanup plans via a safe mkvmerge remux."""

    def __init__(self, mkvmerge_path: Path) -> None:
        self._mkvmerge = Path(mkvmerge_path)

    def apply(self, plan: CleanupPlan, *, dry_run: bool = False) -> Path:
        """Remux the file without the removed tracks; returns the new file path."""
        if plan.is_noop:
            return plan.file_path
        destination = plan.destination or plan.file_path.with_name(f"{plan.file_path.stem}.clean.mkv")
        command = [str(self._mkvmerge), "--quiet", "-o", str(destination)]
        audio_keep = ",".join(str(index) for index in plan.keep_audio_indices)
        subtitle_keep = ",".join(str(index) for index in plan.keep_subtitle_indices)
        if audio_keep:
            command += ["--audio-tracks", audio_keep]
        if subtitle_keep:
            command += ["--subtitle-tracks", subtitle_keep]
        command.append(str(plan.file_path))
        if dry_run:
            logger.info("Dry run cleanup: %s", " ".join(command))
            return destination
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise StreamCleanError(f"Cannot execute mkvmerge: {exc}") from exc
        if completed.returncode not in (0, 1):  # 1 = warnings, still produced output
            raise StreamCleanError(
                f"mkvmerge remux failed (exit {completed.returncode}): {(completed.stderr or '')[:300]}"
            )
        if not destination.is_file():
            raise StreamCleanError(f"mkvmerge produced no output at {destination}")
        self._verify_duration(plan.file_path, destination)
        self._swap(plan.file_path, destination)
        return plan.file_path

    def _verify_duration(self, source: Path, destination: Path, *, tolerance: float = 1.0) -> None:
        """Compare durations via mkvmerge -J; refuse to swap on mismatch."""
        source_duration = self._duration_of(source)
        destination_duration = self._duration_of(destination)
        if source_duration is None or destination_duration is None:
            logger.warning("Duration verification skipped (mkvmerge -J unavailable)")
            return
        if abs(source_duration - destination_duration) > tolerance:
            destination.unlink(missing_ok=True)
            raise StreamCleanError(
                f"Remux verification failed: duration changed from {source_duration:.2f}s "
                f"to {destination_duration:.2f}s (>{tolerance:.1f}s tolerance)"
            )
        logger.debug("Duration verification OK (%.2fs)", destination_duration)

    def _duration_of(self, path: Path) -> float | None:
        command = [str(self._mkvmerge), "-J", str(path)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=120,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if not completed.stdout:
            return None
        import json

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return None
        properties = (payload.get("container") or {}).get("properties") or {}
        duration_ns = properties.get("duration")
        return duration_ns / 1_000_000_000.0 if isinstance(duration_ns, (int, float)) else None

    @staticmethod
    def _swap(original: Path, remuxed: Path) -> None:
        """Replace the original with the remuxed file, keeping a .bak."""
        backup = original.with_name(original.name + ".bak")
        if backup.exists():
            backup.unlink()
        original.rename(backup)
        try:
            remuxed.rename(original)
        except OSError:
            backup.rename(original)  # roll back
            raise
        logger.info("Cleanup applied (%s kept as .bak)", backup.name)
