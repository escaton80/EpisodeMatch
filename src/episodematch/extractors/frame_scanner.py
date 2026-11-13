"""Frame scanning: find title cards and credits frames via ffmpeg + OCR (pillar 4).

Pipeline per file:

1. **Scene detection** — ``ffprobe -f lavfi`` with ``select=gt(scene,T)`` over the two
   interesting windows (first N minutes, credits window). Fallback: ``ffmpeg -vf
   select=...,showinfo`` with pts_time parsed from the showinfo log lines.
2. **Keyframe extraction** — precise single-frame PNG extraction at the detected
   timestamps (``ffmpeg -ss <t> -frames:v 1``).
3. **Deduplication** — visually near-identical frames (fade-ins around one scene cut)
   are collapsed via downsampled RMS difference.
4. **OCR + classification** — each surviving frame is OCR'd; regexes extract
   "Directed by X", "Written by Y", guest-star patterns and title-card candidates.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from episodematch.core.exceptions import ExtractionError
from episodematch.core.models import TitleCardHit, TitleCardKind
from episodematch.extractors.ocr_engine import OcrEngine

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Credit text extraction (pure functions, heavily tested)
# ---------------------------------------------------------------------------

_DIRECTED_RE = re.compile(
    r"(?:directed\s+by|director|regie)[:\s]+(?P<names>[A-Z][\w'.\- ]*(?:[,;&/]\s*[A-Z][\w'.\- ]*)*)",
    re.IGNORECASE,
)
_WRITTEN_RE = re.compile(
    r"(?:(?:written|story|teleplay|screenplay)\s+by|writer|script|buch|drehbuch)[:\s]+"
    r"(?P<names>[A-Z][\w'.\- ]*(?:[,;&/]\s*[A-Z][\w'.\- ]*)*)",
    re.IGNORECASE,
)
_CREATED_RE = re.compile(
    r"created\s+by[:\s]+(?P<names>[A-Z][\w'.\- ]*(?:[,;&/]\s*[A-Z][\w'.\- ]*)*)",
    re.IGNORECASE,
)
_GUEST_RE = re.compile(
    r"^(?:guest\s+(?:starring|star|cast)|special\s+guest\s+star|with|and)\s+"
    r"(?P<names>[A-Z][\w'.\- ]*(?:\s+as\s+[A-Z][\w'.\- ]*)?(?:\s*[,;&]\s*[A-Z][\w'.\- ]*(?:\s+as\s+[A-Z][\w'.\- ]*)?)*)",
    re.IGNORECASE | re.MULTILINE,
)
_NAME_SPLIT = re.compile(r"\s*[,;&/]\s*|\s+and\s+", re.IGNORECASE)
_AS_ROLE = re.compile(r"^(?P<actor>[A-Za-z][\w'.\- ]*?)\s+as\s+(?P<role>[A-Za-z][\w'.\- ]*)$", re.IGNORECASE)
_EPISODE_TITLE_HINT = re.compile(r"(?:episode|ep\.?|folge)\s*[:#]?\s*(?P<title>.+)", re.IGNORECASE)

_NOISE_WORDS = {
    "written", "directed", "story", "teleplay", "screenplay", "by", "producer",
    "produced", "executive", "co", "starring", "guest", "created", "director",
    "and", "the", "with", "music", "edited", "photography", "production", "designer",
}


def split_names(raw: str) -> list[str]:
    """Split a credit line ("Vince Gilligan and Peter Gould") into name tokens."""
    names: list[str] = []
    for part in _NAME_SPLIT.split(raw.strip()):
        candidate = re.sub(r"\s+", " ", part).strip(" .,-")
        if not candidate or len(candidate) < 3:
            continue
        words = candidate.split()
        # Drop trailing noise words ("Directed by Peter and") and pure noise lines.
        while words and words[-1].lower().rstrip(".") in _NOISE_WORDS:
            words.pop()
        if not words or len(words) > 5:
            continue
        cleaned = " ".join(words)
        if cleaned.lower() in _NOISE_WORDS:
            continue
        names.append(cleaned)
    return names


@dataclass
class CreditExtraction:
    """Structured result of OCR text analysis for one frame."""

    directors: list[str] = field(default_factory=list)
    writers: list[str] = field(default_factory=list)
    creators: list[str] = field(default_factory=list)
    guest_stars: list[str] = field(default_factory=list)
    title_candidates: list[str] = field(default_factory=list)
    kind: TitleCardKind = TitleCardKind.UNKNOWN

    @property
    def has_credits(self) -> bool:
        return bool(self.directors or self.writers or self.creators or self.guest_stars)


def extract_credits_from_lines(lines: list[str], *, line_heights: list[int] | None = None) -> CreditExtraction:
    """Classify OCR'd text lines into directors/writers/guests/title candidates."""
    result = CreditExtraction()
    joined = "\n".join(lines)

    for match in _DIRECTED_RE.finditer(joined):
        result.directors.extend(split_names(match["names"]))
    for match in _WRITTEN_RE.finditer(joined):
        result.writers.extend(split_names(match["names"]))
    for match in _CREATED_RE.finditer(joined):
        result.creators.extend(split_names(match["names"]))
    for match in _GUEST_RE.finditer(joined):
        for name in split_names(match["names"]):
            role_match = _AS_ROLE.match(name)
            result.guest_stars.append(role_match["actor"] if role_match else name)

    # Episode title candidates: explicit "Episode: X" hints and short non-credit lines.
    for match in _EPISODE_TITLE_HINT.finditer(joined):
        title = match["title"].strip()
        if 2 <= len(title) <= 80:
            result.title_candidates.append(title)

    heights = line_heights or [0] * len(lines)
    for text, height in zip(lines, heights):
        cleaned = re.sub(r"\s+", " ", text).strip(" .:-")
        if not cleaned or len(cleaned) < 3 or len(cleaned) > 60:
            continue
        lowered = cleaned.lower()
        if any(word in lowered for word in ("directed", "written", "story", "teleplay", "produced", "starring", "created")):
            continue
        words = cleaned.split()
        looks_like_name = all(part[0].isupper() for part in words[:2] if part)
        # Title-card heuristic: short text, ideally title case, ideally big font.
        # Single-word text only qualifies when the font is clearly large (big "Quick"-
        # style title words), otherwise it would match every stray OCR token.
        max_words = 8 if (looks_like_name or height >= 40) else 2
        if 1 <= len(words) <= max_words and (looks_like_name or height >= 40):
            result.title_candidates.append(cleaned)
    # Deduplicate, keep order.
    result.title_candidates = list(dict.fromkeys(result.title_candidates))
    result.directors = list(dict.fromkeys(result.directors))
    result.writers = list(dict.fromkeys(result.writers))
    result.guest_stars = list(dict.fromkeys(result.guest_stars))

    if result.directors or result.writers or result.guest_stars:
        result.kind = TitleCardKind.CREDITS
    elif result.title_candidates:
        result.kind = TitleCardKind.TITLE_CARD
    return result


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------


def parse_lavfi_scene_lines(output: str) -> list[float]:
    """Parse ``ffprobe -of csv`` scene-detection output (``frame,<pts_time>`` rows)."""
    times: list[float] = []
    for line in output.replace("\r\n", "\n").split("\n"):
        parts = [part.strip() for part in line.strip().split(",")]
        for index, part in enumerate(parts):
            if part == "pts_time" or part.startswith("pts_time"):
                continue
            try:
                # Typical CSV rows: "frame,0.000000" or with more columns.
                # Audit C5: "-of csv=p=0" with one field prints bare pts_time
                # rows (no leading "frame" column) — accept those too.
                if (len(parts) == 1 or index > 0) and re.fullmatch(r"\d+(?:\.\d+)?", part):
                    times.append(float(part))
            except ValueError:
                continue
    # Fallback parser for rows like "frame,video,0,0,12.345" (pts_time in 3rd+ column).
    if not times:
        for line in output.replace("\r\n", "\n").split("\n"):
            match = re.search(r",(\d+\.\d{3,})(?:,|$)", line.strip())
            if match:
                times.append(float(match.group(1)))
    # Deduplicate while preserving order.
    seen: set[float] = set()
    unique: list[float] = []
    for value in times:
        key = round(value, 3)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def parse_showinfo_scene_times(log: str) -> list[float]:
    """Parse pts_time values out of ffmpeg ``showinfo`` filter log lines."""
    times: list[float] = []
    for match in re.finditer(r"pts_time:([\d.]+)", log):
        times.append(float(match.group(1)))
    return times


@dataclass
class SceneWindows:
    """The two time windows scanned for title cards / credits."""

    title_card_start: float
    title_card_end: float
    credits_start: float
    credits_end: float


def compute_scan_windows(
    duration_seconds: float | None,
    *,
    title_card_minutes: float = 5.0,
    credits_fraction: float = 0.12,
) -> SceneWindows:
    """Compute scan windows; the credits window defaults to the last 12% of runtime."""
    title_end = max(30.0, title_card_minutes * 60.0)
    if not duration_seconds or duration_seconds <= 2 * title_end:
        return SceneWindows(0.0, title_end, max(0.0, title_end), max(0.0, title_end) + 60.0)
    credits_length = max(60.0, duration_seconds * credits_fraction)
    credits_start = duration_seconds - credits_length
    return SceneWindows(0.0, title_end, credits_start, duration_seconds)


class FrameScanner:
    """Finds and classifies title cards / credits frames of one MKV file."""

    def __init__(
        self,
        *,
        ffmpeg_path: Path,
        ffprobe_path: Path,
        ocr: OcrEngine,
        work_dir: Path | None = None,
        scene_threshold: float = 0.30,
        max_frames_per_window: int = 12,
        frame_similarity_threshold: float = 12.0,
        keep_frames: bool = False,
        title_card_minutes: float = 5.0,
    ) -> None:
        self._ffmpeg = Path(ffmpeg_path)
        self._ffprobe = Path(ffprobe_path)
        self._ocr = ocr
        self._work_dir = Path(work_dir) if work_dir else Path.cwd() / "data" / "frames"
        self._scene_threshold = scene_threshold
        self._max_frames = max_frames_per_window
        self._similarity_threshold = frame_similarity_threshold
        self._keep_frames = keep_frames
        # Audit F4: the title-card window comes from ExtractionSettings.
        self._title_card_minutes = title_card_minutes
        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def title_card_minutes(self) -> float:
        return self._title_card_minutes

    # -- public API ----------------------------------------------------------

    def scan(
        self,
        mkv_path: Path,
        *,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
    ) -> list[TitleCardHit]:
        """Scan both windows and return OCR-classified hits."""
        windows = compute_scan_windows(duration_seconds, title_card_minutes=self._title_card_minutes)
        hits: list[TitleCardHit] = []
        cap = max_frames or self._max_frames

        for window_name, start, end in (
            ("title", windows.title_card_start, windows.title_card_end),
            ("credits", windows.credits_start, windows.credits_end),
        ):
            if end <= start:
                continue
            try:
                timestamps = self.detect_scene_changes(mkv_path, start, end, limit=cap)
            except ExtractionError as exc:
                logger.warning("Scene detection failed for %s (%s): %s", mkv_path.name, window_name, exc)
                continue
            if not timestamps:
                continue
            frames: list[tuple[float, Path]] = []
            for timestamp in timestamps:
                try:
                    frame_path = self.extract_frame(mkv_path, timestamp, window_name)
                except ExtractionError as exc:
                    logger.debug("Frame extraction failed at %.2fs: %s", timestamp, exc)
                    continue
                frames.append((timestamp, frame_path))
            frames = self.deduplicate_frames(frames)
            for timestamp, frame_path in frames:
                hit = self.classify_frame(frame_path, timestamp)
                if hit is not None:
                    hits.append(hit)
        hits.sort(key=lambda hit: hit.timestamp_seconds)
        return hits

    # -- steps -----------------------------------------------------------------

    def detect_scene_changes(self, mkv_path: Path, start: float, end: float, *, limit: int) -> list[float]:
        """Scene-change timestamps inside [start, end]; lavfi first, showinfo fallback."""
        length = end - start
        lavfi_input = (
            f"movie=filename='{self._escape_filter_path(mkv_path)}':seek_point={max(0.0, start - 0.5)},"
            f"select=gt(scene\\,{self._scene_threshold})"
        )
        command = [
            str(self._ffprobe),
            "-v", "error",
            "-f", "lavfi",
            "-i", lavfi_input,
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            "-read_intervals", f"%+{length:.2f}",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_FFMPEG_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode == 0:
                times = [t for t in parse_lavfi_scene_lines(completed.stdout) if start <= t <= end]
                if times:
                    return self._limit_evenly(times, limit)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("lavfi scene detection failed: %s", exc)

        # Fallback: ffmpeg select + showinfo, parse the log.
        log_path = self._work_dir / f"{mkv_path.stem}_{int(start)}.log"
        command = [
            str(self._ffmpeg),
            "-y",
            "-v", "info",
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(mkv_path),
            "-vf", f"select='gt(scene,{self._scene_threshold})',showinfo",
            "-fps_mode", "passthrough",
            "-f", "null",
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_FFMPEG_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExtractionError(f"ffmpeg scene detection failed to run: {exc}") from exc
        log_path.write_text(completed.stderr or "", encoding="utf-8", errors="replace")
        times = [start + t for t in parse_showinfo_scene_times(completed.stderr or "")]
        return self._limit_evenly(times, limit)

    def extract_frame(self, mkv_path: Path, timestamp: float, window_name: str) -> Path:
        """Extract one precise frame as PNG."""
        out_path = self._work_dir / f"{mkv_path.stem}_{window_name}_{timestamp:.3f}.png"
        command = [
            str(self._ffmpeg),
            "-y",
            "-v", "error",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", str(mkv_path),
            "-frames:v", "1",
            str(out_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_FFMPEG_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ExtractionError(f"Cannot execute ffmpeg: {exc}") from exc
        if completed.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
            raise ExtractionError(f"Frame extraction failed at {timestamp:.2f}s")
        return out_path

    def deduplicate_frames(self, frames: list[tuple[float, Path]]) -> list[tuple[float, Path]]:
        """Drop frames that are near-identical to the previous kept frame."""
        kept: list[tuple[float, Path]] = []
        previous: tuple[tuple[int, int], bytes] | None = None
        for timestamp, path in frames:
            try:
                key, pixels = _downsample(path)
            except (OSError, ValueError) as exc:
                logger.debug("Frame unreadable (%s): %s", path, exc)
                continue
            if previous is not None:
                diff = _rms_difference(previous[1], pixels)
                if diff < self._similarity_threshold:
                    if not self._keep_frames:
                        path.unlink(missing_ok=True)
                    continue
            kept.append((timestamp, path))
            previous = (key, pixels)
        return kept[: self._max_frames]

    def classify_frame(self, frame_path: Path, timestamp: float) -> TitleCardHit | None:
        """OCR one frame and classify it as title card / credits."""
        try:
            image = Image.open(frame_path)
            image.load()
        except (OSError, ValueError) as exc:
            logger.debug("Cannot open frame %s: %s", frame_path, exc)
            return None
        try:
            result = self._ocr.recognize(image)
        except Exception as exc:  # noqa: BLE001 - OCR backend failures are non-fatal
            logger.debug("OCR failed for %s: %s", frame_path, exc)
            return None
        lines = [line.text for line in result.lines if line.text.strip()]
        if not lines:
            if not self._keep_frames:
                frame_path.unlink(missing_ok=True)
            return None
        heights = [line.height for line in result.lines]
        extraction = extract_credits_from_lines(lines, line_heights=heights)
        if extraction.kind is TitleCardKind.UNKNOWN and not extraction.title_candidates:
            if not self._keep_frames:
                frame_path.unlink(missing_ok=True)
            return None
        confidences = [line.confidence for line in result.lines if line.confidence is not None]
        hit = TitleCardHit(
            timestamp_seconds=timestamp,
            kind=extraction.kind,
            raw_lines=lines,
            directors=extraction.directors,
            writers=extraction.writers + extraction.creators,
            guest_stars=extraction.guest_stars,
            title_candidates=extraction.title_candidates[:5],
            ocr_confidence=(sum(confidences) / len(confidences)) if confidences else None,
            frame_path=frame_path if self._keep_frames else None,
        )
        if not self._keep_frames:
            frame_path.unlink(missing_ok=True)
        return hit

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        text = str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        return text

    @staticmethod
    def _limit_evenly(times: list[float], limit: int) -> list[float]:
        if limit <= 0 or len(times) <= limit:
            return times
        step = len(times) / float(limit)
        return [times[min(len(times) - 1, int(i * step))] for i in range(limit)]


# ---------------------------------------------------------------------------
# Frame similarity helpers
# ---------------------------------------------------------------------------

_DOWNSAMPLE_SIZE = (32, 18)


def _downsample(path: Path) -> tuple[tuple[int, int], bytes]:
    """Downsample a frame to a tiny grayscale raster for cheap comparisons."""
    with Image.open(path) as image:
        gray = image.convert("L").resize(_DOWNSAMPLE_SIZE, Image.Resampling.BILINEAR)
        return _DOWNSAMPLE_SIZE, gray.tobytes()


def _rms_difference(a: bytes, b: bytes) -> float:
    """Root-mean-square pixel difference between two equal-size grayscale rasters."""
    if len(a) != len(b):
        return 255.0
    total = 0
    for x, y in zip(a, b):
        delta = x - y
        total += delta * delta
    return math.sqrt(total / max(1, len(a)))
