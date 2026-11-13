"""Subtitle extraction: text tracks (SRT/ASS/VTT) and PGS bitmaps -> text.

Extraction strategy per subtitle track:

* Text tracks are pulled with ``mkvextract tracks`` (preferred, keeps timing exact)
  and fall back to ``ffmpeg -map 0:<absolute index>``.
* PGS tracks are dumped as ``.sup`` (``ffmpeg -map 0:<idx> -c:s copy -f sup`` or
  ``mkvextract``), decoded with :mod:`episodematch.extractors.pgs_parser` and OCR'd
  (optionally limited to the first/last N minutes of the episode).

All subtitle text parsers are native (no external subtitle library).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

from episodematch.core.exceptions import ExtractionError
from episodematch.core.models import MkvFile, StreamInfo, SubtitleFormat, SubtitleLine, Transcript
from episodematch.extractors.ocr_engine import OcrEngine, prepare_for_ocr
from episodematch.extractors.pgs_parser import (
    PgsEvent,
    PgsSupParser,
    compute_window_ranges,
    filter_events_window,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native subtitle text parsers (pure functions, fully testable)
# ---------------------------------------------------------------------------

_SRT_TIME = re.compile(
    r"^\s*(?P<sh>\d+):(?P<sm>\d{1,2}):(?P<ss>\d{1,2})[,.](?P<sms>\d{1,3})\s*"
    r"-->\s*"
    r"(?P<eh>\d+):(?P<em>\d{1,2}):(?P<es>\d{1,2})[,.](?P<ems>\d{1,3})\s*$"
)
_ASS_TIME = re.compile(r"^\s*(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d{1,2})?)\s*$")
_ASS_DIALOGUE = re.compile(
    r"Dialogue:\s*[^,]*,\s*(?P<start>[^,]+),\s*(?P<end>[^,]+),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(?P<text>.*)$",
    re.IGNORECASE,
)
_TAG_BLOCK = re.compile(r"\{[^}]*\}")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_HI_MARKER = re.compile(r"[\[\(][^\]\)]*[\]\)]")


def _clock_to_seconds(hours: str, minutes: str, seconds: str, millis: str = "0") -> float:
    fraction = float("0." + millis.ljust(3, "0")) if millis else 0.0
    seconds_int = int(float(seconds))
    seconds_fraction = float(seconds) - seconds_int
    return int(hours) * 3600 + int(minutes) * 60 + seconds_int + seconds_fraction + fraction


def strip_markup(text: str, *, keep_hi_markers: bool = False) -> str:
    """Remove ASS override tags, HTML tags and optional hearing-impaired markers."""
    cleaned = _TAG_BLOCK.sub("", text)
    cleaned = _HTML_TAG.sub("", cleaned)
    cleaned = cleaned.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    if not keep_hi_markers:
        cleaned = _HI_MARKER.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_srt(data: str) -> list[SubtitleLine]:
    """Parse SubRip text (tolerant: skips malformed blocks silently)."""
    lines: list[SubtitleLine] = []
    block: list[str] = []

    def flush() -> None:
        nonlocal block
        if not block:
            return
        time_index = next((i for i, part in enumerate(block) if "-->" in part), None)
        if time_index is not None:
            match = _SRT_TIME.match(block[time_index])
            if match:
                start = _clock_to_seconds(match["sh"], match["sm"], match["ss"], match["sms"])
                end = _clock_to_seconds(match["eh"], match["em"], match["es"], match["ems"])
                text = strip_markup("\n".join(block[time_index + 1 :]))
                if text:
                    lines.append(SubtitleLine(start_seconds=start, end_seconds=end, text=text))
        block = []

    for raw_line in data.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            flush()
            continue
        # Skip bare cue-counter lines (an integer directly before a timestamp line).
        if stripped.isdigit() and "-->" not in "\n".join(block):
            continue
        block.append(stripped)
    flush()
    return lines


def parse_ass(data: str) -> list[SubtitleLine]:
    """Parse ASS/SSA dialogue lines (override tags stripped)."""
    lines: list[SubtitleLine] = []
    for raw_line in data.replace("\r\n", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped.lower().startswith("dialogue:"):
            continue
        match = _ASS_DIALOGUE.match(stripped)
        if not match:
            continue
        start_match = _ASS_TIME.match(match["start"].strip())
        end_match = _ASS_TIME.match(match["end"].strip())
        if not start_match or not end_match:
            continue
        try:
            start = _clock_to_seconds(start_match["h"], start_match["m"], start_match["s"])
            end = _clock_to_seconds(end_match["h"], end_match["m"], end_match["s"])
        except ValueError:
            continue
        text = strip_markup(match["text"])
        if text:
            lines.append(SubtitleLine(start_seconds=start, end_seconds=end, text=text))
    lines.sort(key=lambda line: line.start_seconds)
    return lines


def parse_vtt(data: str) -> list[SubtitleLine]:
    """Parse WebVTT (same timestamp style as SRT plus cue settings/headers)."""
    cleaned: list[str] = []
    for raw_line in data.replace("\r\n", "\n").split("\n"):
        if raw_line.strip().upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        # VTT cue settings trail the timestamp line; strip them.
        if "-->" in raw_line:
            raw_line = raw_line.split("-->")[0] + "-->" + raw_line.split("-->")[1].split()[0] if len(raw_line.split("-->")) > 1 else raw_line
        cleaned.append(raw_line)
    return parse_srt("\n".join(cleaned))


def parse_subtitle_text(fmt: SubtitleFormat | str, data: str) -> list[SubtitleLine]:
    """Dispatch to the right text parser by format."""
    fmt_value = fmt.value if isinstance(fmt, SubtitleFormat) else str(fmt).lower()
    if fmt_value in ("srt", "subrip"):
        return parse_srt(data)
    if fmt_value in ("ass", "ssa"):
        return parse_ass(data)
    if fmt_value == "vtt":
        return parse_vtt(data)
    raise ExtractionError(f"Unsupported subtitle text format: {fmt!r}")


def sample_evenly(events: list[PgsEvent], max_events: int) -> list[PgsEvent]:
    """Down-sample events evenly (linspace) so both endpoints are always kept."""
    if max_events <= 0 or len(events) <= max_events:
        return list(events)
    if max_events == 1:
        return [events[0]]
    last_index = len(events) - 1
    indices = {round(i * last_index / (max_events - 1)) for i in range(max_events)}
    return [events[index] for index in sorted(indices)]


def limit_first_n(events: list[PgsEvent], max_events: int) -> list[PgsEvent]:
    """Keep only the first N subtitle events in chronological order.

    This is the strict guard for the pure-Python PGS fallback: the beginning of
    an episode is the most discriminative region for matching, and scanning a
    whole film's PGS bitmaps in Python is far too slow. The opening events
    (title cards, opening dialog) arrive within the first minutes anyway.
    A cap of 0 (or less) strictly extracts nothing.
    """
    if max_events <= 0:
        return []
    if len(events) <= max_events:
        return list(events)
    ordered = sorted(events, key=lambda event: event.start_seconds)
    return ordered[:max_events]


# ---------------------------------------------------------------------------
# Extraction from MKV containers
# ---------------------------------------------------------------------------


def select_subtitle_streams(
    mkv: MkvFile,
    preferred_languages: Sequence[str] | None = None,
    *,
    max_tracks: int = 2,
) -> list[StreamInfo]:
    """Pick the subtitle tracks worth extracting (before any expensive work).

    Policy: only tracks whose language is in the preferred chain (MKV tags are
    ISO-639-2, config entries may use either form — both are normalized) are
    extracted, at most ``max_tracks``. Tracks without a language tag are used
    only when no preferred track exists (mis-tagged discs). Foreign tracks
    (Spanish, Portuguese, ...) are skipped entirely: converting them costs
    minutes and their dialog would never be fuzzy-matched against
    German/English references.
    """
    from episodematch.core.languages import to_osdt_language

    order: list[str] = []
    for code in preferred_languages or []:
        mapped = to_osdt_language(code)
        if mapped and mapped not in order:
            order.append(mapped)

    def base_rank(stream: StreamInfo) -> tuple[int, int]:
        lang = to_osdt_language(stream.language)
        lang_rank = order.index(lang) if lang in order else len(order)
        format_rank = 0 if stream.subtitle_format is SubtitleFormat.PGS else 1
        return (lang_rank, format_rank)

    streams = list(mkv.subtitle_streams)
    if not streams:
        return []
    if order:
        preferred_tracks = [s for s in streams if to_osdt_language(s.language) in order]
        unknown_tracks = [s for s in streams if not to_osdt_language(s.language)]
        if preferred_tracks:
            return sorted(preferred_tracks, key=base_rank)[:max_tracks]
        if unknown_tracks:
            # Mis-tagged disc: the untagged tracks are the only candidate pool.
            return sorted(unknown_tracks, key=base_rank)[:max_tracks]
    # Last resort (no preference list or nothing matched/untagged): best-ranked foreign track.
    return sorted(streams, key=base_rank)[:max_tracks]


#: seconv's --ocr-language expects ISO 639-2/T; MKV tags are usually B-form.
_SECONV_B_TO_T = {
    "ger": "deu", "fre": "fra", "chi": "zho", "cze": "ces", "dut": "nld",
    "gre": "ell", "mac": "mkd", "rum": "ron", "slo": "slk", "scr": "hrv",
    "alb": "sqi", "may": "msa", "per": "fas",
}

#: Search order for the nOCR/binary-image OCR databases (installed by the
#: SubtitleEdit GUI on first OCR use).
_SECONV_DB_CANDIDATES = (
    ("Latin.nocr", "nocr"),
    ("Latin.db", "binaryocr"),
)


def resolve_seconv_ocr_db(configured: Path | str | None) -> tuple[str, Path] | None:
    """Resolve (engine, database path) for seconv's in-process OCR engines.

    ``configured`` wins when set. Otherwise the databases installed by the
    SubtitleEdit GUI (``%APPDATA%\\Subtitle Edit\\OCR\\``) are probed. Returns
    None when nothing is found — the caller then omits the engine switch (and
    seconv falls back to Tesseract, which errors if absent; our fallback chain
    handles that).
    """
    if configured:
        db_path = Path(configured)
        engine = "nocr" if db_path.suffix.lower() == ".nocr" else "binaryocr"
        if db_path.is_file():
            return engine, db_path
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    ocr_dir = Path(appdata) / "Subtitle Edit" / "OCR"
    for file_name, engine in _SECONV_DB_CANDIDATES:
        candidate = ocr_dir / file_name
        if candidate.is_file():
            return engine, candidate
    return None


def seconv_ocr_language(track_language: str | None) -> str | None:
    """Map an MKV language tag to seconv's --ocr-language form (ISO 639-2/T)."""
    normalized = (track_language or "").strip().lower()
    if not normalized:
        return None
    return _SECONV_B_TO_T.get(normalized, normalized)


class SubtitleExtractor:
    """Extracts subtitle text (or OCR text from PGS) out of MKV files.

    For PGS tracks the strategy is:

    1. **SubtitleEdit CLI** (when available in ``./bin/``): convert the dumped
       ``.sup`` to SRT natively — fast, full-film, no Python bitmap rendering.
    2. **Python fallback**: decode PGS + OCR, but strictly limited to the first
       ``ocr_max_events`` events (default 40) and the configured time window —
       never the whole film.
    """

    SE_FORMAT_NAME = "SubRip"  # SubtitleEdit /convert format identifier

    def __init__(
        self,
        *,
        mkvextract_path: Path | None = None,
        ffmpeg_path: Path | None = None,
        ocr: OcrEngine | None = None,
        work_dir: Path | None = None,
        keep_temp_files: bool = False,
        subtitle_edit_path: Path | None = None,
        seconv_path: Path | None = None,
        subtitle_edit_timeout: float = 180.0,
        seconv_ocr_engine: str = "nocr",
        seconv_ocr_db: Path | str | None = None,
    ) -> None:
        self._mkvextract = Path(mkvextract_path) if mkvextract_path else None
        self._ffmpeg = Path(ffmpeg_path) if ffmpeg_path else None
        self._ocr = ocr
        self._work_dir = Path(work_dir) if work_dir else Path.cwd() / "data" / "subtitles"
        self._keep_temp_files = keep_temp_files
        self._subtitle_edit = Path(subtitle_edit_path) if subtitle_edit_path else None
        self._seconv = Path(seconv_path) if seconv_path else None
        self._subtitle_edit_timeout = subtitle_edit_timeout
        self._seconv_ocr_engine = seconv_ocr_engine
        self._seconv_ocr_db = seconv_ocr_db
        # Set when the SE CLI proves to require manual interaction (GUI OCR
        # dialog): every later track then skips it immediately.
        self._se_broken = False
        self._work_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ----------------------------------------------------------

    def extract(
        self,
        mkv: MkvFile,
        track: StreamInfo,
        *,
        duration_seconds: float | None = None,
        window_minutes: float | None = None,
        ocr_max_events: int = 40,
    ) -> Transcript:
        """Extract one subtitle track into a :class:`Transcript`.

        For PGS tracks ``ocr_max_events`` caps the pure-Python fallback at the
        first N events (default 40); ``window_minutes`` additionally restricts
        which events qualify. The SubtitleEdit CLI path (if configured) ignores
        the cap because it converts the whole track natively.
        """
        if not track.is_subtitle:
            raise ExtractionError(f"Track {track.index} of {mkv.path.name} is not a subtitle track")
        if track.subtitle_format is SubtitleFormat.PGS:
            return self._extract_pgs(
                mkv,
                track,
                duration_seconds=duration_seconds,
                window_minutes=window_minutes,
                max_events=ocr_max_events,
            )
        if track.subtitle_format is SubtitleFormat.VOB:
            raise ExtractionError(
                "VOB (DVD) subtitles are image-based; only text tracks and PGS are supported"
            )
        return self._extract_text(mkv, track)

    def extract_all(
        self,
        mkv: MkvFile,
        *,
        duration_seconds: float | None = None,
        window_minutes: float | None = None,
        languages_first: list[str] | None = None,
        ocr_max_events: int = 40,
        max_tracks: int = 2,
    ) -> list[Transcript]:
        """Extract the preferred subtitle tracks of one file.

        Only tracks inside the preferred language chain are extracted (see
        :func:`select_subtitle_streams`) — a typical Blu-ray with German,
        English, Spanish and Portuguese tracks touches at most the German and
        English ones, so the SubtitleEdit CLI / Python OCR never converts
        foreign tracks.
        """
        selected = select_subtitle_streams(mkv, languages_first, max_tracks=max_tracks)
        skipped = {stream.index for stream in mkv.subtitle_streams} - {stream.index for stream in selected}
        if skipped:
            skipped_languages = sorted({(stream.language or "und") for stream in mkv.subtitle_streams if stream.index in skipped})
            logger.info(
                "Skipping %d subtitle track(s) outside the preferred languages (%s): %s",
                len(skipped),
                ", ".join(languages_first or []),
                ", ".join(skipped_languages),
            )

        transcripts: list[Transcript] = []
        for track in selected:
            try:
                transcripts.append(
                    self.extract(
                        mkv,
                        track,
                        duration_seconds=duration_seconds,
                        window_minutes=window_minutes,
                        ocr_max_events=ocr_max_events,
                    )
                )
            except ExtractionError as exc:
                logger.warning("Subtitle track %d of %s: %s", track.index, mkv.path.name, exc)
        return transcripts

    # -- text tracks -----------------------------------------------------------

    def _extract_text(self, mkv: MkvFile, track: StreamInfo) -> Transcript:
        fmt = track.subtitle_format or SubtitleFormat.SRT
        suffix = {SubtitleFormat.SRT: ".srt", SubtitleFormat.ASS: ".ass", SubtitleFormat.VTT: ".vtt"}.get(fmt, ".srt")
        out_path = self._work_dir / f"{mkv.path.stem}.track{track.index}{suffix}"
        data: str | None = None
        if self._mkvextract is not None and self._mkvextract.is_file():
            data = self._run_mkvextract_text(mkv.path, track.index, out_path)
        if data is None and self._ffmpeg is not None and self._ffmpeg.is_file():
            data = self._run_ffmpeg_text(mkv, track, out_path)
        if data is None:
            raise ExtractionError(
                f"Neither mkvextract nor ffmpeg is available to extract subtitles from {mkv.path.name}",
                hint="Run 'episodematch fetch-tools'.",
            )
        lines = parse_subtitle_text(fmt, data)
        if not self._keep_temp_files:
            out_path.unlink(missing_ok=True)
        return Transcript(
            source=fmt.value,
            language=track.language,
            track_index=track.index,
            lines=lines,
        )

    def _run_mkvextract_text(self, mkv_path: Path, track_index: int, out_path: Path) -> str | None:
        command = [str(self._mkvextract), "tracks", str(mkv_path), f"{track_index}:{out_path}"]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=300,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.debug("mkvextract failed to start: %s", exc)
            return None
        if completed.returncode != 0 or not out_path.is_file():
            logger.debug("mkvextract failed (exit %s): %s", completed.returncode, (completed.stderr or "")[:300])
            return None
        return out_path.read_text(encoding="utf-8", errors="replace")

    def _run_ffmpeg_text(self, mkv: MkvFile, track: StreamInfo, out_path: Path) -> str | None:
        # ffmpeg expects the Nth *subtitle* stream, not the absolute stream index.
        relative = self._relative_subtitle_position(mkv, track)
        if relative is None:
            return None
        command = [
            str(self._ffmpeg),
            "-y",
            "-v", "error",
            "-i", str(mkv.path),
            "-map", f"0:s:{relative}",
            "-f", self._ffmpeg_text_format(track),
            str(out_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=300,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.debug("ffmpeg failed to start: %s", exc)
            return None
        if completed.returncode != 0 or not out_path.is_file():
            logger.debug("ffmpeg text extraction failed: %s", (completed.stderr or "")[:300])
            return None
        return out_path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _ffmpeg_text_format(track: StreamInfo) -> str:
        if track.subtitle_format is SubtitleFormat.ASS:
            return "ass"
        if track.subtitle_format is SubtitleFormat.VTT:
            return "webvtt"
        return "srt"

    @staticmethod
    def _relative_subtitle_position(mkv: MkvFile, track: StreamInfo) -> int | None:
        """Index of ``track`` among the subtitle streams (0-based), or None if absent."""
        if track not in mkv.streams:
            return None
        return sum(1 for s in mkv.streams if s.is_subtitle and s.index < track.index)

    # -- PGS tracks ---------------------------------------------------------------

    def _extract_pgs(
        self,
        mkv: MkvFile,
        track: StreamInfo,
        *,
        duration_seconds: float | None,
        window_minutes: float | None,
        max_events: int,
    ) -> Transcript:
        sup_path = self._dump_sup(mkv, track)
        try:
            # Fast path: SubtitleEdit/seconv CLI converts the whole track natively
            # (its own OCR) — no Python bitmap rendering at all.
            if not self._se_broken and (self._seconv is not None or self._subtitle_edit is not None):
                transcript = self._convert_pgs_via_se_cli(mkv, track, sup_path)
                if transcript is not None:
                    return transcript
                if not self._se_broken:
                    logger.info(
                        "SubtitleEdit conversion failed for %s; falling back to the Python PGS pipeline (capped at %d events)",
                        mkv.path.name,
                        max_events,
                    )
            if self._ocr is None:
                raise ExtractionError(
                    "PGS subtitles need either the SubtitleEdit CLI (not found) or an OCR backend (not configured)",
                    hint="Put SubtitleEdit.exe into ./bin/ or install an OCR backend: pip install -e .[ocr-winrt]",
                )
            try:
                parser = PgsSupParser()
                # Audit C4: restrict RLE decoding to the scan window up front —
                # a full-film .sup otherwise decodes every event bitmap before
                # the cap applies.
                pts_windows = (
                    compute_window_ranges(duration_seconds=duration_seconds, window_minutes=window_minutes)
                    if window_minutes is not None and duration_seconds
                    else None
                )
                events = parser.parse_file(sup_path, pts_windows=pts_windows)
            finally:
                if not self._keep_temp_files:
                    sup_path.unlink(missing_ok=True)
            if window_minutes is not None and duration_seconds:
                events = filter_events_window(
                    events,
                    duration_seconds=duration_seconds,
                    window_minutes=window_minutes,
                )
            events = [event for event in events if not event.is_empty]
            # Strict guard: never scan a whole film in Python — only the first N
            # events (opening dialog + title cards are the best discriminators).
            events = limit_first_n(events, max_events)
            lines: list[SubtitleLine] = []
            for event in events:
                text = self._ocr_event(event)
                if text:
                    lines.append(
                        SubtitleLine(
                            start_seconds=event.start_seconds,
                            end_seconds=event.end_seconds,
                            text=text,
                            confidence=None,
                        )
                    )
            return Transcript(
                source="pgs-ocr",
                language=track.language,
                track_index=track.index,
                lines=lines,
            )
        finally:
            # The SE fast path leaves stray .srt outputs next to the sup; clean up.
            if not self._keep_temp_files:
                sup_path.with_suffix(".srt").unlink(missing_ok=True)

    def _convert_pgs_via_se_cli(
        self,
        mkv: MkvFile,
        track: StreamInfo,
        sup_path: Path,
    ) -> Transcript | None:
        """Convert the dumped ``.sup`` to SRT via SubtitleEdit's console tools.

        ``seconv.exe`` (ships with SubtitleEdit, fully headless with OCR) is
        preferred. The plain GUI ``SubtitleEdit.exe /convert`` is tried as a
        fallback under a strict timeout: some SE versions open the manual OCR
        dialog for image-based subtitles — the timeout detects that, kills the
        window and permanently switches this extractor to the Python pipeline.
        Returns None when conversion fails — the caller falls back.
        """
        if self._seconv is not None and self._seconv.is_file():
            binary = self._seconv
            command = [str(binary), str(sup_path), "--format", "srt"]  # modern seconv syntax
            # Headless OCR: seconv's default engine (Tesseract) is usually not
            # installed — use the in-process engine whose database the SE GUI
            # already downloaded (nOCR by default), plus the track language.
            resolution = resolve_seconv_ocr_db(self._seconv_ocr_db)
            if resolution is not None:
                engine, db_path = resolution
                if self._seconv_ocr_engine:
                    engine = self._seconv_ocr_engine
                command += ["--ocr-engine", engine, "--ocr-db", str(db_path)]
            ocr_language = seconv_ocr_language(track.language)
            if ocr_language:
                command += ["--ocr-language", ocr_language]
            tool_name = "seconv"
        elif self._subtitle_edit is not None and self._subtitle_edit.is_file():
            binary = self._subtitle_edit
            command = [str(binary), "/convert", str(sup_path), self.SE_FORMAT_NAME]
            tool_name = "SubtitleEdit"
        else:
            return None

        srt_path = sup_path.with_suffix(".srt")
        srt_path.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=self._subtitle_edit_timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            self._se_broken = True
            logger.warning(
                "%s did not finish within %.0fs for %s — it probably opened the manual OCR "
                "dialog. The process was killed; falling back to the Python pipeline "
                "(first %d events). Hint: seconv.exe (ships with SubtitleEdit) converts "
                "PGS headlessly — place it in bin/SubtitleEdit/.",
                tool_name,
                self._subtitle_edit_timeout,
                mkv.path.name,
                40,
            )
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("%s failed to run: %s", tool_name, exc)
            return None
        if completed.returncode != 0:
            logger.debug(
                "%s /convert failed (exit %s): %s",
                tool_name,
                completed.returncode,
                (completed.stderr or completed.stdout or "")[:300],
            )
            return None
        # seconv honors --output-folder when given; without it the SRT lands next
        # to the input. Accept both spots to be safe across versions.
        candidates = (srt_path, Path.cwd() / (sup_path.stem + ".srt"))
        output = next((candidate for candidate in candidates if candidate.is_file()), None)
        if output is None:
            logger.debug("%s produced no SRT next to %s", tool_name, sup_path.name)
            return None
        try:
            text = output.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("Cannot read %s output: %s", tool_name, exc)
            return None
        lines = parse_srt(text)
        if not lines:
            logger.debug("%s produced an empty SRT for %s", tool_name, mkv.path.name)
            return None
        logger.info(
            "PGS track %d of %s converted via %s CLI (%d lines)",
            track.index,
            mkv.path.name,
            tool_name,
            len(lines),
        )
        return Transcript(
            source="pgs-se-cli",
            language=track.language,
            track_index=track.index,
            lines=lines,
        )

    def _dump_sup(self, mkv: MkvFile, track: StreamInfo) -> Path:
        """Dump a PGS track as a ``.sup`` file (ffmpeg preferred, mkvextract fallback)."""
        out_path = self._work_dir / f"{mkv.path.stem}.track{track.index}.sup"
        if self._ffmpeg is not None and self._ffmpeg.is_file():
            command = [
                str(self._ffmpeg),
                "-y",
                "-v", "error",
                "-i", str(mkv.path),
                "-map", f"0:{track.index}",
                "-c:s", "copy",
                "-f", "sup",
                str(out_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=600,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
                    return out_path
                logger.debug("ffmpeg sup dump failed: %s", (completed.stderr or "")[:300])
            except OSError as exc:
                logger.debug("ffmpeg failed to start: %s", exc)
        if self._mkvextract is not None and self._mkvextract.is_file():
            command = [str(self._mkvextract), "tracks", str(mkv.path), f"{track.index}:{out_path}"]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=600,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
                    return out_path
                logger.debug("mkvextract sup dump failed: %s", (completed.stderr or "")[:300])
            except OSError as exc:
                logger.debug("mkvextract failed to start: %s", exc)
        raise ExtractionError(
            f"Could not dump PGS track {track.index} from {mkv.path.name}",
            hint="Ensure ffmpeg (or mkvextract) is installed: episodematch fetch-tools",
        )

    def _ocr_event(self, event: PgsEvent) -> str:
        """OCR all bitmaps of one event; returns the joined text (may be empty)."""
        if self._ocr is None:
            return ""
        lut = event.lut_or_default()
        texts: list[str] = []
        for bitmap in event.bitmaps:
            try:
                image = bitmap.to_pil_image(lut)
            except Exception as exc:  # noqa: BLE001 - Pillow render issues must not kill the scan
                logger.debug("Bitmap render failed (object %d): %s", bitmap.object_id, exc)
                continue
            try:
                result = self._ocr.recognize(prepare_for_ocr(image))
            except Exception as exc:  # noqa: BLE001
                logger.debug("OCR failed on event at %.2fs: %s", event.start_seconds, exc)
                continue
            text = " ".join(line.text for line in result.lines if line.text.strip())
            if text.strip():
                texts.append(text.strip())
        return " ".join(texts)
