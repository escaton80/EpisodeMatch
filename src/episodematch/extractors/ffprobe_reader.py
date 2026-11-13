"""Probe MKV files: ffprobe JSON parsing plus a native mini-EBML header reader.

Two independent sources are combined:

* ``ffprobe -show_format -show_streams -show_chapters`` (JSON) — the full picture.
* A small pure-Python EBML parser that reads the Matroska ``Segment > Info``
  (title, duration, segment UID) and ``Tracks`` summary directly from the file.
  It works without any external tool and cross-checks ffprobe, especially for the
  MakeMKV playlist title (e.g. ``00800.mpls``).
"""

from __future__ import annotations

import json
import logging
import re
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from episodematch.core.exceptions import ExtractionError
from episodematch.core.models import (
    ChapterInfo,
    MkvFile,
    StreamInfo,
    StreamType,
    SubtitleFormat,
)

logger = logging.getLogger(__name__)

#: Matroska codec names (ffprobe) -> subtitle format.
_SUBTITLE_CODEC_MAP = {
    "subrip": SubtitleFormat.SRT,
    "srt": SubtitleFormat.SRT,
    "ass": SubtitleFormat.ASS,
    "ssa": SubtitleFormat.ASS,
    "webvtt": SubtitleFormat.VTT,
    "hdmv_pgs_subtitle": SubtitleFormat.PGS,
    "pgssub": SubtitleFormat.PGS,
    "dvd_subtitle": SubtitleFormat.VOB,
    "dvbsub": SubtitleFormat.OTHER,
}

#: Matroska CodecIDs (EBML) -> subtitle format.
_EBML_CODEC_MAP = {
    "S_TEXT/UTF8": SubtitleFormat.SRT,
    "S_TEXT/ASS": SubtitleFormat.ASS,
    "S_TEXT/SSA": SubtitleFormat.ASS,
    "S_TEXT/WEBVTT": SubtitleFormat.VTT,
    "S_HDMV/PGS": SubtitleFormat.PGS,
    "S_VOBSUB": SubtitleFormat.VOB,
}

_MPLS_PATTERN = re.compile(r"(\d{5}\.mpls)", re.IGNORECASE)

_FFPROBE_ARGS = (
    "-v", "quiet",
    "-print_format", "json",
    "-show_format",
    "-show_streams",
    "-show_chapters",
)


# ---------------------------------------------------------------------------
# ffprobe JSON -> models (pure functions, fully testable)
# ---------------------------------------------------------------------------


def classify_subtitle_format(codec_name: str | None) -> SubtitleFormat | None:
    """Map an ffprobe/EBML codec name to a :class:`SubtitleFormat`."""
    if not codec_name:
        return None
    return _SUBTITLE_CODEC_MAP.get(codec_name.lower())


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip() in ("1", "true", "yes")
    return False


def _parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_ffprobe_json(payload: dict, *, path: Path) -> MkvFile:
    """Build an :class:`MkvFile` from an ffprobe JSON document."""
    fmt = payload.get("format") or {}
    tags = fmt.get("tags") or {}

    # Tags may appear under different casings depending on the muxer.
    def tag(*names: str) -> str | None:
        for name in names:
            for key, value in tags.items():
                if key.lower() == name.lower() and value:
                    return str(value)
        return None

    title_tag = tag("TITLE", "title")
    source_id = tag("SOURCE_ID", "source_id")
    if source_id is None and title_tag:
        match = _MPLS_PATTERN.search(title_tag)
        if match:
            source_id = match.group(1)

    streams: list[StreamInfo] = []
    for raw in payload.get("streams") or []:
        stream_tags = raw.get("tags") or {}
        disposition = raw.get("disposition") or {}
        codec_type = (raw.get("codec_type") or "other").lower()
        try:
            stream_type = StreamType(codec_type)
        except ValueError:
            stream_type = StreamType.OTHER
        language = stream_tags.get("language") or raw.get("language")
        streams.append(
            StreamInfo(
                index=int(raw.get("index", len(streams))),
                stream_type=stream_type,
                codec_name=raw.get("codec_name"),
                codec_long_name=raw.get("codec_long_name"),
                language=(language.lower() if isinstance(language, str) else None) or None,
                title=stream_tags.get("title"),
                default=_parse_bool(disposition.get("default")),
                forced=_parse_bool(disposition.get("forced")),
                commentary=_parse_bool(disposition.get("comment")),
                hearing_impaired=_parse_bool(disposition.get("hearing_impaired")),
                channels=_parse_int(raw.get("channels")),
                sample_rate=_parse_int(raw.get("sample_rate")),
                width=_parse_int(raw.get("width")),
                height=_parse_int(raw.get("height")),
                frame_rate=raw.get("avg_frame_rate") or raw.get("r_frame_rate"),
                subtitle_format=(
                    classify_subtitle_format(raw.get("codec_name"))
                    if stream_type is StreamType.SUBTITLE
                    else None
                ),
            )
        )

    chapters: list[ChapterInfo] = []
    for index, raw in enumerate(payload.get("chapters") or []):
        chapter_tags = raw.get("tags") or {}
        chapters.append(
            ChapterInfo(
                index=index,
                start_seconds=_parse_float(raw.get("start_time")) or 0.0,
                end_seconds=_parse_float(raw.get("end_time")),
                title=chapter_tags.get("title") or chapter_tags.get("TITLE"),
            )
        )

    return MkvFile(
        path=path,
        size_bytes=_parse_int(fmt.get("size")) or 0,
        duration_seconds=_parse_float(fmt.get("duration")),
        container_name=fmt.get("format_name"),
        title_tag=title_tag,
        source_id=source_id,
        overall_bit_rate=_parse_int(fmt.get("bit_rate")),
        streams=streams,
        chapters=chapters,
    )


# ---------------------------------------------------------------------------
# ffprobe subprocess
# ---------------------------------------------------------------------------


class FfprobeReader:
    """Runs ffprobe and returns :class:`MkvFile` models."""

    def __init__(self, ffprobe_path: Path | str) -> None:
        self._ffprobe = Path(ffprobe_path)

    def probe(self, path: Path | str, *, with_ebml: bool = True) -> MkvFile:
        """Probe one file; raises :class:`ExtractionError` on failure."""
        file_path = Path(path)
        if not file_path.is_file():
            raise ExtractionError(f"File not found: {file_path}")
        command = [str(self._ffprobe), *_FFPROBE_ARGS, str(file_path)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=120,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ExtractionError(f"Cannot execute ffprobe at {self._ffprobe}: {exc}") from exc
        if completed.returncode != 0:
            raise ExtractionError(
                f"ffprobe failed for {file_path} (exit {completed.returncode}): {completed.stderr.strip()[:400]}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"ffprobe returned invalid JSON for {file_path}: {exc}") from exc

        mkv = parse_ffprobe_json(payload, path=file_path)

        if with_ebml:
            try:
                ebml = read_ebml_metadata(file_path)
            except ExtractionError as exc:
                logger.debug("EBML header read failed for %s: %s", file_path, exc)
            else:
                if not mkv.duration_seconds and ebml.duration_seconds:
                    mkv.duration_seconds = ebml.duration_seconds
                if not mkv.title_tag and ebml.title:
                    mkv.title_tag = ebml.title
                if not mkv.source_id and ebml.title and _MPLS_PATTERN.search(ebml.title):
                    mkv.source_id = _MPLS_PATTERN.search(ebml.title).group(1)  # type: ignore[union-attr]
                if not mkv.segment_uid and ebml.segment_uid:
                    mkv.segment_uid = ebml.segment_uid
                if not mkv.subtitle_streams:
                    self._merge_ebml_tracks(mkv, ebml)
        return mkv

    @staticmethod
    def _merge_ebml_tracks(mkv: MkvFile, ebml: EbmlMetadata) -> None:
        """Very last resort: when ffprobe saw no subtitle tracks, fall back to EBML."""
        if not ebml.tracks:
            return
        offset = max((s.index for s in mkv.streams), default=-1) + 1
        for track in ebml.tracks:
            if track.track_type != "subtitle":
                continue
            mkv.streams.append(
                StreamInfo(
                    index=offset,
                    stream_type=StreamType.SUBTITLE,
                    codec_name=track.codec_id,
                    language=track.language,
                    title=track.name,
                    subtitle_format=_EBML_CODEC_MAP.get((track.codec_id or "").upper()),
                )
            )
            offset += 1


# ---------------------------------------------------------------------------
# Native mini-EBML parser (no external tools needed)
# ---------------------------------------------------------------------------

_EBML_HEADER_ID = 0x1A45DFA3
_SEGMENT_ID = 0x18538067
_CLUSTER_ID = 0x1F43B675

_INFO_ID = 0x1549A966
_TRACKS_ID = 0x1654AE6B
_TRACK_ENTRY_ID = 0xAE

_TITLE_ID = 0x7BA9
_DURATION_ID = 0x4489
_TIMESTAMP_SCALE_ID = 0x2AD7B1
_MUXING_APP_ID = 0x4D80
_WRITING_APP_ID = 0x7384
_DATE_UTC_ID = 0x4461
_SEGMENT_UID_ID = 0x73A4

_TRACK_NUMBER_ID = 0xD7
_TRACK_TYPE_ID = 0x83
_CODEC_ID_ID = 0x86
_LANGUAGE_ID = 0x22B59C
_NAME_ID = 0x536E
_FLAG_FORCED_ID = 0x55AA
_FLAG_DEFAULT_ID = 0x88

_TRACK_TYPE_NAMES = {1: "video", 2: "audio", 0x11: "subtitle", 0x10: "logo", 0x12: "buttons", 0x20: "control"}

_DEFAULT_TIMESTAMP_SCALE = 1_000_000  # nanoseconds per tick


class _Reader:
    """Small seek/read helper over a binary file object."""

    def __init__(self, handle) -> None:  # noqa: ANN001 - BinaryIO
        self._handle = handle

    def read(self, size: int) -> bytes:
        return self._handle.read(size)

    def seek(self, position: int) -> None:
        self._handle.seek(position)

    def tell(self) -> int:
        return self._handle.tell()


def _read_vint(reader: _Reader, *, keep_marker: bool) -> tuple[int, int, bool]:
    """Read one EBML variable-size integer. Returns (value, length, unknown_flag)."""
    first = reader.read(1)
    if not first:
        raise ExtractionError("Unexpected end of file while reading EBML vint")
    byte0 = first[0]
    if byte0 == 0:
        raise ExtractionError("Invalid EBML vint (leading zero byte)")
    length = 1
    mask = 0x80
    while not (byte0 & mask):
        mask >>= 1
        length += 1
        if length > 8:
            raise ExtractionError("Invalid EBML vint (length > 8)")
    rest = reader.read(length - 1)
    if len(rest) != length - 1:
        raise ExtractionError("Unexpected end of file inside EBML vint")
    if keep_marker:
        value = byte0
        for byte in rest:
            value = (value << 8) | byte
        # Unknown-size markers only apply to size vints; kept-marker IDs never flag unknown.
        return value, length, False
    value = byte0 & (mask - 1)
    for byte in rest:
        value = (value << 8) | byte
    all_ones = (mask - 1) == 0x7F and all(byte == 0xFF for byte in rest) and (byte0 == 0xFF)
    unknown = all_ones or (length < 8 and value == (1 << (7 * length)) - 1)
    return value, length, unknown


def _read_element_header(reader: _Reader) -> tuple[int, int | None, int]:
    """Read one element header. Returns (id, data_size_or_None, header_length)."""
    element_id, id_len, _ = _read_vint(reader, keep_marker=True)
    size, size_len, unknown = _read_vint(reader, keep_marker=False)
    return element_id, (None if unknown else size), id_len + size_len


def _decode_string(data: bytes) -> str:
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _decode_uint(data: bytes) -> int:
    if not data or len(data) > 8:
        return 0
    return int.from_bytes(data, "big")


def _decode_float(data: bytes) -> float | None:
    if len(data) == 4:
        return struct.unpack(">f", data)[0]
    if len(data) == 8:
        return struct.unpack(">d", data)[0]
    return None


@dataclass
class EbmlTrack:
    """Summary of one Matroska track entry."""

    number: int | None = None
    track_type: str | None = None
    codec_id: str | None = None
    language: str | None = None
    name: str | None = None
    forced: bool = False
    default: bool = False


@dataclass
class EbmlMetadata:
    """Matroska header metadata read natively (no ffprobe required)."""

    title: str | None = None
    duration_seconds: float | None = None
    segment_uid: str | None = None
    timestamp_scale: int = _DEFAULT_TIMESTAMP_SCALE
    muxing_app: str | None = None
    writing_app: str | None = None
    date_utc: str | None = None
    tracks: list[EbmlTrack] = field(default_factory=list)


def _parse_info_children(reader: _Reader, start: int, end: int, meta: EbmlMetadata) -> None:
    reader.seek(start)
    while reader.tell() < end:
        element_id, size, _header = _read_element_header(reader)
        if size is None:
            break
        data_start = reader.tell()
        if element_id == _TITLE_ID:
            meta.title = _decode_string(reader.read(size))
        elif element_id == _DURATION_ID:
            value = _decode_float(reader.read(size))
            if value is not None:
                meta.duration_seconds = value * meta.timestamp_scale / 1_000_000_000.0
        elif element_id == _TIMESTAMP_SCALE_ID:
            meta.timestamp_scale = _decode_uint(reader.read(size)) or _DEFAULT_TIMESTAMP_SCALE
            # Duration may have been read before the scale; recompute if so.
            # (TimestampScale is required to precede Duration per the spec, but be safe.)
        elif element_id == _MUXING_APP_ID:
            meta.muxing_app = _decode_string(reader.read(size))
        elif element_id == _WRITING_APP_ID:
            meta.writing_app = _decode_string(reader.read(size))
        elif element_id == _DATE_UTC_ID:
            meta.date_utc = reader.read(size).hex()
        elif element_id == _SEGMENT_UID_ID:
            meta.segment_uid = reader.read(size).hex()
        else:
            reader.seek(data_start + size)
    reader.seek(end)


def _parse_track_entry(reader: _Reader, start: int, end: int) -> EbmlTrack:
    track = EbmlTrack()
    reader.seek(start)
    while reader.tell() < end:
        element_id, size, _header = _read_element_header(reader)
        if size is None:
            break
        data_start = reader.tell()
        if element_id == _TRACK_NUMBER_ID:
            track.number = _decode_uint(reader.read(size))
        elif element_id == _TRACK_TYPE_ID:
            code = _decode_uint(reader.read(size))
            track.track_type = _TRACK_TYPE_NAMES.get(code, f"unknown({code})")
        elif element_id == _CODEC_ID_ID:
            track.codec_id = _decode_string(reader.read(size))
        elif element_id == _LANGUAGE_ID:
            track.language = _decode_string(reader.read(size))
        elif element_id == _NAME_ID:
            track.name = _decode_string(reader.read(size))
        elif element_id == _FLAG_FORCED_ID:
            track.forced = _decode_uint(reader.read(size)) == 1
        elif element_id == _FLAG_DEFAULT_ID:
            track.default = _decode_uint(reader.read(size)) == 1
        else:
            reader.seek(data_start + size)
    reader.seek(end)
    return track


def _parse_master_children(reader: _Reader, start: int, end: int, on_element) -> None:
    """Walk children of a master element, invoking ``on_element(id, start, end)``."""
    reader.seek(start)
    while reader.tell() < end:
        element_id, size, _header = _read_element_header(reader)
        data_start = reader.tell()
        if size is None:
            # Unknown-size child (rare outside of Segment): parse until parent end.
            child_end = end
        else:
            child_end = min(data_start + size, end)
        on_element(element_id, data_start, child_end)
        if size is None:
            break
        reader.seek(data_start + size)


def read_ebml_metadata(path: Path | str, *, include_tracks: bool = True) -> EbmlMetadata:
    """Read Matroska header metadata natively from an MKV/WebM file.

    Only the header region is read (Info/Tracks live before the first Cluster);
    the function never loads the media payload.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ExtractionError(f"File not found: {file_path}")
    meta = EbmlMetadata()
    with open(file_path, "rb") as handle:
        reader = _Reader(handle)
        file_size = file_path.stat().st_size
        top_id, top_size, _ = _read_element_header(reader)
        if top_id == _EBML_HEADER_ID:
            if top_size is None:
                raise ExtractionError("EBML header with unknown size is invalid")
            reader.seek(reader.tell() + top_size)
            top_id, top_size, _ = _read_element_header(reader)
        if top_id != _SEGMENT_ID:
            raise ExtractionError(f"Not a Matroska file (top-level id 0x{top_id:X})")
        segment_start = reader.tell()
        segment_end = file_size if top_size is None else min(segment_start + top_size, file_size)

        def on_segment_child(element_id: int, start: int, end: int) -> None:
            if element_id == _INFO_ID:
                _parse_info_children(reader, start, end, meta)
            elif element_id == _TRACKS_ID and include_tracks:
                _parse_tracks_container(reader, start, end, meta)
            elif element_id == _CLUSTER_ID:
                # Everything relevant precedes the first cluster; stop scanning.
                raise _StopScan()

        try:
            _parse_master_children(reader, segment_start, segment_end, on_segment_child)
        except _StopScan:
            pass
    return meta


class _StopScan(Exception):
    """Internal control flow: stop scanning the segment."""


def _parse_tracks_container(reader: _Reader, start: int, end: int, meta: EbmlMetadata) -> None:
    def on_tracks_child(element_id: int, track_start: int, track_end: int) -> None:
        if element_id == _TRACK_ENTRY_ID:
            meta.tracks.append(_parse_track_entry(reader, track_start, track_end))

    _parse_master_children(reader, start, end, on_tracks_child)
