"""Native binary parser for Blu-ray PGS (Presentation Graphic Stream) subtitles.

Supported inputs
----------------

* ``.sup`` files (the common Blu-ray subtitle dump): each segment is wrapped as
  ``"PG" | PTS:u32be | DTS:u32be | type:u8 | length:u16be | payload``.
* raw PGS elementary segments exactly as stored inside Matroska ``S_HDMV/PGS``
  tracks: ``type:u8 | length:u16be | payload`` (timestamps supplied by the caller,
  matching how ``mkvmerge`` stores blocks and how ``mkvextract``/``ffmpeg`` dump them).

The header layout and RLE semantics follow the reference implementations of
FFmpeg (``libavformat/supdec.c``, ``libavcodec/pgssubdec.c``) and SubtitleEdit
(``BluRaySupParser.cs``).

Segment types::

    0x14 PDS  Palette Definition Segment
    0x15 ODS  Object Definition Segment (RLE bitmap fragments)
    0x16 PCS  Presentation Composition Segment
    0x17 WDS  Window Definition Segment
    0x80 END  End of Display Set

RLE decoding (identical to ``pgssubdec.c``)::

    byte != 0x00                  -> exactly 1 pixel of color index <byte>
    00 00                         -> end of line
    00 flags:
        run    = flags & 0x3F
        if flags & 0x40: run   = (run << 8) | next_byte      (up to 16383)
        color  = next_byte if flags & 0x80 else 0

Palette conversion uses limited-range YCbCr -> RGB (Y expands by 255/219) with
BT.709 coefficients for HD video (> 576 lines) and BT.601 for SD, exactly like
the FFmpeg decoder.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import Iterator, Sequence

from episodematch.core.exceptions import PgsParseError

logger = logging.getLogger(__name__)

SUP_MAGIC = b"PG"
PTS_DTS_HEADER_SIZE = 13  # "PG" (2) + PTS (4) + DTS (4) + type (1) + length (2)
RAW_HEADER_SIZE = 3  # type (1) + length (2)

_90K = 90_000.0


class SegmentType(IntEnum):
    PDS = 0x14
    ODS = 0x15
    PCS = 0x16
    WDS = 0x17
    END = 0x80


class CompositionState(IntEnum):
    EPOCH_START = 0x80
    ACQUISITION_POINT = 0x40
    NORMAL_CASE = 0xC0

    @classmethod
    def from_code(cls, code: int) -> CompositionState | None:
        try:
            return cls(code)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Data model (dataclasses: they carry binary buffers / PIL images)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaletteEntry:
    color_id: int
    y: int
    cr: int
    cb: int
    alpha: int

    def rgba(self, video_height: int | None = None) -> tuple[int, int, int, int]:
        r, g, b = ycbcr_to_rgb(self.y, self.cr, self.cb, video_height=video_height)
        return (r, g, b, self.alpha)


@dataclass
class Palette:
    palette_id: int
    version: int
    entries: dict[int, PaletteEntry] = field(default_factory=dict)

    def rgba_lut(self, video_height: int | None = None) -> list[tuple[int, int, int, int]]:
        """RGBA lookup table indexed by color id (0..255); missing ids are transparent."""
        lut: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * 256
        for color_id, entry in self.entries.items():
            if 0 <= color_id < 256:
                lut[color_id] = entry.rgba(video_height)
        return lut


@dataclass(frozen=True)
class CompositionObject:
    """One bitmap placement inside a PCS."""

    object_id: int
    window_id: int
    forced: bool
    x: int
    y: int
    crop: tuple[int, int, int, int] | None = None  # (x, y, w, h) when cropped


@dataclass(frozen=True)
class WindowDefinition:
    window_id: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Pcs:
    """Parsed Presentation Composition Segment payload."""

    video_width: int
    video_height: int
    frame_rate_code: int
    composition_number: int
    composition_state: CompositionState | None
    palette_update: bool
    palette_id: int
    objects: tuple[CompositionObject, ...]


@dataclass(frozen=True)
class Segment:
    """One raw PGS segment (header + payload)."""

    type: SegmentType | int
    pts_seconds: float | None
    payload: bytes

    @property
    def type_code(self) -> int:
        return int(self.type)


@dataclass
class ObjectDefinition:
    """Assembled ODS bitmap (fragments joined)."""

    object_id: int
    version: int
    width: int
    height: int
    rle_data: bytearray = field(default_factory=bytearray)
    _complete: bool = False

    @property
    def complete(self) -> bool:
        return self._complete


@dataclass(frozen=True)
class PgsBitmap:
    """A decoded subtitle bitmap positioned on the video frame."""

    object_id: int
    x: int
    y: int
    width: int
    height: int
    indices: bytes  # one palette index per pixel, row-major
    forced: bool = False

    def rgba_bytes(self, lut: Sequence[tuple[int, int, int, int]]) -> bytes:
        """Render the bitmap into RGBA bytes using the given palette LUT."""
        out = bytearray(len(self.indices) * 4)
        for i, color_id in enumerate(self.indices):
            r, g, b, a = lut[color_id]
            out[i * 4] = r
            out[i * 4 + 1] = g
            out[i * 4 + 2] = b
            out[i * 4 + 3] = a
        return bytes(out)

    def to_pil_image(self, lut: Sequence[tuple[int, int, int, int]]):  # noqa: ANN202 - PIL optional
        """Return a ``PIL.Image`` in RGBA mode (requires Pillow)."""
        from PIL import Image  # noqa: PLC0415 - Pillow is a hard dependency but keep import local

        return Image.frombytes("RGBA", (self.width, self.height), self.rgba_bytes(lut))


@dataclass
class PgsEvent:
    """One complete subtitle event (a display set with decoded bitmaps).

    ``end_seconds`` may be adjusted by :func:`refine_event_timing` after parsing.
    ``palette_lut`` holds the RGBA palette (256 entries) used to render ``bitmaps``.
    """

    start_seconds: float
    end_seconds: float
    forced: bool
    bitmaps: tuple[PgsBitmap, ...]
    video_width: int
    video_height: int
    composition_number: int
    palette_lut: tuple[tuple[int, int, int, int], ...] | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def is_empty(self) -> bool:
        return not self.bitmaps

    def lut_or_default(self) -> tuple[tuple[int, int, int, int], ...]:
        if self.palette_lut is not None and len(self.palette_lut) == 256:
            return self.palette_lut
        return ((0, 0, 0, 0),) * 256


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def ycbcr_to_rgb(
    y: int,
    cr: int,
    cb: int,
    *,
    video_height: int | None = None,
    matrix: str | None = None,
) -> tuple[int, int, int]:
    """Convert full-value YCbCr (limited-range Y) to RGB.

    ``matrix`` may be ``"bt601"`` or ``"bt709"``; by default BT.709 is used for
    HD (> 576 lines) and BT.601 for SD, matching FFmpeg's PGS decoder.
    """
    if matrix is None:
        matrix = "bt709" if (video_height is None or video_height > 576) else "bt601"
    matrix = matrix.lower()
    cb_offset = cb - 128
    cr_offset = cr - 128
    y_expanded = (y - 16) * (255.0 / 219.0)
    if matrix == "bt709":
        r = y_expanded + 1.5748 * cr_offset
        g = y_expanded - 0.18732 * cb_offset - 0.46812 * cr_offset
        b = y_expanded + 1.8556 * cb_offset
    elif matrix == "bt601":
        r = y_expanded + 1.402 * cr_offset
        g = y_expanded - 0.34414 * cb_offset - 0.71414 * cr_offset
        b = y_expanded + 1.772 * cb_offset
    else:
        raise ValueError(f"Unknown YCbCr matrix: {matrix!r} (use 'bt601' or 'bt709')")

    def clamp(value: float) -> int:
        return max(0, min(255, int(value + 0.5)))

    return clamp(r), clamp(g), clamp(b)


def decode_rle(data: bytes | bytearray, width: int, height: int) -> bytearray:
    """Decode PGS RLE data into ``width*height`` palette indices (row-major).

    Semantics follow ``pgssubdec.c``: a non-zero byte yields exactly one pixel of
    that color index; the ``0x00`` escape starts a run with a 6/14-bit counter and
    an optional explicit color; ``00 00`` ends the current line.
    """
    if width <= 0 or height <= 0:
        raise PgsParseError(f"Invalid bitmap dimensions {width}x{height}")
    out = bytearray(width * height)
    pos = 0
    pixel = 0
    total = width * height
    line = 0
    data_len = len(data)
    while pos < data_len and line < height:
        color = data[pos]
        pos += 1
        run = 1
        if color == 0x00:
            if pos >= data_len:
                break  # truncated escape: treat as end of stream
            flags = data[pos]
            pos += 1
            run = flags & 0x3F
            if flags & 0x40:
                if pos >= data_len:
                    raise PgsParseError(
                        "Truncated RLE data: 14-bit run length incomplete",
                        offset=pos,
                    )
                run = (run << 8) + data[pos]
                pos += 1
            color = 0
            if flags & 0x80:
                if pos >= data_len:
                    raise PgsParseError(
                        "Truncated RLE data: explicit color missing",
                        offset=pos,
                    )
                color = data[pos]
                pos += 1
        if run > 0:
            if pixel + run > total:
                # Overrun: clip instead of failing (some encoders pad the last line).
                run = total - pixel
                if run <= 0:
                    break
            for i in range(run):
                out[pixel + i] = color
            pixel += run
        else:
            # End of line.
            line += 1
    if pixel < total:
        raise PgsParseError(
            f"Insufficient RLE data: decoded {pixel} of {total} pixels",
            offset=pos,
        )
    return out


def encode_rle(indices: bytes | bytearray, width: int, height: int) -> bytes:
    """Encode palette indices into PGS RLE (inverse of :func:`decode_rle`).

    Strategy: emit single-pixel non-zero bytes verbatim; collapse zero runs with
    the escape grammar; terminate every line with ``00 00``.
    """
    if len(indices) != width * height:
        raise ValueError(f"indices length {len(indices)} != {width}*{height}")
    out = bytearray()

    def emit_zero_run(count: int) -> None:
        while count > 0:
            if count <= 0x3F:
                out.append(0x00)
                out.append(count)
                count = 0
            else:
                take = min(count, 0x3FFF)
                out.append(0x00)
                out.append(0x40 | (take >> 8))
                out.append(take & 0xFF)
                count -= take

    def flush_run(run_value: int, run_length: int) -> None:
        if run_length == 0:
            return
        if run_value == 0:
            emit_zero_run(run_length)
        else:
            out.extend(bytes([run_value]) * run_length)

    area = width * height
    for line_start in range(0, area, width):
        line = indices[line_start : line_start + width]
        run_value = line[0] if line else 0
        run_length = 0
        for value in line:
            if value == run_value:
                run_length += 1
            else:
                flush_run(run_value, run_length)
                run_value, run_length = value, 1
        flush_run(run_value, run_length)
        out.append(0x00)
        out.append(0x00)
    return bytes(out)


# ---------------------------------------------------------------------------
# Segment reader
# ---------------------------------------------------------------------------


def iter_sup_segments(data: bytes | bytearray) -> Iterator[Segment]:
    """Iterate segments of a ``.sup`` stream (``PG`` + PTS/DTS wrapper)."""
    view = memoryview(data)
    offset = 0
    total = len(view)
    while offset < total:
        if total - offset < PTS_DTS_HEADER_SIZE:
            raise PgsParseError(
                f"Truncated segment header at offset {offset} "
                f"({total - offset} bytes left, need {PTS_DTS_HEADER_SIZE})",
                offset=offset,
            )
        if bytes(view[offset : offset + 2]) != SUP_MAGIC:
            raise PgsParseError(
                f"Bad segment magic at offset {offset}: "
                f"{bytes(view[offset : offset + 2])!r} (expected b'PG')",
                offset=offset,
            )
        pts_raw = struct.unpack_from(">I", view, offset + 2)[0]
        segment_type = view[offset + 10]
        length = struct.unpack_from(">H", view, offset + 11)[0]
        payload_start = offset + PTS_DTS_HEADER_SIZE
        payload_end = payload_start + length
        if payload_end > total:
            raise PgsParseError(
                f"Truncated segment payload at offset {offset}: "
                f"declared {length} bytes, only {total - payload_start} available",
                offset=offset,
            )
        yield Segment(
            type=_coerce_segment_type(segment_type),
            pts_seconds=pts_raw / _90K,
            payload=bytes(view[payload_start:payload_end]),
        )
        offset = payload_end


def iter_raw_segments(data: bytes | bytearray) -> Iterator[Segment]:
    """Iterate raw Matroska PGS segments (``type`` + ``length``, no wrapper)."""
    view = memoryview(data)
    offset = 0
    total = len(view)
    while offset < total:
        if total - offset < RAW_HEADER_SIZE:
            raise PgsParseError(
                f"Truncated raw segment header at offset {offset}",
                offset=offset,
            )
        segment_type = view[offset]
        length = struct.unpack_from(">H", view, offset + 1)[0]
        payload_start = offset + RAW_HEADER_SIZE
        payload_end = payload_start + length
        if payload_end > total:
            raise PgsParseError(
                f"Truncated raw segment payload at offset {offset}: "
                f"declared {length} bytes, only {total - payload_start} available",
                offset=offset,
            )
        yield Segment(
            type=_coerce_segment_type(segment_type),
            pts_seconds=None,
            payload=bytes(view[payload_start:payload_end]),
        )
        offset = payload_end


def _coerce_segment_type(code: int) -> SegmentType | int:
    try:
        return SegmentType(code)
    except ValueError:
        return code


# ---------------------------------------------------------------------------
# Payload parsers
# ---------------------------------------------------------------------------


def parse_pcs(payload: bytes) -> Pcs:
    """Parse a PCS payload into :class:`Pcs`."""
    if len(payload) < 11:
        raise PgsParseError(f"PCS payload too short: {len(payload)} bytes (need >= 11)")
    video_width = struct.unpack_from(">H", payload, 0)[0]
    video_height = struct.unpack_from(">H", payload, 2)[0]
    frame_rate_code = payload[4]
    composition_number = struct.unpack_from(">H", payload, 5)[0]
    composition_state = CompositionState.from_code(payload[7])
    palette_update = (payload[8] & 0x80) != 0
    palette_id = payload[9]
    object_count = payload[10]

    objects: list[CompositionObject] = []
    offset = 11
    for _ in range(object_count):
        if len(payload) - offset < 8:
            raise PgsParseError(
                f"Truncated composition object at offset {offset}",
                offset=offset,
            )
        object_id = struct.unpack_from(">H", payload, offset)[0]
        window_id = payload[offset + 2]
        flags = payload[offset + 3]
        cropped = bool(flags & 0x80)
        forced = bool(flags & 0x40)
        offset += 4
        crop: tuple[int, int, int, int] | None = None
        if cropped:
            if len(payload) - offset < 8:
                raise PgsParseError(
                    f"Truncated crop rectangle at offset {offset}",
                    offset=offset,
                )
            crop_x, crop_y, crop_w, crop_h = struct.unpack_from(">HHHH", payload, offset)
            crop = (crop_x, crop_y, crop_w, crop_h)
            offset += 8
        if len(payload) - offset < 4:
            raise PgsParseError(
                f"Truncated object position at offset {offset}",
                offset=offset,
            )
        x, y = struct.unpack_from(">HH", payload, offset)
        offset += 4
        objects.append(
            CompositionObject(
                object_id=object_id,
                window_id=window_id,
                forced=forced,
                x=x,
                y=y,
                crop=crop,
            )
        )
    return Pcs(
        video_width=video_width,
        video_height=video_height,
        frame_rate_code=frame_rate_code,
        composition_number=composition_number,
        composition_state=composition_state,
        palette_update=palette_update,
        palette_id=palette_id,
        objects=tuple(objects),
    )


def parse_wds(payload: bytes) -> list[WindowDefinition]:
    """Parse a WDS payload into a list of window definitions."""
    if not payload:
        return []
    count = payload[0]
    windows: list[WindowDefinition] = []
    offset = 1
    for index in range(count):
        if len(payload) - offset < 9:
            raise PgsParseError(
                f"Truncated window definition {index} at offset {offset}",
                offset=offset,
            )
        window_id = payload[offset]
        x, y, width, height = struct.unpack_from(">HHHH", payload, offset + 1)
        windows.append(WindowDefinition(window_id=window_id, x=x, y=y, width=width, height=height))
        offset += 9
    return windows


def parse_pds(payload: bytes) -> Palette:
    """Parse a PDS payload into a :class:`Palette`."""
    if len(payload) < 2:
        raise PgsParseError(f"PDS payload too short: {len(payload)} bytes (need >= 2)")
    palette_id = payload[0]
    version = payload[1]
    palette = Palette(palette_id=palette_id, version=version)
    offset = 2
    while offset + 5 <= len(payload):
        color_id, y, cr, cb, alpha = payload[offset : offset + 5]
        palette.entries[color_id] = PaletteEntry(color_id=color_id, y=y, cr=cr, cb=cb, alpha=alpha)
        offset += 5
    return palette


class _OdsAssembler:
    """Reassembles fragmented ODS payloads into complete :class:`ObjectDefinition`s.

    Length bookkeeping: the 3-byte ``object_data_length`` in the first fragment
    counts the object data *including* the 4-byte width/height pair that follows
    it, so the RLE payload length is ``declared - 4`` (verified against real
    Blu-ray discs and consistent with SubtitleEdit's parser).
    """

    def __init__(self) -> None:
        self._partial: dict[int, tuple[ObjectDefinition, int]] = {}

    def feed(self, payload: bytes) -> list[ObjectDefinition]:
        """Consume one ODS payload; returns objects completed by it."""
        if len(payload) < 4:
            raise PgsParseError(f"ODS payload too short: {len(payload)} bytes")
        object_id = struct.unpack_from(">H", payload, 0)[0]
        version = payload[2]
        sequence_descriptor = payload[3]
        is_first = bool(sequence_descriptor & 0x80)
        is_last = bool(sequence_descriptor & 0x40)

        if is_first:
            if len(payload) < 11:
                raise PgsParseError(
                    f"First ODS fragment too short: {len(payload)} bytes (need >= 11)",
                )
            data_length = int.from_bytes(payload[4:7], "big")
            width = struct.unpack_from(">H", payload, 7)[0]
            height = struct.unpack_from(">H", payload, 9)[0]
            fragment = payload[11:]
            obj = ObjectDefinition(
                object_id=object_id,
                version=version,
                width=width,
                height=height,
                rle_data=bytearray(fragment),
            )
            self._partial[object_id] = (obj, data_length)
        else:
            entry = self._partial.get(object_id)
            if entry is None:
                raise PgsParseError(
                    f"ODS continuation for unknown object id {object_id}",
                    hint="The first fragment was missing or already consumed.",
                )
            entry[0].rle_data.extend(payload[4:])

        completed: list[ObjectDefinition] = []
        if is_last:
            entry = self._partial.pop(object_id, None)
            if entry is not None:
                obj, declared_length = entry
                # The declared object_data_length INCLUDES the 4 bytes of the
                # width/height fields (exactly how real Blu-ray encoders count it),
                # so the actual RLE payload is declared_length - 4. A mismatch
                # still completes the object but is surfaced as a warning.
                expected_rle = declared_length - 4 if declared_length >= 4 else declared_length
                if declared_length and len(obj.rle_data) != expected_rle:
                    logger.warning(
                        "ODS %d: RLE data length %d differs from declared %d (-4 width/height header)",
                        object_id,
                        len(obj.rle_data),
                        declared_length,
                    )
                obj._complete = True
                completed.append(obj)
        return completed

    def finish(self) -> list[ObjectDefinition]:
        """Return (and clear) objects that never received their final fragment."""
        leftovers = [entry[0] for entry in self._partial.values()]
        self._partial.clear()
        return leftovers

    def reset(self) -> None:
        self._partial.clear()


# ---------------------------------------------------------------------------
# Display-set builder -> events
# ---------------------------------------------------------------------------


class PgsSupParser:
    """Parses ``.sup`` streams (or raw Matroska PGS segments) into subtitle events."""

    def __init__(self, *, default_event_duration: float = 2.0) -> None:
        self._default_event_duration = default_event_duration
        self._reset()

    def _reset(self, pts_windows: Sequence[tuple[float, float]] | None = None) -> None:
        self._palettes: dict[int, Palette] = {}
        self._ods = _OdsAssembler()
        self._assembled_store: dict[int, ObjectDefinition] = {}
        self._events: list[PgsEvent] = []
        self._boundaries: list[float] = []
        # Audit C4: when a time window is given, display sets outside it are
        # folded (objects/palettes still cached) but NOT RLE-decoded.
        self._pts_windows = list(pts_windows) if pts_windows else None

    # -- public API -------------------------------------------------------------

    def parse_file(
        self,
        path: Path | str,
        *,
        pts_windows: Sequence[tuple[float, float]] | None = None,
    ) -> list[PgsEvent]:
        data = Path(path).read_bytes()
        return self.parse_sup(data, pts_windows=pts_windows)

    def parse_sup(
        self,
        data: bytes | bytearray,
        *,
        pts_windows: Sequence[tuple[float, float]] | None = None,
    ) -> list[PgsEvent]:
        """Parse a full ``.sup`` byte stream.

        ``pts_windows`` (list of ``(start, end)`` second ranges) restricts which
        display sets are RLE-decoded into events — sets outside every window
        only contribute timing boundaries. Objects and palettes from sets
        outside the window are still cached, so in-window events that reuse
        them keep working.
        """
        self._reset(pts_windows)
        pending: list[Segment] = []
        for segment in iter_sup_segments(data):
            if segment.type_code == SegmentType.END:
                self._consume_display_set(pending)
                pending = []
            else:
                if segment.type_code == SegmentType.PCS and pending:
                    # No END between two PCS segments: treat as a new display set
                    # (tolerant towards streams that omit the END segment).
                    self._consume_display_set(pending)
                    pending = []
                pending.append(segment)
        # Trailing display set without END: still emit it (tolerant, like SubtitleEdit).
        if pending:
            self._consume_display_set(pending)
        events = self._events
        refine_event_timing(events, boundaries=self._boundaries, default_event_duration=self._default_event_duration)
        self._reset()
        return events

    def parse_raw(
        self,
        data: bytes | bytearray,
        *,
        pts_seconds: float | None = None,
        pts_windows: Sequence[tuple[float, float]] | None = None,
    ) -> list[PgsEvent]:
        """Parse raw Matroska-style segments; ``pts_seconds`` seeds all segment PTS values."""
        self._reset(pts_windows)
        pending: list[Segment] = []
        for raw in iter_raw_segments(data):
            pending.append(Segment(type=raw.type, pts_seconds=pts_seconds, payload=raw.payload))
        if pending:
            self._consume_display_set(pending)
        events = self._events
        refine_event_timing(events, boundaries=self._boundaries, default_event_duration=self._default_event_duration)
        self._reset()
        return events

    # -- internals -----------------------------------------------------------------

    def _consume_display_set(self, segments: list[Segment]) -> None:
        """Fold one display set (segments up to END) into a PgsEvent."""
        if not segments:
            return
        self._ods.reset()
        # Audit C1: assembled objects persist for the whole epoch (a normal-case
        # PCS may reference an object defined in an earlier display set), so the
        # store is NOT cleared per display set — only on EPOCH_START/reset.
        pcs: Pcs | None = None
        pcs_pts: float | None = None
        for segment in segments:
            code = segment.type_code
            try:
                if code == SegmentType.PCS:
                    pcs = parse_pcs(segment.payload)
                    pcs_pts = segment.pts_seconds
                    if pcs.composition_state == CompositionState.EPOCH_START:
                        # New epoch: palettes and objects from the previous epoch
                        # are invalidated (objects are re-sent by this set).
                        self._palettes.clear()
                        self._assembled_store.clear()
                elif code == SegmentType.WDS:
                    parse_wds(segment.payload)  # validated for structure; placements come from PCS
                elif code == SegmentType.PDS:
                    palette = parse_pds(segment.payload)
                    existing = self._palettes.get(palette.palette_id)
                    if existing is not None:
                        # Audit C2: real streams send partial palette updates —
                        # merge per color id instead of wiping the old palette.
                        existing.version = palette.version
                        existing.entries.update(palette.entries)
                    else:
                        self._palettes[palette.palette_id] = palette
                elif code == SegmentType.ODS:
                    for obj in self._ods.feed(segment.payload):
                        self._assembled_store[obj.object_id] = obj
                elif code == SegmentType.END:
                    break
                else:
                    logger.debug("Ignoring unknown PGS segment type 0x%02X", code)
            except PgsParseError:
                raise
            except (struct.error, IndexError) as exc:
                raise PgsParseError(f"Malformed PGS segment type 0x{code:02X}: {exc}") from exc

        if pcs is None:
            logger.debug("Display set without PCS skipped")
            return
        if pcs_pts is None:
            logger.debug("Display set without PTS skipped")
            return

        self._boundaries.append(pcs_pts)
        if not self._assembled_store or not pcs.objects:
            # Clearing/empty display set: contributes a timing boundary only.
            return

        if self._pts_windows is not None and not any(
            lo <= pcs_pts <= hi for lo, hi in self._pts_windows
        ):
            # Audit C4: out-of-window display set — cache state only, skip the
            # expensive RLE decode (the event is not scanned anyway).
            return

        start = pcs_pts
        end = start + self._default_event_duration

        bitmaps: list[PgsBitmap] = []
        for placement in pcs.objects:
            obj = self._assembled_store.get(placement.object_id)
            if obj is None:
                logger.debug("Composition references object %d that was not decoded", placement.object_id)
                continue
            indices = decode_rle(obj.rle_data, obj.width, obj.height)
            bitmaps.append(
                PgsBitmap(
                    object_id=placement.object_id,
                    x=placement.x,
                    y=placement.y,
                    width=obj.width,
                    height=obj.height,
                    indices=bytes(indices),
                    forced=placement.forced,
                )
            )
        if not bitmaps:
            return

        active_palette = self._palettes.get(pcs.palette_id)
        event = PgsEvent(
            start_seconds=start,
            end_seconds=end,
            forced=any(b.forced for b in bitmaps),
            bitmaps=tuple(bitmaps),
            video_width=pcs.video_width,
            video_height=pcs.video_height,
            composition_number=pcs.composition_number,
            palette_lut=(
                tuple(active_palette.rgba_lut(pcs.video_height)) if active_palette else None
            ),
        )
        self._events.append(event)


def refine_event_timing(
    events: list[PgsEvent],
    *,
    boundaries: Sequence[float],
    default_event_duration: float = 2.0,
) -> None:
    """Clamp each event's end time to the next PCS boundary (clearing or next event).

    Audit C3: an event with no later boundary (stream ends with a visible
    display set) keeps the ``default_event_duration`` instead of collapsing to
    zero length.
    """
    ordered = sorted(set(boundaries))
    for index, event in enumerate(events):
        candidates = [b for b in ordered if b > event.start_seconds + 1e-6]
        if candidates:
            event_end = candidates[0]
        else:
            following = [e.start_seconds for e in events[index + 1 :] if e.start_seconds > event.start_seconds + 1e-6]
            event_end = min(following) if following else event.start_seconds + default_event_duration
        event.end_seconds = event_end if event_end > event.start_seconds else event.start_seconds + default_event_duration


def parse_sup_events(
    data: bytes | bytearray,
    *,
    default_event_duration: float = 2.0,
    pts_windows: Sequence[tuple[float, float]] | None = None,
) -> list[PgsEvent]:
    """Convenience wrapper: parse a ``.sup`` stream in one call."""
    return PgsSupParser(default_event_duration=default_event_duration).parse_sup(data, pts_windows=pts_windows)


def compute_window_ranges(duration_seconds: float | None, window_minutes: float) -> list[tuple[float, float]]:
    """Absolute second ranges of the first/last ``window_minutes`` scan windows.

    Mirrors :func:`filter_events_window`; a short track merges into one range.
    Returns ``[]`` when ``duration_seconds`` is unknown — i.e. "no restriction".
    """
    if duration_seconds is None or duration_seconds <= 0:
        return []
    window = window_minutes * 60.0
    if duration_seconds <= 2 * window:
        return [(0.0, duration_seconds)]
    return [(0.0, window), (duration_seconds - window, duration_seconds)]


# ---------------------------------------------------------------------------
# Time-window filtering (pillar 2 uses first/last N minutes)
# ---------------------------------------------------------------------------


def filter_events_window(
    events: Sequence[PgsEvent],
    *,
    duration_seconds: float | None,
    window_minutes: float = 10.0,
) -> list[PgsEvent]:
    """Keep only events within the first/last ``window_minutes`` of the episode.

    When ``duration_seconds`` is unknown, all events are returned (the caller may
    fall back to full transcription).
    """
    if duration_seconds is None or duration_seconds <= 0:
        return list(events)
    window = window_minutes * 60.0
    kept: list[PgsEvent] = []
    for event in events:
        if event.start_seconds <= window or (duration_seconds - event.start_seconds) <= window:
            kept.append(event)
    return kept


# ---------------------------------------------------------------------------
# Writer (used by tests/fixtures and potential export features)
# ---------------------------------------------------------------------------


def build_sup_segment(
    segment_type: int,
    payload: bytes,
    *,
    pts_seconds: float,
    dts_seconds: float | None = None,
) -> bytes:
    """Serialize one segment in the ``.sup`` container format."""
    pts = int(round(pts_seconds * _90K)) & 0xFFFFFFFF
    dts = int(round((dts_seconds or pts_seconds) * _90K)) & 0xFFFFFFFF
    header = SUP_MAGIC + struct.pack(">II", pts, dts) + struct.pack(">BH", segment_type, len(payload))
    return header + payload


def build_pcs_payload(
    *,
    video_width: int = 1920,
    video_height: int = 1080,
    frame_rate_code: int = 0x30,
    composition_number: int = 0,
    composition_state: CompositionState = CompositionState.EPOCH_START,
    palette_update: bool = False,
    palette_id: int = 0,
    objects: Sequence[CompositionObject] = (),
) -> bytes:
    """Serialize a PCS payload (inverse of :func:`parse_pcs`)."""
    out = bytearray()
    out += struct.pack(">HH", video_width, video_height)
    out.append(frame_rate_code & 0x0F)
    out += struct.pack(">H", composition_number)
    out.append(int(composition_state))
    out.append(0x80 if palette_update else 0x00)
    out.append(palette_id)
    out.append(len(objects))
    for placement in objects:
        flags = 0x00
        if placement.crop is not None:
            flags |= 0x80
        if placement.forced:
            flags |= 0x40
        out += struct.pack(">HB", placement.object_id, placement.window_id)
        out.append(flags)
        if placement.crop is not None:
            out += struct.pack(">HHHH", *placement.crop)
        out += struct.pack(">HH", placement.x, placement.y)
    return bytes(out)


def build_wds_payload(windows: Sequence[WindowDefinition]) -> bytes:
    """Serialize a WDS payload (inverse of :func:`parse_wds`)."""
    out = bytearray()
    out.append(len(windows))
    for window in windows:
        out.append(window.window_id)
        out += struct.pack(">HHHH", window.x, window.y, window.width, window.height)
    return bytes(out)


def build_pds_payload(palette: Palette) -> bytes:
    """Serialize a PDS payload (inverse of :func:`parse_pds`)."""
    out = bytearray()
    out.append(palette.palette_id)
    out.append(palette.version)
    for color_id in sorted(palette.entries):
        entry = palette.entries[color_id]
        out += bytes((entry.color_id, entry.y, entry.cr, entry.cb, entry.alpha))
    return bytes(out)


def build_ods_payloads(
    object_id: int,
    version: int,
    width: int,
    height: int,
    rle_data: bytes,
    *,
    max_fragment: int = 65_000,
) -> list[bytes]:
    """Serialize one ODS object into one or more fragment payloads.

    The declared ``object_data_length`` includes the 4 bytes of the width/height
    pair (matching real Blu-ray streams); i.e. ``declared = len(rle_data) + 4``.
    """
    header = struct.pack(">HB", object_id, version)
    declared_length = len(rle_data) + 4  # width (2) + height (2) + RLE
    if len(rle_data) <= max_fragment:
        seq = 0x80 | 0x40  # first + last
        payload = header + bytes((seq,)) + declared_length.to_bytes(3, "big") + struct.pack(">HH", width, height) + rle_data
        return [payload]
    chunks = [rle_data[i : i + max_fragment] for i in range(0, len(rle_data), max_fragment)]
    payloads: list[bytes] = []
    for index, chunk in enumerate(chunks):
        first = index == 0
        last = index == len(chunks) - 1
        seq = (0x80 if first else 0x00) | (0x40 if last else 0x00)
        if first:
            payloads.append(
                header + bytes((seq,)) + declared_length.to_bytes(3, "big") + struct.pack(">HH", width, height) + chunk
            )
        else:
            payloads.append(header + bytes((seq,)) + chunk)
    return payloads


def build_sup_event(
    *,
    start_seconds: float,
    end_seconds: float,
    bitmaps: Sequence[PgsBitmap],
    palette: Palette,
    video_width: int = 1920,
    video_height: int = 1080,
    composition_number: int = 0,
    frame_rate_code: int = 0x30,
) -> bytes:
    """Build a complete display set (PCS+WDS+PDS+ODS+END) as ``.sup`` bytes.

    The event ends at ``end_seconds`` by starting the *next* epoch one frame after
    it; callers typically concatenate events, where each event's PCS carries its
    own start PTS and a clearing PCS is appended at ``end_seconds``.
    """
    rle_objects: dict[int, ObjectDefinition] = {}
    out = bytearray()

    placements: list[CompositionObject] = []
    for index, bitmap in enumerate(bitmaps):
        placements.append(
            CompositionObject(
                object_id=bitmap.object_id,
                window_id=index,
                forced=bitmap.forced,
                x=bitmap.x,
                y=bitmap.y,
            )
        )
        rle = encode_rle(bitmap.indices, bitmap.width, bitmap.height)
        rle_objects[bitmap.object_id] = ObjectDefinition(
            object_id=bitmap.object_id,
            version=0,
            width=bitmap.width,
            height=bitmap.height,
            rle_data=bytearray(rle),
            _complete=True,
        )

    pcs = build_pcs_payload(
        video_width=video_width,
        video_height=video_height,
        frame_rate_code=frame_rate_code,
        composition_number=composition_number,
        composition_state=CompositionState.EPOCH_START,
        palette_id=palette.palette_id,
        objects=placements,
    )
    out += build_sup_segment(SegmentType.PCS, pcs, pts_seconds=start_seconds)

    windows = [
        WindowDefinition(window_id=index, x=b.x, y=b.y, width=b.width, height=b.height)
        for index, b in enumerate(bitmaps)
    ]
    out += build_sup_segment(SegmentType.WDS, build_wds_payload(windows), pts_seconds=start_seconds)
    out += build_sup_segment(SegmentType.PDS, build_pds_payload(palette), pts_seconds=start_seconds)
    for object_id, obj in rle_objects.items():
        for payload in build_ods_payloads(object_id, obj.version, obj.width, obj.height, bytes(obj.rle_data)):
            out += build_sup_segment(SegmentType.ODS, payload, pts_seconds=start_seconds)
    out += build_sup_segment(SegmentType.END, b"", pts_seconds=start_seconds)

    # Empty clearing PCS that ends the subtitle at end_seconds (its own display set).
    clear_pcs = build_pcs_payload(
        video_width=video_width,
        video_height=video_height,
        frame_rate_code=frame_rate_code,
        composition_number=composition_number + 1,
        composition_state=CompositionState.NORMAL_CASE,
        palette_id=palette.palette_id,
        objects=[],
    )
    out += build_sup_segment(SegmentType.PCS, clear_pcs, pts_seconds=end_seconds)
    out += build_sup_segment(SegmentType.END, b"", pts_seconds=end_seconds)
    return bytes(out)


# ---------------------------------------------------------------------------
# OCR-friendly bitmap export
# ---------------------------------------------------------------------------


def events_to_subtitle_lines(
    events: Sequence[PgsEvent],
    *,
    lut_cache: dict[int, list[tuple[int, int, int, int]]] | None = None,
) -> list[tuple[float, float, list[PgsBitmap]]]:
    """Group events into (start, end, bitmaps) tuples ready for OCR pipelines."""
    return [(event.start_seconds, event.end_seconds, list(event.bitmaps)) for event in events if not event.is_empty]


def _read_sup_stream(fileobj: BytesIO) -> list[PgsEvent]:
    """Parse from a file-like object (helper for streaming callers)."""
    return parse_sup_events(fileobj.getvalue())
