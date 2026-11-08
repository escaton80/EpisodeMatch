"""Tests for the native PGS (Blu-ray subtitle) binary parser."""

from __future__ import annotations

import logging
import struct

import pytest

from episodematch.core.exceptions import PgsParseError
from episodematch.extractors import pgs_parser as pgs
from tests.conftest import make_bitmap, make_sup_bytes

# ---------------------------------------------------------------------------
# RLE codec
# ---------------------------------------------------------------------------


class TestRleDecode:
    def test_single_nonzero_pixels(self) -> None:
        # Each non-zero byte is exactly one pixel of that color index;
        # literal zero pixels are encoded as "00 01" (run of 1 in color 0).
        data = bytes([3, 0, 0x01, 0, 0x01, 7, 9])
        out = pgs.decode_rle(data, 5, 1)
        assert bytes(out) == bytes([3, 0, 0, 7, 9])

    def test_zero_run_short(self) -> None:
        # 00 n (0x01..0x3F) -> n transparent pixels, then EOL
        data = bytes([0x00, 0x05, 0x02, 0x00, 0x02, 0x00, 0x00])
        out = pgs.decode_rle(data, 8, 1)
        assert bytes(out) == bytes([0, 0, 0, 0, 0, 2, 0, 0])

    def test_zero_run_extended_14bit(self) -> None:
        # 00 (0x40|hi) lo -> ((hi<<8)|lo) transparent pixels, then EOL
        count = 0x0123
        data = bytes([0x00, 0x40 | (count >> 8), count & 0xFF]) + bytes([0x00, 0x00])
        out = pgs.decode_rle(data, count, 1)
        assert bytes(out) == bytes([0] * count)

    def test_zero_run_with_explicit_color(self) -> None:
        # 00 flags with 0x80 set -> explicit color byte follows, then EOL
        data = bytes([0x00, 0x80 | 0x04, 0x0B]) + bytes([0x00, 0x00])
        out = pgs.decode_rle(data, 4, 1)
        assert bytes(out) == bytes([11, 11, 11, 11])

    def test_end_of_line_advances_row(self) -> None:
        data = bytes([1, 2, 0, 0, 3, 4])  # row 0: 1,2 then EOL; row 1: 3,4
        out = pgs.decode_rle(data, 2, 2)
        assert bytes(out) == bytes([1, 2, 3, 4])

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(PgsParseError):
            pgs.decode_rle(bytes([1, 2, 3]), 4, 2)

    def test_truncated_extended_run_raises(self) -> None:
        with pytest.raises(PgsParseError):
            pgs.decode_rle(bytes([0x00, 0x40]), 4, 1)

    def test_invalid_dimensions_raise(self) -> None:
        with pytest.raises(PgsParseError):
            pgs.decode_rle(bytes([0, 0]), 0, 5)

    def test_run_overflow_is_clipped(self) -> None:
        # A run longer than the remaining area is clipped, not fatal.
        count = 10
        data = bytes([0x00, 0x00 | count]) + bytes([0x00, 0x00])
        out = pgs.decode_rle(data, 8, 1)
        assert len(out) == 8 and all(v == 0 for v in out)


class TestRleEncodeRoundtrip:
    def test_roundtrip_solid_lines(self) -> None:
        indices = bytes([1] * 20 + [0] * 20)
        encoded = pgs.encode_rle(indices, 20, 2)
        assert bytes(pgs.decode_rle(encoded, 20, 2)) == indices

    def test_roundtrip_random_bitmaps(self) -> None:
        import random

        rng = random.Random(1234)
        for _ in range(20):
            width, height = rng.randint(4, 60), rng.randint(2, 10)
            indices = bytes(rng.choice([0, 0, 0, 1, 2, 3, 15]) for _ in range(width * height))
            encoded = pgs.encode_rle(indices, width, height)
            assert bytes(pgs.decode_rle(encoded, width, height)) == indices

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            pgs.encode_rle(bytes([0, 0]), 3, 1)


# ---------------------------------------------------------------------------
# YCbCr -> RGB
# ---------------------------------------------------------------------------


class TestYcbcrConversion:
    def test_bt601_green(self) -> None:
        r, g, b = pgs.ycbcr_to_rgb(145, 34, 54, matrix="bt601")
        assert g > 200 and r < 60 and b < 60

    def test_bt601_white_and_black(self) -> None:
        assert pgs.ycbcr_to_rgb(235, 128, 128, matrix="bt601") == (255, 255, 255)
        assert pgs.ycbcr_to_rgb(16, 128, 128, matrix="bt601") == (0, 0, 0)

    def test_bt709_differs_from_bt601_for_colors(self) -> None:
        bt601 = pgs.ycbcr_to_rgb(145, 34, 54, matrix="bt601")
        bt709 = pgs.ycbcr_to_rgb(145, 34, 54, matrix="bt709")
        assert bt601 != bt709

    def test_default_matrix_by_height(self) -> None:
        # 1080 lines -> BT.709, 576 lines -> BT.601
        hd = pgs.ycbcr_to_rgb(145, 34, 54, video_height=1080)
        sd = pgs.ycbcr_to_rgb(145, 34, 54, video_height=576)
        assert hd == pgs.ycbcr_to_rgb(145, 34, 54, matrix="bt709")
        assert sd == pgs.ycbcr_to_rgb(145, 34, 54, matrix="bt601")

    def test_unknown_matrix_raises(self) -> None:
        with pytest.raises(ValueError):
            pgs.ycbcr_to_rgb(128, 128, 128, matrix="bt2020")

    def test_alpha_passthrough_in_palette_entry(self) -> None:
        entry = pgs.PaletteEntry(color_id=1, y=235, cr=128, cb=128, alpha=200)
        rgba = entry.rgba(1080)
        assert rgba[:3] == (255, 255, 255) and rgba[3] == 200


# ---------------------------------------------------------------------------
# Segment payload parsers
# ---------------------------------------------------------------------------


class TestPcsParser:
    def test_full_pcs_with_objects(self) -> None:
        payload = bytearray()
        payload += struct.pack(">HH", 1920, 1080)
        payload.append(0x30)  # frame rate code
        payload += struct.pack(">H", 42)  # composition number
        payload.append(0x80)  # epoch start
        payload.append(0x80)  # palette update flag
        payload.append(0x00)  # palette id
        payload.append(2)  # object count
        payload += struct.pack(">HB", 5, 1) + bytes([0x40]) + struct.pack(">HH", 100, 200)  # forced object
        payload += struct.pack(">HB", 6, 2) + bytes([0x80]) + struct.pack(">HHHH", 0, 0, 100, 50)
        payload += struct.pack(">HH", 300, 400)
        pcs = pgs.parse_pcs(bytes(payload))
        assert pcs.video_width == 1920 and pcs.video_height == 1080
        assert pcs.composition_number == 42
        assert pcs.composition_state is pgs.CompositionState.EPOCH_START
        assert pcs.palette_update is True
        assert len(pcs.objects) == 2
        assert pcs.objects[0].forced is True and pcs.objects[0].x == 100
        assert pcs.objects[1].crop == (0, 0, 100, 50) and pcs.objects[1].x == 300

    def test_pcs_too_short_raises(self) -> None:
        with pytest.raises(PgsParseError):
            pgs.parse_pcs(b"\x00" * 5)

    def test_truncated_object_raises(self) -> None:
        payload = bytearray()
        payload += struct.pack(">HH", 1920, 1080) + bytes([0x30, 0x00, 0x00, 0x80, 0x00, 0x00, 1])
        payload += struct.pack(">HB", 5, 1)[:4]  # cut off the position fields
        with pytest.raises(PgsParseError):
            pgs.parse_pcs(bytes(payload))

    def test_unknown_composition_state_is_none(self) -> None:
        payload = bytearray()
        payload += struct.pack(">HH", 640, 480) + bytes([0x30, 0x00, 0x00, 0x33, 0x00, 0x00, 0])
        pcs = pgs.parse_pcs(bytes(payload))
        assert pcs.composition_state is None


class TestWdsAndPds:
    def test_parse_wds(self) -> None:
        payload = bytes([1]) + bytes([7]) + struct.pack(">HHHH", 10, 20, 300, 40)
        windows = pgs.parse_wds(payload)
        assert len(windows) == 1
        assert windows[0] == pgs.WindowDefinition(window_id=7, x=10, y=20, width=300, height=40)

    def test_parse_wds_empty(self) -> None:
        assert pgs.parse_wds(b"") == []

    def test_parse_pds_entries(self) -> None:
        payload = bytes([3, 9]) + bytes([1, 235, 128, 128, 255]) + bytes([2, 16, 128, 128, 0])
        palette = pgs.parse_pds(payload)
        assert palette.palette_id == 3 and palette.version == 9
        assert palette.entries[1].alpha == 255
        assert palette.entries[2].y == 16

    def test_parse_pds_too_short_raises(self) -> None:
        with pytest.raises(PgsParseError):
            pgs.parse_pds(b"\x01")


class TestOdsAssembler:
    def _first_fragment(self, object_id: int, width: int, height: int, rle: bytes) -> bytes:
        return (
            struct.pack(">HB", object_id, 0)
            + bytes([0xC0])  # first + last
            + len(rle).to_bytes(3, "big")
            + struct.pack(">HH", width, height)
            + rle
        )

    def test_single_fragment_completes(self) -> None:
        assembler = pgs._OdsAssembler()
        completed = assembler.feed(self._first_fragment(1, 4, 2, b"\x01\x02\x03\x04\x00\x00\x05\x06"))
        assert len(completed) == 1
        assert completed[0].object_id == 1
        assert completed[0].width == 4 and completed[0].height == 2
        assert completed[0].complete

    def test_fragmented_object_reassembles(self) -> None:
        assembler = pgs._OdsAssembler()
        header = struct.pack(">HB", 2, 0)
        body = b"\x07" * 12
        first = header + bytes([0x80]) + len(body).to_bytes(3, "big") + struct.pack(">HH", 6, 2) + body[:4]
        middle = header + bytes([0x00]) + body[4:8]
        last = header + bytes([0x40]) + body[8:]
        assert assembler.feed(first) == []
        assert assembler.feed(middle) == []
        completed = assembler.feed(last)
        assert len(completed) == 1 and bytes(completed[0].rle_data) == body

    def test_continuation_without_first_raises(self) -> None:
        assembler = pgs._OdsAssembler()
        with pytest.raises(PgsParseError):
            assembler.feed(struct.pack(">HB", 9, 0) + bytes([0x40]) + b"\x00")

    def test_length_mismatch_warns_but_completes(self, caplog: pytest.LogCaptureFixture) -> None:
        assembler = pgs._OdsAssembler()
        header = struct.pack(">HB", 3, 0)
        first = header + bytes([0xC0]) + (999).to_bytes(3, "big") + struct.pack(">HH", 2, 1) + b"\x01\x02"
        with caplog.at_level("WARNING"):
            completed = assembler.feed(first)
        assert len(completed) == 1
        assert any("differs from declared" in record.message for record in caplog.records)

    def test_declared_length_includes_width_height(self, caplog: pytest.LogCaptureFixture) -> None:
        """Real Blu-ray semantics: object_data_length includes the 4-byte width/height
        pair, so declared = len(rle) + 4 must parse WITHOUT a warning."""
        assembler = pgs._OdsAssembler()
        rle = b"\x01\x02\x03\x04\x00\x00\x05\x06"
        payload = (
            struct.pack(">HB", 7, 0)
            + bytes([0xC0])
            + (len(rle) + 4).to_bytes(3, "big")
            + struct.pack(">HH", 4, 2)
            + rle
        )
        with caplog.at_level("WARNING"):
            completed = assembler.feed(payload)
        assert len(completed) == 1
        assert completed[0].width == 4 and completed[0].height == 2
        assert bytes(completed[0].rle_data) == rle
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_declared_length_without_header_offset_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A declaration that omits the width/height bytes now produces exactly the
        live-reported off-by-4 warning (declared counted RLE bytes only)."""
        assembler = pgs._OdsAssembler()
        rle = b"\x01\x02\x03\x04\x00\x00\x05\x06"
        payload = (
            struct.pack(">HB", 8, 0)
            + bytes([0xC0])
            + len(rle).to_bytes(3, "big")  # forgot the 4 header bytes
            + struct.pack(">HH", 4, 2)
            + rle
        )
        with caplog.at_level("WARNING"):
            assembler.feed(payload)
        warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("differs from declared" in message and "(-4" in message for message in warnings)

    def test_fragmented_writer_matches_real_declaration(self, caplog: pytest.LogCaptureFixture) -> None:
        """The writer must declare len(rle)+4 like real encoders, so a fragmented
        build -> parse roundtrip emits no ODS warnings."""
        assembler = pgs._OdsAssembler()
        body = b"\x07" * 12
        payloads = pgs.build_ods_payloads(2, 0, 6, 2, body, max_fragment=4)
        assert len(payloads) == 3  # 12 bytes / max_fragment 4 -> first + 2 continuations
        completed: list[pgs.ObjectDefinition] = []
        with caplog.at_level("WARNING"):
            for payload in payloads:
                completed.extend(assembler.feed(payload))
        assert len(completed) == 1 and bytes(completed[0].rle_data) == body
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_sup_roundtrip_emits_no_ods_warnings(self, standard_palette: Palette, caplog: pytest.LogCaptureFixture) -> None:
        indices = bytes([0] * 4 + [1, 1, 0, 0] + [1, 1, 1, 0] + [0] * 4)
        bitmap = make_bitmap(indices, 4, 4, forced=True)
        sup = make_sup_bytes([(10.0, 13.0, bitmap)], standard_palette)
        with caplog.at_level("WARNING"):
            events = pgs.parse_sup_events(sup)
        assert len(events) == 1
        assert bytes(events[0].bitmaps[0].indices) == indices
        warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
        assert not any("ODS" in message for message in warnings)


# ---------------------------------------------------------------------------
# Segment container
# ---------------------------------------------------------------------------


class TestSegmentIteration:
    def test_iter_sup_segments(self) -> None:
        segment = pgs.build_sup_segment(pgs.SegmentType.END, b"", pts_seconds=1.5)
        segments = list(pgs.iter_sup_segments(segment))
        assert len(segments) == 1
        assert segments[0].type_code == 0x80
        assert segments[0].pts_seconds == pytest.approx(1.5)

    def test_bad_magic_raises(self) -> None:
        data = b"XX" + struct.pack(">II", 0, 0) + struct.pack(">BH", 0x80, 0)
        with pytest.raises(PgsParseError):
            list(pgs.iter_sup_segments(data))

    def test_truncated_header_raises(self) -> None:
        with pytest.raises(PgsParseError):
            list(pgs.iter_sup_segments(b"PG\x00\x00"))

    def test_truncated_payload_raises(self) -> None:
        data = pgs.build_sup_segment(pgs.SegmentType.PCS, b"\x00" * 20, pts_seconds=0.0)[:-5]
        with pytest.raises(PgsParseError):
            list(pgs.iter_sup_segments(data))

    def test_iter_raw_segments(self) -> None:
        raw = struct.pack(">BH", 0x80, 2) + b"ok" + struct.pack(">BH", 0x16, 0)
        segments = list(pgs.iter_raw_segments(raw))
        assert [s.type_code for s in segments] == [0x80, 0x16]
        assert segments[0].pts_seconds is None


# ---------------------------------------------------------------------------
# Display sets / events / windows
# ---------------------------------------------------------------------------


class TestSupParsing:
    def test_single_event_roundtrip(self, standard_palette: Palette) -> None:
        indices = bytes([0] * 4 + [1, 1, 0, 0] + [1, 1, 1, 0] + [0] * 4)
        bitmap = make_bitmap(indices, 4, 4, forced=True)
        sup = make_sup_bytes([(10.0, 13.0, bitmap)], standard_palette)
        events = pgs.parse_sup_events(sup)
        assert len(events) == 1
        event = events[0]
        assert event.start_seconds == pytest.approx(10.0)
        assert event.end_seconds == pytest.approx(13.0)
        assert event.forced is True
        assert bytes(event.bitmaps[0].indices) == indices
        assert event.video_width == 1920 and event.video_height == 1080
        assert event.palette_lut is not None and len(event.palette_lut) == 256

    def test_two_consecutive_events_get_exact_boundaries(self, standard_palette: Palette) -> None:
        bitmap = make_bitmap(bytes([1] * 16), 4, 4)
        sup = make_sup_bytes([(5.0, 9.0, bitmap), (9.0, 12.0, bitmap)], standard_palette)
        events = pgs.parse_sup_events(sup)
        assert len(events) == 2
        assert events[0].end_seconds == pytest.approx(9.0)
        assert events[1].start_seconds == pytest.approx(9.0)
        assert events[1].end_seconds == pytest.approx(12.0)

    def test_palette_lut_reflects_palette_values(self, standard_palette: Palette) -> None:
        bitmap = make_bitmap(bytes([2] * 8), 8, 1)
        sup = make_sup_bytes([(0.0, 1.0, bitmap)], standard_palette)
        event = pgs.parse_sup_events(sup)[0]
        lut = event.lut_or_default()
        r, g, b, a = lut[2]
        assert a == 128
        # Y=128 neutral gray in BT.709 limited-range
        assert abs(r - g) <= 1 and abs(g - b) <= 1

    def test_crop_object_uses_full_bitmap_position(self, standard_palette: Palette) -> None:
        # Cropped composition objects still decode the full object bitmap.
        indices = bytes([1] * 16)
        bitmap = make_bitmap(indices, 4, 4)
        sup = make_sup_bytes([(1.0, 2.0, bitmap)], standard_palette)
        events = pgs.parse_sup_events(sup)
        assert events[0].bitmaps[0].x == 640

    def test_garbage_stream_raises(self) -> None:
        with pytest.raises(PgsParseError):
            pgs.parse_sup_events(b"\x00\x01\x02\x03" * 50)

    def test_empty_stream_returns_no_events(self) -> None:
        assert pgs.parse_sup_events(b"") == []

    def test_file_parsing(self, tmp_path, standard_palette: Palette) -> None:
        bitmap = make_bitmap(bytes([1] * 16), 4, 4)
        sup_path = tmp_path / "test.sup"
        sup_path.write_bytes(make_sup_bytes([(2.0, 4.0, bitmap)], standard_palette))
        events = pgs.PgsSupParser().parse_file(sup_path)
        assert len(events) == 1 and events[0].start_seconds == pytest.approx(2.0)


class TestWindowFilter:
    def _events_at(self, *positions: float) -> list[pgs.PgsEvent]:
        return [
            pgs.PgsEvent(start_seconds=pos, end_seconds=pos + 1.0, forced=False, bitmaps=(), video_width=1920, video_height=1080, composition_number=0)
            for pos in positions
        ]

    def test_first_and_last_windows_kept(self) -> None:
        duration = 45 * 60.0
        events = self._events_at(60.0, 20 * 60.0, duration - 60.0)
        kept = pgs.filter_events_window(events, duration_seconds=duration, window_minutes=10.0)
        assert [event.start_seconds for event in kept] == [60.0, duration - 60.0]

    def test_unknown_duration_keeps_all(self) -> None:
        events = self._events_at(100.0, 200.0)
        assert len(pgs.filter_events_window(events, duration_seconds=None)) == 2

    def test_custom_window(self) -> None:
        events = self._events_at(11 * 60.0)
        assert pgs.filter_events_window(events, duration_seconds=45 * 60.0, window_minutes=10.0) == []
        assert len(pgs.filter_events_window(events, duration_seconds=45 * 60.0, window_minutes=12.0)) == 1


class TestPgsBitmapRender:
    def test_rgba_bytes_and_pil_export(self, standard_palette: Palette) -> None:
        indices = bytes([0, 1, 2, 1])
        bitmap = make_bitmap(indices, 4, 1)
        lut = standard_palette.rgba_lut(1080)
        rgba = bitmap.rgba_bytes(lut)
        assert len(rgba) == 16
        assert rgba[3] == 0 and rgba[7] == 255 and rgba[11] == 128
        image = bitmap.to_pil_image(lut)
        assert image.size == (4, 1) and image.mode == "RGBA"
