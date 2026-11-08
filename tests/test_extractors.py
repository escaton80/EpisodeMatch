"""Tests for the extractors: subtitle parsers, EBML, ffprobe JSON, audio windows, frame scanner, OCR."""

from __future__ import annotations

import io
import struct

import pytest
from PIL import Image

from episodematch.core.exceptions import ExtractionError
from episodematch.core.models import MkvFile, StreamType, SubtitleFormat
from episodematch.extractors.audio_snipper import compute_audio_windows
from episodematch.extractors.ffprobe_reader import (
    parse_ffprobe_json,
    read_ebml_metadata,
)
from episodematch.extractors.frame_scanner import (
    compute_scan_windows,
    extract_credits_from_lines,
    parse_lavfi_scene_lines,
    parse_showinfo_scene_times,
    split_names,
    _downsample,
    _rms_difference,
)
from episodematch.extractors.ocr_engine import (
    NullBackend,
    OcrEngine,
    crop_image,
    prepare_for_ocr,
)
from episodematch.extractors.subtitle_extractor import (
    parse_ass,
    parse_srt,
    parse_vtt,
    sample_evenly,
    strip_markup,
)

# ---------------------------------------------------------------------------
# SRT parser
# ---------------------------------------------------------------------------


class TestSrtParser:
    def test_basic_parse(self) -> None:
        srt = "1\n00:00:01,500 --> 00:00:03,000\nHello!\n\n2\n00:00:10,000 --> 00:00:12,500\nBye."
        lines = parse_srt(srt)
        assert len(lines) == 2
        assert lines[0].text == "Hello!"
        assert lines[0].start_seconds == pytest.approx(1.5)
        assert lines[0].end_seconds == pytest.approx(3.0)
        assert lines[1].text == "Bye."

    def test_crlf_and_multiline(self) -> None:
        srt = "1\r\n00:00:05,000 --> 00:00:08,000\r\nLine one\r\nLine two\r\n"
        lines = parse_srt(srt)
        assert len(lines) == 1
        assert lines[0].text == "Line one Line two"

    def test_hi_markers_and_markup_stripped(self) -> None:
        srt = "1\n00:00:00,000 --> 00:00:02,000\n[explosion] <i>Run!</i>\n"
        lines = parse_srt(srt)
        assert lines[0].text == "Run!"

    def test_malformed_blocks_skipped(self) -> None:
        srt = "garbage block\n\n1\n00:00:01,000 --> 00:00:02,000\nGood line\n\nbad-->timestamps\nbroken\n"
        lines = parse_srt(srt)
        assert len(lines) == 1 and lines[0].text == "Good line"

    def test_dots_as_millisecond_separator(self) -> None:
        srt = "1\n00:00:01.250 --> 00:00:02.750\nDot style\n"
        lines = parse_srt(srt)
        assert lines[0].start_seconds == pytest.approx(1.25)

    def test_empty_input(self) -> None:
        assert parse_srt("") == []

    def test_counter_line_not_swallowed(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:02,000\nA\n\n2\n00:00:03,000 --> 00:00:04,000\nB\n"
        lines = parse_srt(srt)
        assert [line.text for line in lines] == ["A", "B"]


class TestAssParser:
    def test_dialogue_with_override_tags(self) -> None:
        ass = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            r"Dialogue: 0,0:01:02.50,0:01:05.00,Default,,0,0,0,,{\i1}Hi{\i0} there\Nfriend" + "\n"
        )
        lines = parse_ass(ass)
        assert len(lines) == 1
        assert lines[0].text == "Hi there friend"
        assert lines[0].start_seconds == pytest.approx(62.5)
        assert lines[0].end_seconds == pytest.approx(65.0)

    def test_dialogues_sorted_by_start(self) -> None:
        ass = (
            "[Events]\n"
            r"Dialogue: 0,0:00:10.00,0:00:11.00,Default,,0,0,0,,Second" + "\n"
            r"Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,First" + "\n"
        )
        lines = parse_ass(ass)
        assert [line.text for line in lines] == ["First", "Second"]

    def test_non_dialogue_lines_ignored(self) -> None:
        assert parse_ass("Comment: 0,0:00:00.00,0:00:01.00,,,,,,nope\n[Fonts]\n") == []


class TestVttParser:
    def test_header_and_cue_settings(self) -> None:
        vtt = "WEBVTT\n\n00:00:05.000 --> 00:00:07.000 position:50% align:center\nHello VTT\n"
        lines = parse_vtt(vtt)
        assert len(lines) == 1 and lines[0].text == "Hello VTT"
        assert lines[0].start_seconds == pytest.approx(5.0)

    def test_note_blocks_ignored(self) -> None:
        vtt = "WEBVTT\n\nNOTE this is a note\nwith two lines\n\n00:00:01.000 --> 00:00:02.000\nHi\n"
        lines = parse_vtt(vtt)
        assert len(lines) == 1 and lines[0].text == "Hi"


class TestStripMarkup:
    def test_ass_tags(self) -> None:
        assert strip_markup(r"{\an8}Text{\r}") == "Text"

    def test_keeps_hi_markers_when_asked(self) -> None:
        assert strip_markup("[music] play", keep_hi_markers=True).startswith("[music]")

    def test_html_tags(self) -> None:
        assert strip_markup("<b>bold</b> and <font color=\"red\">red</font>") == "bold and red"


class TestSampleEvenly:
    def test_no_sampling_needed(self) -> None:
        items = list(range(5))
        assert sample_evenly(items, 10) == items

    def test_even_sampling(self) -> None:
        items = list(range(100))
        sampled = sample_evenly(items, 10)
        assert len(sampled) == 10
        assert sampled[0] == 0 and sampled[-1] == items[-1]

    def test_zero_cap_returns_all(self) -> None:
        items = [1, 2, 3]
        assert sample_evenly(items, 0) == items


class TestLimitFirstN:
    """The strict Python-PGS fallback guard: never scan a whole film."""

    def _events(self, count: int) -> list:
        from episodematch.extractors.pgs_parser import PgsEvent

        return [
            PgsEvent(
                start_seconds=float(index) * 2.0,
                end_seconds=float(index) * 2.0 + 1.5,
                forced=False,
                bitmaps=(),
                video_width=1920,
                video_height=1080,
                composition_number=index,
            )
            for index in range(count)
        ]

    def test_under_cap_returns_all(self) -> None:
        from episodematch.extractors.subtitle_extractor import limit_first_n

        events = self._events(10)
        assert limit_first_n(events, 40) == events

    def test_caps_chronologically(self) -> None:
        from episodematch.extractors.subtitle_extractor import limit_first_n

        events = self._events(100)
        kept = limit_first_n(events, 40)
        assert len(kept) == 40
        assert kept[0].start_seconds == 0.0
        assert kept[-1].start_seconds == pytest.approx(78.0)  # event 39, not a spread

    def test_unsorted_input_sorted_first(self) -> None:
        from episodematch.extractors.subtitle_extractor import limit_first_n

        events = list(reversed(self._events(50)))
        kept = limit_first_n(events, 3)
        assert [event.composition_number for event in kept] == [0, 1, 2]

    def test_zero_cap_returns_empty(self) -> None:
        from episodematch.extractors.subtitle_extractor import limit_first_n

        assert limit_first_n(self._events(5), 0) == []


# ---------------------------------------------------------------------------
# ffprobe JSON parsing
# ---------------------------------------------------------------------------


class TestFfprobeJson:
    def test_full_document(self, sample_ffprobe_json: dict) -> None:
        mkv = parse_ffprobe_json(sample_ffprobe_json, path="X:/rips/title_t00.mkv")
        assert mkv.duration_seconds == pytest.approx(1273.512)
        assert mkv.size_bytes == 2028240896
        assert mkv.title_tag == "00800.mpls"
        assert mkv.source_id == "00800.mpls"
        assert mkv.container_name == "matroska,webm"
        assert len(mkv.streams) == 5
        assert len(mkv.chapters) == 2

    def test_stream_properties(self, sample_ffprobe_json: dict) -> None:
        mkv = parse_ffprobe_json(sample_ffprobe_json, path="title_t00.mkv")
        video = mkv.video_streams[0]
        assert video.stream_type is StreamType.VIDEO and video.width == 1920 and video.default
        audio = mkv.audio_streams
        assert audio[0].language == "eng" and audio[0].channels == 6
        assert audio[1].commentary is True
        subs = mkv.subtitle_streams
        assert subs[0].subtitle_format is SubtitleFormat.PGS and subs[0].forced
        assert subs[1].subtitle_format is SubtitleFormat.SRT
        assert len(mkv.pgs_subtitle_streams) == 1
        assert len(mkv.text_subtitle_streams) == 1

    def test_chapters(self, sample_ffprobe_json: dict) -> None:
        mkv = parse_ffprobe_json(sample_ffprobe_json, path="t.mkv")
        assert mkv.chapters[0].title == "Chapter 1"
        assert mkv.chapters[0].start_seconds == 0.0
        assert mkv.chapters[1].end_seconds == pytest.approx(1273.512)

    def test_source_id_from_mpls_like_title(self) -> None:
        payload = {"format": {"duration": "10", "size": "1", "tags": {"Title": "BDMV 00421.mpls"}}}
        mkv = parse_ffprobe_json(payload, path="x.mkv")
        assert mkv.source_id == "00421.mpls"

    def test_empty_payload(self) -> None:
        mkv = parse_ffprobe_json({}, path="x.mkv")
        assert mkv.duration_seconds is None and mkv.streams == []


# ---------------------------------------------------------------------------
# Native EBML parser
# ---------------------------------------------------------------------------


def _ebml_element(element_id: int, payload: bytes) -> bytes:
    size = len(payload)
    if size < 127:
        size_bytes = bytes([0x80 | size])
    elif size < 16383:
        size_bytes = bytes([0x40 | (size >> 8), size & 0xFF])
    else:
        size_bytes = bytes([(size >> 16) | 0x20, (size >> 8) & 0xFF, size & 0xFF])
    return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big") + size_bytes + payload


class TestEbmlReader:
    def _mini_mkv(self, *, title: str = "00800.mpls", duration_ticks: float = 123_500.0) -> bytes:
        header = _ebml_element(0x1A45DFA3, _ebml_element(0x4282, b"matroska"))
        info = _ebml_element(
            0x1549A966,
            _ebml_element(0x2AD7B1, (1_000_000).to_bytes(3, "big"))
            + _ebml_element(0x7BA9, title.encode())
            + _ebml_element(0x4489, struct.pack(">d", duration_ticks))
            + _ebml_element(0x73A4, bytes(range(16))),
        )
        track = _ebml_element(
            0xAE,
            _ebml_element(0xD7, b"\x81")
            + _ebml_element(0x83, b"\x11")
            + _ebml_element(0x86, b"S_HDMV/PGS")
            + _ebml_element(0x536E, b"German forced")
            + _ebml_element(0x55AA, b"\x01"),
        )
        tracks = _ebml_element(0x1654AE6B, track)
        cluster = _ebml_element(0x1F43B675, _ebml_element(0xE7, b"\x00"))
        return header + _ebml_element(0x18538067, info + tracks + cluster)

    def test_header_metadata(self, tmp_path) -> None:
        path = tmp_path / "mini.mkv"
        path.write_bytes(self._mini_mkv())
        meta = read_ebml_metadata(path)
        assert meta.title == "00800.mpls"
        assert meta.duration_seconds == pytest.approx(123.5)
        assert meta.segment_uid == "000102030405060708090a0b0c0d0e0f"
        assert meta.timestamp_scale == 1_000_000

    def test_tracks(self, tmp_path) -> None:
        path = tmp_path / "mini.mkv"
        path.write_bytes(self._mini_mkv())
        meta = read_ebml_metadata(path)
        assert len(meta.tracks) == 1
        track = meta.tracks[0]
        assert track.track_type == "subtitle"
        assert track.codec_id == "S_HDMV/PGS"
        assert track.language is None and track.name == "German forced"
        assert track.forced is True and track.default is False

    def test_stops_at_cluster(self, tmp_path) -> None:
        path = tmp_path / "mini.mkv"
        path.write_bytes(self._mini_mkv())
        meta = read_ebml_metadata(path)
        assert meta.muxing_app is None  # never reached other elements

    def test_not_matroska_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.mkv"
        path.write_bytes(b"\x00" * 32)
        with pytest.raises(ExtractionError):
            read_ebml_metadata(path)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ExtractionError):
            read_ebml_metadata(tmp_path / "nope.mkv")


# ---------------------------------------------------------------------------
# Audio windows
# ---------------------------------------------------------------------------


class TestAudioWindows:
    def test_two_windows_for_long_file(self) -> None:
        windows = compute_audio_windows(2400.0)
        assert len(windows) == 2
        first, second = windows
        assert first == (150.0, 270.0)
        # Second window starts 5 minutes before the (2s-trimmed) end, 2 minutes long.
        assert second[0] == pytest.approx(2400.0 - 2.0 - 300.0)
        assert second[1] == pytest.approx(2400.0 - 2.0 - 300.0 + 120.0)

    def test_unknown_duration_single_window(self) -> None:
        windows = compute_audio_windows(None)
        assert windows == [(150.0, 270.0)]

    def test_short_file_only_first_window(self) -> None:
        windows = compute_audio_windows(50.0)
        assert len(windows) == 1

    def test_overlapping_windows_dropped(self) -> None:
        # A 280 s file: end window would overlap the first window -> only one window.
        windows = compute_audio_windows(280.0, minutes_before_end=300.0)
        assert len(windows) == 1

    def test_clamping_negative_start(self) -> None:
        windows = compute_audio_windows(30.0, window_1_start=-5.0, window_1_end=10.0)
        assert windows == [(0.0, 10.0)]


# ---------------------------------------------------------------------------
# Frame scanner pure logic
# ---------------------------------------------------------------------------


class TestCreditExtraction:
    def test_directed_and_written(self) -> None:
        result = extract_credits_from_lines(["Directed by Michelle MacLaren", "Written by Vince Gilligan"])
        assert result.directors == ["Michelle MacLaren"]
        assert result.writers == ["Vince Gilligan"]
        assert result.kind.value == "credits"

    def test_guest_starring_with_as_role(self) -> None:
        result = extract_credits_from_lines(["Guest Starring Bryan Cranston as Walter Blake"])
        assert result.guest_stars == ["Bryan Cranston"]
        assert result.kind.value == "credits"

    def test_title_card_detection(self) -> None:
        result = extract_credits_from_lines(["THE LONG NIGHT"])
        assert result.kind.value == "title_card"
        assert "THE LONG NIGHT" in result.title_candidates

    def test_episode_hint_line(self) -> None:
        result = extract_credits_from_lines(["Episode: The Betrayal"])
        assert "The Betrayal" in result.title_candidates

    def test_unknown_when_nothing_matches(self) -> None:
        result = extract_credits_from_lines(["A random sentence with no credit semantics here"])
        assert result.kind.value == "unknown"
        assert not result.has_credits

    def test_multi_name_credit_line(self) -> None:
        result = extract_credits_from_lines(["Written by Vince Gilligan and Peter Gould"])
        assert result.writers == ["Vince Gilligan", "Peter Gould"]

    def test_height_boost_for_big_text(self) -> None:
        result = extract_credits_from_lines(["Quick"], line_heights=[120])
        assert "Quick" in result.title_candidates


class TestSplitNames:
    def test_and_separator(self) -> None:
        assert split_names("Vince Gilligan and Peter Gould") == ["Vince Gilligan", "Peter Gould"]

    def test_noise_word_trimming(self) -> None:
        assert split_names("Jane Doe produced") == ["Jane Doe"]

    def test_rejects_too_long(self) -> None:
        assert split_names("This Is Way Too Many Names For One Credit Here") == []


class TestSceneParsing:
    def test_lavfi_csv(self) -> None:
        output = "frame,12.345678\nframe,42.000000\n"
        assert parse_lavfi_scene_lines(output) == [12.345678, 42.0]

    def test_lavfi_csv_with_columns(self) -> None:
        output = "frame,1,0,12.345,1\n"
        times = parse_lavfi_scene_lines(output)
        assert 12.345 in times

    def test_empty_output(self) -> None:
        assert parse_lavfi_scene_lines("") == []

    def test_showinfo_lines(self) -> None:
        log = "[Parsed_showinfo_1 @ 0x0] n:0 pts_time:3.5 pos:123\n[Parsed_showinfo_1 @ 0x0] n:1 pts_time:8.25 pos:456"
        assert parse_showinfo_scene_times(log) == [3.5, 8.25]

    def test_scan_windows_long_movie(self) -> None:
        windows = compute_scan_windows(2400.0)
        assert windows.title_card_start == 0.0
        assert windows.title_card_end == 300.0
        assert windows.credits_start == pytest.approx(2400.0 * 0.88, abs=1.0)
        assert windows.credits_end == pytest.approx(2400.0)

    def test_scan_windows_unknown_duration(self) -> None:
        windows = compute_scan_windows(None)
        assert windows.title_card_end == 300.0


class TestFrameSimilarity:
    def _png(self, tmp_path, name: str, color: int) -> object:
        path = tmp_path / name
        Image.new("L", (64, 36), color).save(path)
        return path

    def test_identical_frames_zero_difference(self, tmp_path) -> None:
        one, two = self._png(tmp_path, "a.png", 128), self._png(tmp_path, "b.png", 128)
        key1, pixels1 = _downsample(one)
        _key2, pixels2 = _downsample(two)
        assert _rms_difference(pixels1, pixels2) == 0.0

    def test_different_frames_big_difference(self, tmp_path) -> None:
        black, white = self._png(tmp_path, "b.png", 0), self._png(tmp_path, "w.png", 255)
        _k1, p1 = _downsample(black)
        _k2, p2 = _downsample(white)
        assert _rms_difference(p1, p2) == pytest.approx(255.0, abs=1.0)

    def test_downsample_size(self, tmp_path) -> None:
        key, pixels = _downsample(self._png(tmp_path, "a.png", 10))
        assert key == (32, 18) and len(pixels) == 32 * 18


# ---------------------------------------------------------------------------
# OCR engine plumbing
# ---------------------------------------------------------------------------


class TestOcrPlumbing:
    def test_prepare_for_ocr_upscales_small_images(self) -> None:
        prepared = prepare_for_ocr(Image.new("RGB", (10, 8)))
        assert prepared.height >= 28 and prepared.mode == "RGB"

    def test_prepare_for_ocr_keeps_large_images(self) -> None:
        image = Image.new("RGB", (800, 200))
        assert prepare_for_ocr(image).size == (800, 200)

    def test_crop_image_clamps(self) -> None:
        image = Image.new("RGB", (100, 100))
        cropped = crop_image(image, 90, 90, 50, 50)
        assert cropped.size == (10, 10)

    def test_null_backend_raises(self) -> None:
        backend = NullBackend(reason="testing")
        with pytest.raises(Exception):
            backend.recognize(Image.new("RGB", (10, 10)))

    def test_engine_falls_back_to_null(self, config, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the "no backend installed" path even on machines with real OCR.
        monkeypatch.setattr("episodematch.extractors.ocr_engine._BACKEND_FACTORIES", ())
        engine = OcrEngine(bin_dir=config.resolved_bin_dir)
        availability = engine.availability
        assert availability.available is False
        assert engine.available is False
        with pytest.raises(Exception):
            engine.recognize(Image.new("RGB", (10, 10)))

    def test_engine_uses_installed_backend_when_present(self, config) -> None:
        # On a machine with any OCR extra installed this probes the real chain;
        # otherwise the engine honestly reports Null. Both outcomes must be consistent.
        engine = OcrEngine(bin_dir=config.resolved_bin_dir)
        availability = engine.availability
        assert availability.available == engine.available
        assert (availability.name == "null") == (not availability.available)

    def test_engine_prefers_requested_backend(self, config) -> None:
        from tests.conftest import FakeOcrBackend

        engine = OcrEngine(bin_dir=config.resolved_bin_dir)
        engine._backend = FakeOcrBackend(["hello"])  # inject fake directly
        result = engine.recognize(Image.new("RGB", (120, 60)))
        assert result.backend == "fake"
        assert result.lines[0].text == "hello"

    def test_subtitle_formats_classification(self) -> None:
        from episodematch.extractors.ffprobe_reader import classify_subtitle_format

        assert classify_subtitle_format("hdmv_pgs_subtitle") is SubtitleFormat.PGS
        assert classify_subtitle_format("subrip") is SubtitleFormat.SRT
        assert classify_subtitle_format("ass") is SubtitleFormat.ASS
        assert classify_subtitle_format("dvd_subtitle") is SubtitleFormat.VOB
        assert classify_subtitle_format(None) is None

    def test_mkv_file_helpers(self, sample_ffprobe_json: dict) -> None:
        mkv = parse_ffprobe_json(sample_ffprobe_json, path="t.mkv")
        assert mkv.file_name == "t.mkv"
        assert mkv.subtitle_stream_by_index(3) is not None
        assert mkv.subtitle_stream_by_index(99) is None
        assert len(mkv.audio_streams) == 2


# ---------------------------------------------------------------------------
# SubtitleEdit CLI fast path + capped Python fallback
# ---------------------------------------------------------------------------


def _pgs_track() -> tuple:
    from episodematch.extractors.ffprobe_reader import parse_ffprobe_json

    payload = {
        "format": {"duration": "2400", "size": "1000000"},
        "streams": [
            {"index": 0, "codec_name": "h264", "codec_type": "video"},
            {
                "index": 1,
                "codec_name": "hdmv_pgs_subtitle",
                "codec_type": "subtitle",
                "tags": {"language": "ger"},
            },
        ],
    }
    mkv = parse_ffprobe_json(payload, path="D:/rips/title_t00.mkv")
    return mkv, mkv.subtitle_streams[0]


class TestSubtitleEditFastPath:
    def _extractor(self, tmp_path, se_path) -> object:
        from episodematch.extractors.subtitle_extractor import SubtitleExtractor

        return SubtitleExtractor(
            mkvextract_path=None,
            ffmpeg_path=None,
            ocr=None,  # SE path must work WITHOUT any Python OCR backend
            work_dir=tmp_path / "subs",
            subtitle_edit_path=se_path,
        )

    def test_se_cli_conversion_preferred(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When SubtitleEdit.exe exists, the sup is converted natively and the
        Python PGS pipeline never runs (works with ocr=None)."""
        from episodematch.extractors.subtitle_extractor import SubtitleExtractor

        se_exe = tmp_path / "bin" / "SubtitleEdit.exe"
        se_exe.parent.mkdir(parents=True)
        se_exe.write_bytes(b"fake")

        sup_path = tmp_path / "subs" / "title_t00.track1.sup"
        sup_path.parent.mkdir(parents=True)
        sup_path.write_bytes(b"PG-fake")
        srt_path = sup_path.with_suffix(".srt")
        srt_payload = (
            "1\n00:00:01,000 --> 00:00:03,500\nErste Zeile vom Blu-ray Stream\n\n"
            "2\n00:00:10,000 --> 00:00:12,000\nZweite Zeile\n"
        )

        calls: list[list[str]] = []

        def fake_run(command, *args, **kwargs):
            calls.append(list(command))
            srt_path.write_text(srt_payload, encoding="utf-8")
            import subprocess

            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        monkeypatch.setattr("episodematch.extractors.subtitle_extractor.subprocess.run", fake_run)
        monkeypatch.setattr(
            SubtitleExtractor, "_dump_sup", lambda self, mkv, track: sup_path
        )

        extractor = self._extractor(tmp_path, se_exe)
        mkv, track = _pgs_track()
        transcript = extractor.extract(mkv, track, duration_seconds=2400.0)

        assert len(calls) == 1
        assert calls[0][1] == "/convert" and calls[0][2] == str(sup_path)
        assert transcript.source == "pgs-se-cli"
        assert [line.text for line in transcript.lines] == [
            "Erste Zeile vom Blu-ray Stream",
            "Zweite Zeile",
        ]
        assert not srt_path.exists()  # temp output cleaned up

    def test_se_failure_falls_back_to_capped_python_pipeline(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the SE CLI fails, the Python fallback runs — capped at the first
        N events and WITHOUT needing the whole film."""
        import logging

        from episodematch.extractors.pgs_parser import PgsBitmap, PgsEvent, Palette, PaletteEntry
        from episodematch.extractors.subtitle_extractor import SubtitleExtractor
        from tests.conftest import FakeOcrBackend

        se_exe = tmp_path / "SubtitleEdit.exe"
        se_exe.write_bytes(b"fake")

        sup_path = tmp_path / "subs" / "title_t00.track1.sup"
        sup_path.parent.mkdir(parents=True)
        sup_path.write_bytes(b"PG-fake")

        import subprocess

        def fake_run(command, *args, **kwargs):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="SE not happy")

        # 100 PGS events spread over the whole film
        palette = Palette(0, 0, {1: PaletteEntry(1, 235, 128, 128, 255)})
        events = [
            PgsEvent(
                start_seconds=float(index) * 30.0,
                end_seconds=float(index) * 30.0 + 2.0,
                forced=False,
                bitmaps=(PgsBitmap(0, 0, 0, 2, 1, b"\x01\x01"),),
                video_width=1920,
                video_height=1080,
                composition_number=index,
            )
            for index in range(100)
        ]
        monkeypatch.setattr("episodematch.extractors.subtitle_extractor.subprocess.run", fake_run)
        monkeypatch.setattr(SubtitleExtractor, "_dump_sup", lambda self, mkv, track: sup_path)
        monkeypatch.setattr(
            "episodematch.extractors.subtitle_extractor.PgsSupParser",
            lambda: type(
                "FakeParser",
                (),
                {"parse_file": lambda self, path: list(events)},
            )(),
        )

        ocr = FakeOcrBackend(["HELLO"])
        extractor = SubtitleExtractor(
            mkvextract_path=None,
            ffmpeg_path=None,
            ocr=ocr,
            work_dir=tmp_path / "subs",
            subtitle_edit_path=se_exe,
        )
        mkv, track = _pgs_track()
        transcript = extractor.extract(mkv, track, duration_seconds=2400.0, ocr_max_events=40)

        assert transcript.source == "pgs-ocr"
        assert len(transcript.lines) == 40  # hard cap, not the whole film
        assert ocr.calls == 40
        assert transcript.lines[0].start_seconds == 0.0
        assert transcript.lines[-1].start_seconds == pytest.approx(39 * 30.0)

    def test_no_se_and_no_ocr_raises_with_hint(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        from episodematch.core.exceptions import ExtractionError
        from episodematch.extractors.subtitle_extractor import SubtitleExtractor

        sup_path = tmp_path / "title_t00.track1.sup"
        sup_path.parent.mkdir(parents=True, exist_ok=True)
        sup_path.write_bytes(b"PG-fake")
        monkeypatch.setattr(SubtitleExtractor, "_dump_sup", lambda self, mkv, track: sup_path)

        extractor = SubtitleExtractor(
            mkvextract_path=None,
            ffmpeg_path=None,
            ocr=None,
            work_dir=tmp_path / "subs",
            subtitle_edit_path=None,
        )
        mkv, track = _pgs_track()
        with pytest.raises(ExtractionError, match="SubtitleEdit"):
            extractor.extract(mkv, track, duration_seconds=2400.0)

    def test_se_gui_timeout_kills_process_and_falls_back(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SubtitleEdit.exe opening the manual OCR dialog never finishes: the
        timeout must kill it, mark the tool broken for later tracks and fall
        back to the capped Python pipeline (which needs a working OCR)."""
        import subprocess as sp

        from episodematch.extractors.pgs_parser import Palette, PaletteEntry, PgsBitmap, build_sup_event
        from episodematch.extractors.subtitle_extractor import SubtitleExtractor
        from tests.conftest import FakeOcrBackend

        se_exe = tmp_path / "SubtitleEdit.exe"
        se_exe.write_bytes(b"fake")
        sup_path = tmp_path / "subs" / "title_t00.track1.sup"
        sup_path.parent.mkdir(parents=True, exist_ok=True)
        # A REAL sup stream so the Python fallback can actually decode it.
        palette = Palette(0, 0, {1: PaletteEntry(1, 235, 128, 128, 255)})
        bitmap = PgsBitmap(0, 640, 900, 20, 5, bytes([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20))
        sup_path.write_bytes(
            build_sup_event(start_seconds=1.0, end_seconds=3.0, bitmaps=[bitmap], palette=palette)
        )

        calls: list[list[str]] = []
        sup_bytes = sup_path.read_bytes()

        def fake_run(command, *args, **kwargs):
            calls.append(list(command))
            raise sp.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout", 0))

        def fake_dump(self, mkv, track):  # noqa: ANN001, ANN202 - recreate like the real dumper
            sup_path.write_bytes(sup_bytes)
            return sup_path

        monkeypatch.setattr("episodematch.extractors.subtitle_extractor.subprocess.run", fake_run)
        monkeypatch.setattr(SubtitleExtractor, "_dump_sup", fake_dump)

        ocr = FakeOcrBackend(["FALLBACK LINE THAT IS LONG ENOUGH"])
        extractor = SubtitleExtractor(
            mkvextract_path=None,
            ffmpeg_path=None,
            ocr=ocr,
            work_dir=tmp_path / "subs",
            subtitle_edit_path=se_exe,
            subtitle_edit_timeout=180.0,
        )
        mkv, track = _pgs_track()
        transcript = extractor.extract(mkv, track, duration_seconds=2400.0, ocr_max_events=5)

        assert calls, "SE CLI was not invoked"
        assert calls[0][1] == "/convert"
        assert extractor._se_broken is True  # tool marked broken
        assert transcript.source == "pgs-ocr"  # Python fallback produced the result
        assert transcript.lines and transcript.lines[0].start_seconds == pytest.approx(1.0)

        # Second extraction skips SE entirely (no new SE process, no new GUI window).
        transcript2 = extractor.extract(mkv, track, duration_seconds=2400.0, ocr_max_events=5)
        assert len(calls) == 1
        assert transcript2.source == "pgs-ocr"

    def test_seconv_preferred_over_subtitle_edit(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When seconv.exe exists it is used with modern --format syntax (headless
        OCR) instead of the GUI SubtitleEdit.exe."""
        import subprocess as sp

        from episodematch.extractors.subtitle_extractor import SubtitleExtractor

        seconv = tmp_path / "bin" / "SubtitleEdit" / "seconv.exe"
        seconv.parent.mkdir(parents=True, exist_ok=True)
        seconv.write_bytes(b"fake")
        se_exe = tmp_path / "SubtitleEdit.exe"
        se_exe.write_bytes(b"fake")

        sup_path = tmp_path / "subs" / "title_t00.track1.sup"
        sup_path.parent.mkdir(parents=True, exist_ok=True)
        sup_path.write_bytes(b"PG-fake")
        srt_path = sup_path.with_suffix(".srt")

        calls: list[list[str]] = []

        def fake_run(command, *args, **kwargs):
            calls.append(list(command))
            srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nSeconv line here\n", encoding="utf-8")
            return sp.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("episodematch.extractors.subtitle_extractor.subprocess.run", fake_run)
        monkeypatch.setattr(SubtitleExtractor, "_dump_sup", lambda self, mkv, track: sup_path)

        extractor = SubtitleExtractor(
            mkvextract_path=None,
            ffmpeg_path=None,
            ocr=None,
            work_dir=tmp_path / "subs",
            subtitle_edit_path=se_exe,
            seconv_path=seconv,
        )
        mkv, track = _pgs_track()
        transcript = extractor.extract(mkv, track, duration_seconds=2400.0)

        assert len(calls) == 1
        assert calls[0][0].endswith("seconv.exe")
        assert "--format" in calls[0] and "srt" in calls[0]  # modern headless syntax
        assert transcript.source == "pgs-se-cli"

    def test_seconv_receives_nocr_engine_db_and_language(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """For PGS input seconv must be told which headless OCR engine to use
        (its default is Tesseract, which is usually absent) plus the track
        language in ISO-639-2/T form."""
        import subprocess as sp

        from episodematch.extractors.subtitle_extractor import SubtitleExtractor

        seconv = tmp_path / "seconv.exe"
        seconv.write_bytes(b"fake")
        sup_path = tmp_path / "subs" / "title_t00.track1.sup"
        sup_path.parent.mkdir(parents=True, exist_ok=True)
        sup_path.write_bytes(b"PG-fake")
        srt_path = sup_path.with_suffix(".srt")

        calls: list[list[str]] = []

        def fake_run(command, *args, **kwargs):
            calls.append(list(command))
            srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
            return sp.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("episodematch.extractors.subtitle_extractor.subprocess.run", fake_run)
        monkeypatch.setattr(SubtitleExtractor, "_dump_sup", lambda self, mkv, track: sup_path)

        db = tmp_path / "Latin.nocr"
        db.write_bytes(b"fake-db")
        extractor = SubtitleExtractor(
            mkvextract_path=None,
            ffmpeg_path=None,
            ocr=None,
            work_dir=tmp_path / "subs",
            subtitle_edit_path=None,
            seconv_path=seconv,
            seconv_ocr_db=db,
        )
        mkv, track = _pgs_track()  # language "ger"
        extractor.extract(mkv, track, duration_seconds=2400.0)

        command = calls[0]
        assert "--ocr-engine" in command and command[command.index("--ocr-engine") + 1] == "nocr"
        assert "--ocr-db" in command and command[command.index("--ocr-db") + 1] == str(db)
        assert "--ocr-language" in command and command[command.index("--ocr-language") + 1] == "deu"

    def test_resolve_seconv_ocr_db_auto_probe(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        from episodematch.extractors.subtitle_extractor import resolve_seconv_ocr_db

        appdata = tmp_path / "appdata"
        ocr_dir = appdata / "Subtitle Edit" / "OCR"
        ocr_dir.mkdir(parents=True)
        (ocr_dir / "Latin.nocr").write_bytes(b"x")
        monkeypatch.setenv("APPDATA", str(appdata))
        resolution = resolve_seconv_ocr_db(None)
        assert resolution is not None
        engine, db_path = resolution
        assert engine == "nocr" and db_path == ocr_dir / "Latin.nocr"

        # explicit override wins, even when the auto-probe would find something
        override = tmp_path / "custom.db"
        override.write_bytes(b"x")
        engine2, db_path2 = resolve_seconv_ocr_db(override)
        assert engine2 == "binaryocr" and db_path2 == override

        # nothing found -> None
        import os as _os

        monkeypatch.setenv("APPDATA", str(tmp_path / "empty"))
        assert resolve_seconv_ocr_db(None) is None
        assert _os.environ.get("APPDATA")  # sanity: env restored by monkeypatch later


class TestSelectSubtitleStreams:
    """The language whitelist must keep foreign PGS tracks away from SE/OCR."""

    def _mkv_with(self, *specs: tuple[str, str]) -> MkvFile:
        """specs: (language, codec) pairs; codec 'hdmv_pgs_subtitle' or 'subrip'."""
        payload = {
            "format": {"duration": "2400", "size": "1000"},
            "streams": [
                {"index": 0, "codec_name": "h264", "codec_type": "video"},
                *[
                    {
                        "index": index + 1,
                        "codec_name": codec,
                        "codec_type": "subtitle",
                        "tags": {"language": language} if language else {},
                    }
                    for index, (language, codec) in enumerate(specs)
                ],
            ],
        }
        return parse_ffprobe_json(payload, path="disc.mkv")

    def test_typical_blu_ray_only_german_english(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        mkv = self._mkv_with(
            ("ger", "hdmv_pgs_subtitle"),
            ("eng", "hdmv_pgs_subtitle"),
            ("spa", "hdmv_pgs_subtitle"),
            ("por", "hdmv_pgs_subtitle"),
            ("fre", "subrip"),
        )
        selected = select_subtitle_streams(mkv, ["ger", "deu", "eng"])
        languages = [stream.language for stream in selected]
        assert languages == ["ger", "eng"]  # spa/por/fre never touched
        assert all(stream.subtitle_format is SubtitleFormat.PGS for stream in selected)

    def test_preferred_order_honored(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        mkv = self._mkv_with(("ger", "subrip"), ("eng", "subrip"))
        selected = select_subtitle_streams(mkv, ["eng", "ger"], max_tracks=1)
        assert [stream.language for stream in selected] == ["eng"]

    def test_pgs_tiebreak_within_same_language(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        mkv = self._mkv_with(("ger", "subrip"), ("ger", "hdmv_pgs_subtitle"))
        selected = select_subtitle_streams(mkv, ["ger"], max_tracks=1)
        assert selected[0].subtitle_format is SubtitleFormat.PGS  # native conversion preferred

    def test_no_preferred_match_falls_back_to_best_foreign(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        mkv = self._mkv_with(("jpn", "subrip"), ("kor", "hdmv_pgs_subtitle"))
        selected = select_subtitle_streams(mkv, ["ger", "eng"], max_tracks=1)
        assert len(selected) == 1
        assert selected[0].subtitle_format is SubtitleFormat.PGS  # last resort: best-ranked foreign

    def test_untagged_tracks_win_when_nothing_else_matches(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        mkv = self._mkv_with(("", "subrip"), ("spa", "subrip"))
        selected = select_subtitle_streams(mkv, ["ger", "eng"], max_tracks=2)
        assert [stream.language for stream in selected] == [None]  # mis-tagged disc

    def test_max_tracks_bounds_cost(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        mkv = self._mkv_with(("ger", "hdmv_pgs_subtitle"), ("ger", "subrip"), ("eng", "hdmv_pgs_subtitle"))
        assert len(select_subtitle_streams(mkv, ["ger", "eng"], max_tracks=2)) == 2
        assert len(select_subtitle_streams(mkv, ["ger", "eng"], max_tracks=1)) == 1

    def test_no_subtitles_returns_empty(self) -> None:
        from episodematch.extractors.subtitle_extractor import select_subtitle_streams

        assert select_subtitle_streams(self._mkv_with(), ["ger"]) == []
