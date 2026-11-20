"""Tests for direct TheDiscDB disc/release URL resolution."""

from __future__ import annotations

import json

import pytest

from episodematch.core.exceptions import ProviderError
from episodematch.core.models import DiscFingerprint
from episodematch.providers.thediscdb import (
    LocalDiscDbImporter,
    normalize_slug,
    parse_disc_url,
    resolve_discs_from_url,
)
from tests.test_pipeline import FakeFfprobe, FakeTmdb, make_guide


def make_mkv(name: str, *, mpls: str, duration: float, size: int):
    from episodematch.core.models import MkvFile, StreamInfo, StreamType

    return MkvFile(
        path=Path_stub(name),
        size_bytes=size,
        duration_seconds=duration,
        container_name="matroska",
        title_tag=mpls,
        source_id=mpls,
        streams=[StreamInfo(index=0, stream_type=StreamType.VIDEO, codec_name="h264")],
    )


def Path_stub(name: str):
    from pathlib import Path

    return Path("D:/rips") / name


class TestParseDiscUrl:
    def test_full_disc_url(self) -> None:
        parsed = parse_disc_url(
            "https://thediscdb.com/series/south-park-1997/releases/seasons-6-10-blu-ray/discs/season-06-disc-01"
        )
        assert parsed == {
            "series": "south-park-1997",
            "release": "seasons-6-10-blu-ray",
            "disc": "season-06-disc-01",
        }

    def test_release_level_url_without_disc(self) -> None:
        parsed = parse_disc_url("https://thediscdb.com/series/south-park-1997/releases/seasons-6-10-blu-ray/")
        assert parsed is not None
        assert parsed["series"] == "south-park-1997"
        assert parsed["release"] == "seasons-6-10-blu-ray"
        assert parsed["disc"] is None

    def test_www_and_trailing_slash(self) -> None:
        parsed = parse_disc_url(
            "https://www.thediscdb.com/series/24-2001/releases/season-1/discs/blu-ray/"
        )
        assert parsed is not None
        assert parsed["series"] == "24-2001"

    def test_non_discdb_url_returns_none(self) -> None:
        assert parse_disc_url("https://example.com/series/x/releases/y") is None
        assert parse_disc_url("") is None


class TestNormalizeSlug:
    def test_series_folder_matches_url_slug(self) -> None:
        assert normalize_slug("South Park (1997)") == normalize_slug("south-park-1997")

    def test_release_folders_are_slugs(self) -> None:
        assert normalize_slug("seasons-6-10-blu-ray") == "seasons610bluray"

    def test_case_and_umlaut_insensitive(self) -> None:
        assert normalize_slug("Die Ärzte (1999)") == normalize_slug("die-rzte-1999")


class TestResolveFromUrl:
    """Both clients expose find_series/list_releases/load_release_discs, so the
    resolver works identically for the GitHub client and the local importer."""

    def test_disc_level_url_loads_single_disc(self, thediscdb_repo) -> None:
        importer = LocalDiscDbImporter(thediscdb_repo)
        discs, meta = resolve_discs_from_url(
            importer,
            "https://thediscdb.com/series/test-show-2020/releases/season-1/discs/blu-ray",
        )
        assert len(discs) == 1
        assert discs[0].disc_name == "Test Show Season 1 Disc 1"
        assert meta["series_folder"] == "Test Show (2020)"
        assert meta["release_folder"] == "season-1"
        assert meta["disc_slug"] == "blu-ray"

    def test_release_level_url_loads_all_release_discs(self, thediscdb_repo) -> None:
        importer = LocalDiscDbImporter(thediscdb_repo)
        discs, meta = resolve_discs_from_url(
            importer,
            "https://thediscdb.com/series/test-show-2020/releases/season-1",
        )
        assert len(discs) == 1  # the fixture ships one discNN.json
        assert meta["disc_slug"] == ""

    def test_unknown_disc_slug_lists_available(self, thediscdb_repo) -> None:
        importer = LocalDiscDbImporter(thediscdb_repo)
        with pytest.raises(ProviderError, match="Available discs"):
            resolve_discs_from_url(
                importer,
                "https://thediscdb.com/series/test-show-2020/releases/season-1/discs/season-06-disc-01",
            )

    def test_unknown_release_lists_available_releases(self, thediscdb_repo) -> None:
        importer = LocalDiscDbImporter(thediscdb_repo)
        with pytest.raises(ProviderError, match="Available releases"):
            resolve_discs_from_url(
                importer,
                "https://thediscdb.com/series/test-show-2020/releases/season-99",
            )

    def test_series_slug_normalization(self, thediscdb_repo) -> None:
        importer = LocalDiscDbImporter(thediscdb_repo)
        discs, meta = resolve_discs_from_url(
            importer,
            "https://thediscdb.com/series/test-show-2020/releases/season-1/discs/blu-ray",
        )
        assert meta["series_folder"] == "Test Show (2020)"


class TestMakeMkvIndexMatch:
    """Live case: South Park S06D1 — MakeMKV set the disc label as title tag (no
    MPLS!), every episode is exactly 1326 s, and rip sizes are ~8% below the disc
    playlist sizes. The MakeMKV title index from the file name is the only
    conclusive signal, and with a pinned disc URL it is near-conclusive."""

    @staticmethod
    def _south_park_disc() -> list:
        from episodematch.providers.thediscdb import parse_disc_json

        payload = {
            "Name": "Season Six Disc One",
            "Slug": "season-06-disc-01",
            "Format": "Blu-Ray",
            "Titles": [
                {"Index": 0, "SourceFile": "00007.mpls", "Duration": "0:22:06", "Size": 3748982784, "Item": {"Title": "The New Terrance and Phillip Movie Trailer", "Type": "Episode", "Season": "6", "Episode": "5"}},
                {"Index": 1, "SourceFile": "00008.mpls", "Duration": "0:22:06", "Size": 4181385216, "Item": {"Title": "Red Hot Catholic Love", "Type": "Episode", "Season": "6", "Episode": "8"}},
                {"Index": 2, "SourceFile": "00000.mpls", "Duration": "0:22:06", "Size": 3680815104, "Item": {"Title": "Jared Has Aides", "Type": "Episode", "Season": "6", "Episode": "1"}},
                {"Index": 3, "SourceFile": "00002.mpls", "Duration": "3:19:01", "Size": 35459149824, "Item": {"Type": "Feature"}},
                {"Index": 5, "SourceFile": "00006.mpls", "Duration": "0:22:06", "Size": 4291983360, "Item": {"Title": "Freak Strike", "Type": "Episode", "Season": "6", "Episode": "3"}},
                {"Index": 6, "SourceFile": "00010.mpls", "Duration": "0:22:07", "Size": 4156133376, "Item": {"Title": "The Simpsons Already Did It", "Type": "Episode", "Season": "6", "Episode": "7"}},
                {"Index": 7, "SourceFile": "00011.mpls", "Duration": "0:22:06", "Size": 4298582016, "Item": {"Title": "Free Hat", "Type": "Episode", "Season": "6", "Episode": "9"}},
                {"Index": 8, "SourceFile": "00005.mpls", "Duration": "0:22:06", "Size": 3511627776, "Item": {"Title": "Asspen", "Type": "Episode", "Season": "6", "Episode": "2"}},
                {"Index": 9, "SourceFile": "00009.mpls", "Duration": "0:22:06", "Size": 4142106624, "Item": {"Title": "Professor Chaos", "Type": "Episode", "Season": "6", "Episode": "6"}},
                {"Index": 10, "SourceFile": "00004.mpls", "Duration": "0:22:06", "Size": 3447533568, "Item": {"Title": "Fun With Veal", "Type": "Episode", "Season": "6", "Episode": "4"}},
            ],
        }
        disc = parse_disc_json(payload, series_title="South Park (1997)", release_title="seasons-6-10-blu-ray", source="test")
        return [disc]

    @staticmethod
    def _rip(name: str, size: int):
        from episodematch.core.models import DiscFingerprint, MkvFile

        mkv = MkvFile(path=f"D:/Video/South Park Season 6 - Disc 1/{name}", size_bytes=size, duration_seconds=1326.8)
        return DiscFingerprint.from_mkv(mkv)

    def test_index_identifies_title_with_pinned_weight(self) -> None:
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._south_park_disc()
        # t02 -> disc index 2 = S06E01 "Jared Has Aides" (no MPLS, size useless).
        fingerprint = self._rip("South Park Season 6 - Disc 1_t02.mkv", size=3358907795)
        matches = match_fingerprint(fingerprint, discs, index_weight=0.90, pinned=True)
        assert matches[0].episode == 1
        assert matches[0].entry.episode_title == "Jared Has Aides"
        assert matches[0].score >= 0.95
        assert any("title index" in reason for reason in matches[0].reasons)

    def test_size_lookalike_does_not_beat_index(self) -> None:
        """t00's rip size is closest to index 10 (S06E4) — but the index wins:
        it maps t00 to index 0 (S06E5)."""
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._south_park_disc()
        fingerprint = self._rip("South Park Season 6 - Disc 1_t00.mkv", size=3434955687)
        matches = match_fingerprint(fingerprint, discs, index_weight=0.90, pinned=True)
        assert matches[0].episode == 5
        assert matches[0].entry.episode_title == "The New Terrance and Phillip Movie Trailer"

    def test_unpinned_index_is_only_a_tiebreak(self) -> None:
        """Without a pinned URL the index repeats across releases — only a small
        weight, so MPLS/duration still dominate the ranking."""
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._south_park_disc()
        fingerprint = self._rip("title_t02.mkv", size=3680815104)  # exact disc size
        matches = match_fingerprint(fingerprint, discs)  # default index_weight=0.25
        assert matches[0].episode == 1  # index 0.25 + duration 0.2 + size 0.1, no MPLS here
        assert matches[0].score == pytest.approx(0.51, abs=0.05)
        # and the unpinned index must not fabricate high confidence
        assert matches[0].score < 0.95

    def test_from_mkv_parses_index_variants(self) -> None:
        from episodematch.core.models import DiscFingerprint, MkvFile

        for name, expected in (
            ("title_t07", 7),
            ("South Park Season 6 - Disc 1_t10", 10),
            ("show-T03", 3),
        ):
            mkv = MkvFile(path=f"D:/x/{name}.mkv")
            assert DiscFingerprint.from_mkv(mkv).make_mkv_index == expected
        assert DiscFingerprint.from_mkv(MkvFile(path="D:/x/episode1.mkv")).make_mkv_index is None


class TestCastleRockIndexCollision:
    """Live case 2: Castle Rock S1D2 — the contributor's playlist enumeration
    differs from the rip's MakeMKV indices (E6 sits at Index 4, E10 at Index 17).
    Bare index collisions (t06 vs Index 6 = E8) produced WRONG assignments at
    0.90 while the true matches (unique durations!) sat at 0.25-0.28."""

    @staticmethod
    def _castle_rock_disc() -> list:
        from episodematch.providers.thediscdb import parse_disc_json

        payload = {
            "Name": "Season One Blu-ray Disc Two",
            "Slug": "blu-ray-2",
            "Format": "Blu-Ray",
            "Titles": [
                {"Index": 4, "SourceFile": "00206.mpls", "Duration": "0:43:35", "Size": 7256635392, "Item": {"Title": "Filter", "Type": "Episode", "Season": "1", "Episode": "6"}},
                {"Index": 5, "SourceFile": "00207.mpls", "Duration": "0:59:51", "Size": 9984196608, "Item": {"Title": "The Queen", "Type": "Episode", "Season": "1", "Episode": "7"}},
                {"Index": 6, "SourceFile": "00208.mpls", "Duration": "0:50:36", "Size": 8408008704, "Item": {"Title": "Past Perfect", "Type": "Episode", "Season": "1", "Episode": "8"}},
                {"Index": 7, "SourceFile": "00209.mpls", "Duration": "0:44:57", "Size": 7472424960, "Item": {"Title": "Henry Deaver", "Type": "Episode", "Season": "1", "Episode": "9"}},
                {"Index": 17, "SourceFile": "00210.mpls", "Duration": "0:51:47", "Size": 8869361664, "Item": {"Title": "Romans", "Type": "Episode", "Season": "1", "Episode": "10"}},
                {"Index": 8, "SourceFile": "00031.mpls", "Duration": "0:00:10", "Size": 122880, "Item": {"Type": "Menu"}},
            ],
        }
        disc = parse_disc_json(payload, series_title="Castle Rock (2018)", release_title="2019-season-1-blu-ray", source="test")
        return [disc]

    @staticmethod
    def _rip(name: str, size: int, duration: float):
        from episodematch.core.models import DiscFingerprint, MkvFile

        mkv = MkvFile(path=f"D:/Video/Castle Rock Season 1 Disc 2/{name}", size_bytes=size, duration_seconds=duration)
        return DiscFingerprint.from_mkv(mkv)

    def test_bare_index_collision_is_ignored(self) -> None:
        """t06's Index 6 collides with disc Index 6 (= E8) but the durations
        disagree (43:36 vs 50:36) — the collision must be ignored."""
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._castle_rock_disc()
        fingerprint = self._rip("Castle Rock Season 1 Disc 2_t06.mkv", size=7350674994, duration=2616.0)
        matches = match_fingerprint(fingerprint, discs, index_weight=0.90, pinned=True)
        assert matches[0].episode == 6  # by its UNIQUE duration, not the index
        assert matches[0].score >= 0.75
        reasons = " ".join(matches[0].reasons)
        assert "corroborated" not in reasons
        assert "unique on its own disc" in reasons

    def test_unique_duration_carries_the_match(self) -> None:
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._castle_rock_disc()
        for name, size, duration, expected_episode in (
            ("Castle Rock Season 1 Disc 2_t06.mkv", 7350674994, 2616.0, 6),
            ("Castle Rock Season 1 Disc 2_t07.mkv", 10104226353, 3591.879, 7),
            ("Castle Rock Season 1 Disc 2_t08.mkv", 8529659891, 3036.384, 8),
            ("Castle Rock Season 1 Disc 2_t09.mkv", 7565292576, 2697.408, 9),
            ("Castle Rock Season 1 Disc 2_t10.mkv", 8978856627, 3107.616, 10),
        ):
            fingerprint = self._rip(name, size=size, duration=duration)
            matches = match_fingerprint(fingerprint, discs, index_weight=0.90, pinned=True)
            assert matches[0].episode == expected_episode, name
            assert matches[0].score >= 0.65

    def test_corroborated_index_still_counts(self) -> None:
        """When index AND duration agree, the index stays a strong signal."""
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._castle_rock_disc()
        # A rip whose MakeMKV index (6) AND duration match E8's entry (idx 6).
        fingerprint = self._rip("Castle Rock Season 1 Disc 2_t06.mkv", size=8408008704, duration=3036.0)
        matches = match_fingerprint(fingerprint, discs, index_weight=0.90, pinned=True)
        assert matches[0].episode == 8
        assert matches[0].score >= 0.95
        assert any("corroborated" in reason for reason in matches[0].reasons)


class TestPipelineWithDiscUrl:
    def test_url_scan_loads_only_target_disc(self, tmp_path, config, thediscdb_repo) -> None:
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        mkv = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
        )
        report = pipeline.scan_folder(
            _ensure_rips(tmp_path),
            series="Test Show",
            season=1,  # matches the fixture disc; the URL still pins the loaded disc
            options=ScanOptions(
                depth="fast",
                discdb_url="https://thediscdb.com/series/test-show-2020/releases/season-1/discs/blu-ray",
            ),
        )
        assert report.assignment.assigned_count == 1
        entry = report.assignment.entries[0]
        assert entry.episode == 1
        evidence = next(e for e in entry.evidence if e.pillar.value == "disc_db")
        assert evidence.details["source"].startswith("https://thediscdb.com/")
        assert evidence.season_hint == 1

    def test_url_scan_still_pins_loading_when_season_heuristic_misses(self, tmp_path, config, thediscdb_repo) -> None:
        """The URL must pin the *loading* regardless of the requested season
        (audit A1 keeps this) — but cross-season evidence is now gated at the
        evidence level, so nothing is assigned for the wrong season."""
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        mkv = make_mkv("title_t00.mkv", mpls="00800.mpls", duration=42 * 60 + 30, size=2_028_000_000)
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(thediscdb_repo),
            tmdb=FakeTmdb(make_guide()),
        )
        report = pipeline.scan_folder(
            _ensure_rips(tmp_path),
            series="Test Show",
            season=6,  # wrong season on purpose
            options=ScanOptions(
                depth="fast",
                discdb_url="https://thediscdb.com/series/test-show-2020/releases/season-1/discs/blu-ray",
            ),
        )
        disc_evidence = [
            ev for c in report.candidates for ev in c.evidence if ev.pillar.value == "disc_db"
        ]
        assert disc_evidence == []  # season mismatch -> no evidence, no assignment
        assert report.assignment.assigned_count == 0


def _ensure_rips(tmp_path):
    from pathlib import Path

    folder = tmp_path / "rips"
    folder.mkdir(exist_ok=True)
    (folder / "title_t00.mkv").write_bytes(b"\x1a\x45\xdf\xa3fake")
    return folder


class TestSeasonMismatchGuard:
    """Audit A1: DiscDB evidence for a *different* season must never become
    evidence for the requested season (S02E01 rip was assigned as S01E01 @ 1.0)."""

    def test_disc_evidence_for_other_season_is_ignored(self, tmp_path, config):
        from episodematch.matcher.pipeline import Pipeline, ScanOptions
        from episodematch.providers.thediscdb import LocalDiscDbImporter
        from tests.test_pipeline import FakeFfprobe, FakeTmdb, make_guide

        release_dir = tmp_path / "discdb" / "data" / "series" / "Test Show (2020)" / "season-2"
        release_dir.mkdir(parents=True)
        disc_payload = {
            "Name": "Season Two Disc One", "Slug": "blu-ray-1", "Format": "Blu-Ray",
            "Titles": [{
                "Index": 0, "SourceFile": "00901.mpls", "Duration": "0:42:30",
                "Size": 2028000000,
                "Item": {"Title": "Season Two Opener", "Type": "Episode",
                         "Season": "2", "Episode": "1"},
            }],
        }
        (release_dir / "disc01.json").write_text(json.dumps(disc_payload), encoding="utf-8")

        mkv = make_mkv("title_t00.mkv", mpls="00901.mpls", duration=42 * 60 + 30, size=2028000000)
        pipeline = Pipeline(
            config,
            ffprobe=FakeFfprobe({"title_t00.mkv": mkv}),
            disc_db_client=LocalDiscDbImporter(tmp_path / "discdb"),
            tmdb=FakeTmdb(make_guide()),
        )
        report = pipeline.scan_folder(
            _ensure_rips(tmp_path),
            series="Test Show",
            season=1,  # requested season does NOT match the disc entry (S02)
            options=ScanOptions(
                depth="fast",
                discdb_url="https://thediscdb.com/series/test-show-2020/releases/season-2",
            ),
        )
        assignment = report.assignment
        assert assignment.assigned_count == 0  # no cross-season assignment
        disc_evidence = [
            ev for c in report.candidates for ev in c.evidence if ev.pillar.value == "disc_db"
        ]
        assert disc_evidence == []  # cross-season evidence must not be recorded
        messages = " | ".join(r.detail for r in report.pillar_reports if r.pillar == "disc_db")
        assert "Season 2" in messages and "ignored" in messages


class TestMplsVsIndexPinned:
    """Audit A2: with a pinned disc URL, a contradicting title index must not
    outrank an exact MPLS identity match (wrong E2 entry beat the correct E1
    entry at 1.0 vs 0.95)."""

    def test_contradicting_index_loses_to_exact_mpls(self) -> None:
        from episodematch.core.models import MkvFile
        from episodematch.providers.thediscdb import match_fingerprint, parse_disc_json

        payload = {"Name": "D", "Slug": "d", "Format": "Blu-Ray", "Titles": [
            {"Index": 2, "SourceFile": "00002.mpls", "Duration": "0:16:40", "Size": 1000,
             "Item": {"Title": "Wrong", "Type": "Episode", "Season": "1", "Episode": "2"}},
            {"Index": 1, "SourceFile": "00001.mpls", "Duration": "0:16:40", "Size": 1000,
             "Item": {"Title": "Right", "Type": "Episode", "Season": "1", "Episode": "1"}}]}
        disc = parse_disc_json(payload, series_title="X", release_title="r", source="t")
        mkv = MkvFile(path="D:/x/title_t02.mkv", size_bytes=1000, duration_seconds=1000.0,
                      title_tag="00001.mpls", source_id="00001.mpls")
        fp = DiscFingerprint.from_mkv(mkv)
        matches = match_fingerprint(fp, [disc], index_weight=0.90)
        assert matches[0].entry.episode == 1  # exact MPLS identity wins
        assert "MPLS 00001.mpls matches disc entry" in " ".join(matches[0].reasons)
        # the contradicting index match was ignored entirely
        assert all("title index" not in r for m in matches for r in m.reasons)

class TestDiscTitleFieldParsing:
    """Review 1.1/1.2: hand-entered Item fields come in two shapes the parser
    must survive — episode ranges ("1-2" = one playlist covering two episodes)
    and suggested filenames pasted into Title."""

    def _payload(self, item: dict) -> dict:
        return {
            "Titles": [
                {
                    "Index": 0,
                    "SourceFile": "00800.mpls",
                    "SegmentMap": "1-11,12",
                    "Duration": "0:43:00",
                    "Size": 2_000_000_000,
                    "Item": item,
                }
            ]
        }

    def _parse(self, item: dict):
        from episodematch.providers.thediscdb import parse_disc_json

        return parse_disc_json(
            self._payload(item), series_title="Show (2020)", release_title="season-1", source="test"
        ).titles[0]

    def test_episode_range_is_parsed_as_span(self) -> None:
        entry = self._parse({"Title": "London", "Type": "Episode", "Season": "6", "Episode": "1-2"})
        assert entry.episode == 1
        assert entry.episode_end == 2
        assert entry.is_episode  # the two-parter must be a candidate
        assert entry.episode_title == "London"

    def test_filename_like_title_is_normalized(self) -> None:
        entry = self._parse(
            {"Title": "Parks and Recreation.S06.E09.The Cones of Dunshire.mkv", "Type": "Episode", "Season": "6", "Episode": "9"}
        )
        assert entry.episode_title == "The Cones of Dunshire"

    def test_dotted_title_without_media_extension_is_untouched(self) -> None:
        """The rewrite is gated on an actual file extension — a genuine title
        containing dots must not be mangled."""
        entry = self._parse(
            {"Title": "Parks and Recreation.S06.E09.The Cones of Dunshire", "Type": "Episode", "Season": "6", "Episode": "9"}
        )
        assert entry.episode_title == "Parks and Recreation.S06.E09.The Cones of Dunshire"

    def test_plain_title_untouched(self) -> None:
        entry = self._parse({"Title": "The Cones of Dunshire", "Type": "Episode", "Season": "6", "Episode": "9"})
        assert entry.episode_title == "The Cones of Dunshire"

class TestDiscLayoutScore:
    """Review 2.2: a disc is confirmed as a unit — the fraction of its episode
    playlists present in the rip with a matching MPLS name AND a matching
    running time (both together; either alone is ambiguous)."""

    @staticmethod
    def _disc() -> list:
        from episodematch.providers.thediscdb import parse_disc_json

        payload = {
            "Name": "Season 1 Disc 1",
            "Slug": "season-1",
            "Format": "Blu-Ray",
            "Titles": [
                {"Index": 0, "SourceFile": "00800.mpls", "SegmentMap": "1-11,12", "Duration": "0:42:30", "Size": 2_028_000_000, "Item": {"Title": "Pilot", "Type": "Episode", "Season": "1", "Episode": "1"}},
                {"Index": 1, "SourceFile": "00801.mpls", "SegmentMap": "1-11,12", "Duration": "0:41:55", "Size": 2_011_000_000, "Item": {"Title": "Second", "Type": "Episode", "Season": "1", "Episode": "2"}},
                {"Index": 2, "SourceFile": "00055.mpls", "SegmentMap": "1012", "Duration": "0:00:10", "Size": 270336, "Item": {"Type": "Extra"}},
            ],
        }
        return [parse_disc_json(payload, series_title="Show (2020)", release_title="season-1", source="test")]

    @staticmethod
    def _fps(*specs: tuple[str, float]) -> list:
        from episodematch.core.models import DiscFingerprint, MkvFile

        out = []
        for index, (mpls, duration) in enumerate(specs):
            mkv = MkvFile(
                path=f"D:/rips/title_t{index:02d}.mkv",
                size_bytes=2_000_000_000,
                duration_seconds=duration,
                title_tag=mpls,
                source_id=mpls,
            )
            out.append(DiscFingerprint.from_mkv(mkv))
        return out

    def test_full_layout_scores_1(self) -> None:
        from episodematch.providers.thediscdb import score_disc_layout

        assert score_disc_layout(self._disc()[0], self._fps(("00800.mpls", 2550.0), ("00801.mpls", 2515.0))) == 1.0

    def test_partial_layout_scores_half(self) -> None:
        from episodematch.providers.thediscdb import score_disc_layout

        assert score_disc_layout(self._disc()[0], self._fps(("00800.mpls", 2550.0))) == 0.5

    def test_duration_without_matching_mpls_counts_nothing(self) -> None:
        from episodematch.providers.thediscdb import score_disc_layout

        assert score_disc_layout(self._disc()[0], self._fps(("00999.mpls", 2550.0))) == 0.0

    def test_empty_rip_scores_zero(self) -> None:
        from episodematch.providers.thediscdb import score_disc_layout

        assert score_disc_layout(self._disc()[0], []) == 0.0


class TestDurationUniquenessLadder:
    """Review 2.1: duration uniqueness is counted PER DISC, and the full
    +0.70 unique bonus only applies to a PINNED disc or one confirmed by the
    layout gate — several unpinned discs must not each hand out +0.70."""

    @staticmethod
    def _two_discs() -> list:
        from episodematch.providers.thediscdb import parse_disc_json

        titles = lambda disc_index: [
            {"Index": 0, "SourceFile": f"0080{disc_index}.mpls", "SegmentMap": "1-11,12", "Duration": "0:42:30", "Size": 2_028_000_000, "Item": {"Title": "Same Length", "Type": "Episode", "Season": "1", "Episode": str(disc_index * 2 + 1)}},
            {"Index": 1, "SourceFile": f"0081{disc_index}.mpls", "SegmentMap": "1-11,12", "Duration": "0:41:55", "Size": 2_011_000_000, "Item": {"Title": "Other", "Type": "Episode", "Season": "1", "Episode": str(disc_index * 2 + 2)}},
        ]
        return [
            parse_disc_json({"Name": f"Disc {i}", "Slug": f"s-{i}", "Format": "Blu-Ray", "Titles": titles(i)},
                            series_title="Show (2020)", release_title="season-1", source="test")
            for i in (0, 1)
        ]

    @staticmethod
    def _fp(mpls: str, duration: float):
        from episodematch.core.models import DiscFingerprint, MkvFile

        mkv = MkvFile(path="D:/rips/title_t00.mkv", size_bytes=2_028_000_000, duration_seconds=duration,
                      title_tag=mpls, source_id=mpls)
        return DiscFingerprint.from_mkv(mkv)

    def test_unpinned_multi_disc_unique_is_damped(self) -> None:
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._two_discs()
        fp = self._fp("00800.mpls", 2550.0)  # unique on EACH disc's own set
        matches = match_fingerprint(fp, discs)
        top = matches[0]
        reasons = " ".join(top.reasons)
        assert "unique on its own disc" in reasons
        assert "bonus damped" in reasons  # NOT the full +0.70

    def test_pinned_disc_gets_full_unique_bonus(self) -> None:
        from episodematch.providers.thediscdb import match_fingerprint

        discs = self._two_discs()
        fp = self._fp("00800.mpls", 2550.0)
        matches = match_fingerprint(fp, discs, pinned=True)
        reasons = " ".join(matches[0].reasons)
        assert "unique on its own disc" in reasons
        assert "bonus damped" not in reasons

    def test_layout_confirmed_disc_gets_full_unique_bonus(self) -> None:
        from episodematch.providers.thediscdb import disc_key, match_fingerprint, score_disc_layout

        discs = self._two_discs()
        from episodematch.core.models import DiscFingerprint, MkvFile

        def make_fp(mpls: str, duration: float) -> DiscFingerprint:
            mkv = MkvFile(path="D:/rips/x.mkv", size_bytes=2_000_000_000, duration_seconds=duration,
                          title_tag=mpls, source_id=mpls)
            return DiscFingerprint.from_mkv(mkv)

        fps = [make_fp("00800.mpls", 2550.0), make_fp("00810.mpls", 2515.0)]
        layout_scores = {disc_key(d): score_disc_layout(d, fps) for d in discs}
        confirmed = {discs[0]}  # only disc 0's playlists are present in the rip
        layout_for_fp = {disc_key(d): s for d, s in zip(discs, layout_scores.values()) if d in confirmed}
        fp = self._fp("00800.mpls", 2550.0)
        matches = match_fingerprint(fp, discs, disc_layout_scores=layout_for_fp)
        reasons = " ".join(matches[0].reasons)
        assert "unique on its own disc" in reasons
        assert "bonus damped" not in reasons
        assert "layout confirmed" in reasons


class TestTypeOverridesDuration:
    """Review 2.3: a title's runtime can never override Item.Type — an
    episode-length Extra/Menü entry is never a candidate."""

    def test_episode_length_extra_is_never_a_candidate(self) -> None:
        from episodematch.providers.thediscdb import parse_disc_json

        payload = {
            "Name": "D", "Slug": "d", "Format": "DVD",
            "Titles": [
                {"Index": 0, "SourceFile": "00001.mpls", "Duration": "0:26:49", "Size": 1_500_000_000, "Item": {"Title": "Producer's Extended Cut", "Type": "Extra"}},
                {"Index": 1, "SourceFile": "00002.mpls", "Duration": "0:21:30", "Size": 1_000_000_000, "Item": {"Title": "Real Episode", "Type": "Episode", "Season": "1", "Episode": "5"}},
            ],
        }
        discs = [parse_disc_json(payload, series_title="X", release_title="r", source="t")]
        from episodematch.core.models import DiscFingerprint, MkvFile

        mkv = MkvFile(path="D:/x/title_t00.mkv", size_bytes=1_500_000_000, duration_seconds=1609.0)  # the Extra's runtime
        fp = DiscFingerprint.from_mkv(mkv)
        from episodematch.providers.thediscdb import match_fingerprint

        matches = match_fingerprint(fp, discs)
        # The Extra matches on its runtime — but it is NEVER an episode
        # candidate: the pipeline treats matches[0].episode is None as "no
        # match" (a title's runtime can never override Item.Type).
        assert matches, "the Extra itself still matches by duration"
        assert matches[0].episode is None
        assert all(m.episode != 5 for m in matches if m.episode is not None)
