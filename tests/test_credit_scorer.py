"""Tests for the pillar-4 credit scorer calibration (PILLAR4_PLAN batch 3):

- short-title protection (partial_ratio off below 5 chars, >= 95 full match),
- normalized season-IDF for repeated credits (log(N/k)/log(N) in [0, 1]),
- transparent credits_summary evidence (timestamp + recognized OCR text).
"""

from __future__ import annotations

import math

import httpx
import pytest

from episodematch.core.models import (
    ChapterInfo,
    EpisodeInfo,
    GuestStar,
    SeasonGuide,
    SeriesInfo,
    TitleCardHit,
    TitleCardKind,
)
from episodematch.matcher.credit_scorer import (
    build_credit_idf,
    score_episode_credits,
)


def _hit(**kwargs) -> TitleCardHit:
    base = dict(
        timestamp_seconds=61.0,  # 01:01
        kind=TitleCardKind.CREDITS,
        raw_lines=[],
        ocr_confidence=0.9,
    )
    base.update(kwargs)
    return TitleCardHit(**base)


def _episode(number: int, *, title: str = "Episode", directors=None, writers=None, guests=None) -> EpisodeInfo:
    return EpisodeInfo(
        season=1,
        episode=number,
        title=title,
        plot=f"plot {number}",
        directors=list(directors or []),
        writers=list(writers or []),
        guest_stars=[GuestStar(name=name) for name in (guests or [])],
    )


def _guide(directors_per_episode: list[list[str]]) -> SeasonGuide:
    series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id="tt1", first_air_year=2020)
    episodes = [
        _episode(index + 1, title=f"Episode {index + 1}", directors=directors)
        for index, directors in enumerate(directors_per_episode)
    ]
    return SeasonGuide(series=series, season=1, episodes=episodes)


class TestShortTitleProtection:
    def test_short_generic_title_never_matches_partial(self) -> None:
        """'Home' (4 chars) inside 'Welcome Home' would previously score 100%
        via partial_ratio — short titles now need a full-line match >= 95."""
        episode = _episode(1, title="Welcome Home")
        hit = _hit(title_candidates=["Home"])
        score = score_episode_credits([hit], episode)
        assert score.title_ratio == 0.0
        assert not any(m.role == "title" for m in score.matches)

    def test_short_exact_title_is_confirmed(self) -> None:
        episode = _episode(1, title="Home")
        hit = _hit(title_candidates=["Home"])
        score = score_episode_credits([hit], episode)
        assert score.best_title_match == "Home"  # full-line match 100 >= 95
        assert score.title_ratio == pytest.approx(100.0)

    def test_long_titles_keep_partial_matching(self) -> None:
        episode = _episode(1, title="The Long Night of the Hunter")
        hit = _hit(title_candidates=["Long Night of the Hunter"])
        score = score_episode_credits([hit], episode)
        assert score.title_ratio >= 80.0  # >= 80 via the partial metric
        assert score.score >= 0.35 * 0.8


class TestSeasonCreditIdf:
    def test_idf_is_none_below_three_episodes(self) -> None:
        guide = _guide([["Jane Doe"], ["Jane Doe"]])
        assert build_credit_idf(guide) is None

    def test_weights_are_normalized_between_zero_and_one(self) -> None:
        guide = _guide([["Jane Doe", "Unique Person"], ["Jane Doe"], ["Jane Doe"], ["Jane Doe"]])
        idf = build_credit_idf(guide)
        assert idf is not None
        assert idf[("director", "jane doe")] == pytest.approx(0.0)  # k = N
        assert idf[("director", "unique person")] == pytest.approx(1.0)  # k = 1
        assert all(0.0 <= weight <= 1.0 for weight in idf.values())

    def test_repeated_director_fades_to_zero(self) -> None:
        """A director credited in ALL episodes of the season contributes no
        score — only rarity discriminates (review: same director in 6 episodes)."""
        guide = _guide([["Jane Doe"], ["Jane Doe"], ["Jane Doe"], ["Jane Doe"]])
        idf = build_credit_idf(guide)
        hit = _hit(directors=["Jane Doe"])
        score = score_episode_credits([hit], guide.episodes[0], credit_idf=idf)
        assert score.score == 0.0  # 0.4 * (1.0 * 0.0)

    def test_unique_director_keeps_full_weight(self) -> None:
        guide = _guide([["Jane Doe", "Unique Person"], ["Jane Doe"], ["Jane Doe"], ["Jane Doe"]])
        idf = build_credit_idf(guide)
        hit = _hit(directors=["Unique Person"])
        score = score_episode_credits([hit], guide.episodes[0], credit_idf=idf)
        assert score.score == pytest.approx(0.4)  # 0.4 * (1.0 * 1.0)

    def test_without_idf_behaviour_unchanged(self) -> None:
        guide = _guide([["Jane Doe"], ["Jane Doe"], ["Jane Doe"], ["Jane Doe"]])
        hit = _hit(directors=["Jane Doe"])
        score = score_episode_credits([hit], guide.episodes[0])
        assert score.score == pytest.approx(0.4)  # legacy: full weight without IDF


class TestCreditsSummaryEvidence:
    def test_summary_carries_timestamp_and_ocr_text(self, config) -> None:
        """The credits_summary evidence must include the timestamp (MM:SS) and
        the recognized OCR text of the top matches — not just a counter."""
        from pathlib import Path
        from types import SimpleNamespace

        from pathlib import Path as _Path

        from episodematch.core.models import MkvFile
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        hits = [
            _hit(
                timestamp_seconds=61.0,
                directors=["Jane Doe"],
                raw_lines=["Directed by Jane Doe"],
            )
        ]
        guide = _guide([["Jane Doe"], ["Other Person"], ["Third Person"]])  # N=3, k=1 -> weight 1.0

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, **kwargs):  # noqa: ANN001, ANN202
                return hits

        mkv = MkvFile(path=_Path("title_t00.mkv"), size_bytes=1, duration_seconds=2400.0)
        pipeline = Pipeline(config, frame_scanner=FakeVisual())
        evidence_list: list = []
        candidate = SimpleNamespace(evidence=evidence_list, add_evidence=lambda ev: evidence_list.append(ev))
        report = SimpleNamespace(pillar_reports=[])
        pipeline._stage_visual([(mkv, candidate)], guide, report, ScanOptions(depth="deep"))
        summaries = [e for e in candidate.evidence if e.kind == "credits_summary"]
        assert summaries, "a credit match must produce a credits_summary evidence"
        summary = summaries[0].summary
        assert "01:01" in summary  # timestamp (MM:SS)
        assert "Jane Doe" in summary  # recognized OCR text
        details = summaries[0].details
        assert details["top"][0]["timestamp_seconds"] == 61.0
        assert details["top"][0]["ocr_text"] == "Jane Doe"

class TestSeriesCastHint:
    """PILLAR4_PLAN batch 3.5: the user already picked the series — a series
    cast name on a card ("Tim Russ" on every Voyager episode) confirms the
    series but must NOT add episode score. It appears as hint-only evidence."""

    def test_season_guide_carries_cast_names(self) -> None:
        from episodematch.providers.tmdb_client import TmdbClient

        def handler(request: httpx.Request) -> httpx.Response:
            path = httpx.URL(request.url).path
            if path == "/3/tv/42/season/1":
                return httpx.Response(200, json={"name": "Test Show", "episodes": [], "cast": [
                    {"name": "Tim Russ", "character": "Lt. Tuvok"},
                    {"name": "Kate Mulgrew", "character": "Capt. Kathryn Janeway"},
                ]})
            if path == "/3/tv/42/external_ids":
                return httpx.Response(200, json={"imdb_id": "tt1"})
            if path == "/3/tv/42/season/1/credits":
                return httpx.Response(200, json={"cast": [{"name": "Tim Russ"}, {"name": "Kate Mulgrew"}]})
            return httpx.Response(404)

        client = TmdbClient("test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
        series = SeriesInfo(name="Test Show", tmdb_id=42, imdb_id=None, first_air_year=2020)
        guide = client.season_guide(series, 1)
        assert guide.cast == ["Tim Russ", "Kate Mulgrew"]

    def test_series_cast_hit_is_hint_evidence_without_score(self, config) -> None:
        from types import SimpleNamespace
        from pathlib import Path as _Path

        from episodematch.core.models import MkvFile
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        hits = [_hit(timestamp_seconds=220.0, guest_stars=["Tim Russ"], raw_lines=["TIM RUSS as Lieutenant Tuvols"])]
        guide = _guide([["Nobody"], ["Nobody"], ["Nobody"]])
        guide.cast = ["Tim Russ", "Kate Mulgrew"]

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, **kwargs):  # noqa: ANN001, ANN202
                return hits

        mkv = MkvFile(path=_Path("title_t00.mkv"), size_bytes=1, duration_seconds=2400.0)
        pipeline = Pipeline(config, frame_scanner=FakeVisual())
        evidence_list: list = []
        candidate = SimpleNamespace(evidence=evidence_list, add_evidence=lambda ev: evidence_list.append(ev))
        report = SimpleNamespace(pillar_reports=[])
        pipeline._stage_visual([(mkv, candidate)], guide, report, ScanOptions(depth="deep"))

        hints = [e for e in evidence_list if e.kind == "series_cast_hint"]
        assert hints, "a series cast name on a card must be reported"
        assert hints[0].score == 0.0  # no episode score — the series is already chosen
        assert "Tim Russ" in hints[0].summary
        # and no credits_summary was produced from the cast hit alone
        assert not [e for e in evidence_list if e.kind == "credits_summary" and e.score > 0]

    def test_last_name_only_ocr_matches_series_cast(self, config) -> None:
        """The Voyager cards print only the LAST name ("PICARDO") — the hint
        matching falls back to a last-name comparison against the cast list."""
        from types import SimpleNamespace
        from pathlib import Path as _Path

        from episodematch.core.models import MkvFile
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        hits = [_hit(timestamp_seconds=212.0, guest_stars=["PICARDO"], raw_lines=["PICARDO as The Doctor"])]
        guide = _guide([["Nobody"], ["Nobody"], ["Nobody"]])
        guide.cast = ["Robert Picardo", "Kate Mulgrew"]

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, **kwargs):  # noqa: ANN001, ANN202
                return hits

        mkv = MkvFile(path=_Path("title_t00.mkv"), size_bytes=1, duration_seconds=2400.0)
        pipeline = Pipeline(config, frame_scanner=FakeVisual())
        evidence_list: list = []
        candidate = SimpleNamespace(evidence=evidence_list, add_evidence=lambda ev: evidence_list.append(ev))
        report = SimpleNamespace(pillar_reports=[])
        pipeline._stage_visual([(mkv, candidate)], guide, report, ScanOptions(depth="deep"))

        hints = [e for e in evidence_list if e.kind == "series_cast_hint"]
        assert hints, "a last-name-only OCR hit must still match the cast"
        assert "Robert Picardo" in hints[0].summary  # the full cast name is reported
        assert hints[0].score == 0.0  # hint-only: no episode score

    def test_no_match_records_transparency_evidence(self, config) -> None:
        """Files whose frames produce no credit/title match must still show
        WHAT was probed and what the OCR read — not 'No evidence recorded'."""
        from types import SimpleNamespace
        from pathlib import Path as _Path

        from episodematch.core.models import MkvFile, TitleCardKind
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        hits = [
            _hit(timestamp_seconds=45.0, kind=TitleCardKind.UNKNOWN, raw_lines=["some random scene text"]),
            _hit(timestamp_seconds=90.0, kind=TitleCardKind.UNKNOWN, raw_lines=[]),
        ]
        guide = _guide([["Nobody"], ["Nobody"], ["Nobody"]])

        class FakeVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, **kwargs):  # noqa: ANN001, ANN202
                return hits

        mkv = MkvFile(path=_Path("title_t00.mkv"), size_bytes=1, duration_seconds=2400.0)
        pipeline = Pipeline(config, frame_scanner=FakeVisual())
        evidence_list: list = []
        candidate = SimpleNamespace(evidence=evidence_list, add_evidence=lambda ev: evidence_list.append(ev))
        report = SimpleNamespace(pillar_reports=[])
        pipeline._stage_visual([(mkv, candidate)], guide, report, ScanOptions(depth="deep"))

        no_match = [e for e in evidence_list if e.kind == "visual_no_match"]
        assert no_match, "zero-match files must record what was probed"
        assert "2 frames probed" in no_match[0].summary
        assert "some random scene text" in no_match[0].summary  # the OCR top text

    def test_pipeline_passes_chapter_marks_to_scanner(self, config) -> None:
        """Batch 2.8: the pipeline reads the MKV chapter marks and hands them
        to the frame scanner (the dense title-card zone is anchored there)."""
        captured: dict = {}

        class CapturingVisual:
            def scan(self, mkv_path, *, duration_seconds=None, max_frames=None, chapter_marks=None):  # noqa: ANN001, ANN202
                captured["chapter_marks"] = chapter_marks
                return []

        from pathlib import Path

        from types import SimpleNamespace

        from episodematch.core.models import MkvFile
        from episodematch.matcher.pipeline import Pipeline, ScanOptions

        mkv = MkvFile(
            path=Path("title_t00.mkv"),
            size_bytes=1,
            duration_seconds=2400.0,
            chapters=[
                ChapterInfo(index=1, start_seconds=0.0, end_seconds=180.0),
                ChapterInfo(index=2, start_seconds=180.0, end_seconds=2400.0),
            ],
        )
        guide = _guide([["Nobody"], ["Nobody"], ["Nobody"]])

        class FakeVisual(CapturingVisual):
            pass

        pipeline = Pipeline(config, frame_scanner=CapturingVisual())
        candidate = SimpleNamespace(evidence=[], add_evidence=lambda ev: None)
        report = SimpleNamespace(pillar_reports=[])
        pipeline._stage_visual([(mkv, candidate)], guide, report, ScanOptions(depth="deep"))
        assert captured["chapter_marks"] == [180.0]  # the 0.0 start is filtered (no zone at file start)

class TestTitleNormalizationDigitBoundary:
    """Voyager case: the OCR reads 'THE37's' (no space between letters and
    digits), the TMDB title is 'The 37's' — after punctuation stripping that
    yields 'the37s' vs 'the 37s', which scored below the 80% match threshold.
    Digit/letter boundaries must be split so both forms normalize identically."""

    def test_normalize_title_splits_digit_letter_boundary(self) -> None:
        from episodematch.matcher.credit_scorer import _normalize_title

        assert _normalize_title("THE37's") == _normalize_title("The 37's")
        assert _normalize_title("the37s") == _normalize_title("The 37s")

    def test_match_works_for_mixed_spacing(self) -> None:
        episode = _episode(1, title="The 37's")
        hit = _hit(title_candidates=["THE37's"])
        score = score_episode_credits([hit], episode, series=_guide(["Nobody"]).series)
        assert score.title_ratio == pytest.approx(100.0)

    def test_series_logo_still_rejected_after_boundary_split(self) -> None:
        """The split must not resurrect the series-logo false alarm:
        'ROCKO' (5 chars, no digits) vs 'Rocko's Happy Sack' stays rejected —
        the filter needs the REAL series name ("Test Show" from the shared
        helper would never trip the partial match)."""
        episode = _episode(1, title="Rocko's Happy Sack")
        series = SeriesInfo(name="Rocko's Modern Life", tmdb_id=42, imdb_id="tt1", first_air_year=1993)
        hit = _hit(title_candidates=["ROCKO"])
        score = score_episode_credits([hit], episode, series=series)
        assert score.title_ratio == 0.0


class TestRockosCalibration:
    """Calibration from the Rockos Modern Life S1 DVD scan (user report):
    1. the series logo 'ROCKO' sits on EVERY card and partial-matches
       "Rocko's Happy Sack" at 100% -> wrong episode won the assignment;
    2. the next-episode teaser card at 23:05 (end of file) matched the NEXT
       episode 'Trash-O-Madness' at 85% -> wrong episode, right next-door.
    Both are show-generic signals and must never add episode score."""

    def _hit(self, **kwargs) -> TitleCardHit:
        base = dict(timestamp_seconds=39.0, kind=TitleCardKind.TITLE_CARD, raw_lines=[], ocr_confidence=0.9)
        base.update(kwargs)
        return TitleCardHit(**base)

    def _guide(self) -> SeasonGuide:
        series = SeriesInfo(name="Rocko's Modern Life", tmdb_id=12345, imdb_id="tt1", first_air_year=1993)
        episodes = [
            _episode(19, title="The Good, The Bad and The Wallaby"),
            _episode(21, title="Rocko's Happy Sack"),
            _episode(12, title="Trash-O-Madness"),
        ]
        return SeasonGuide(series=series, season=1, episodes=episodes)

    def test_series_logo_title_is_rejected(self) -> None:
        """The series logo 'ROCKO' partial-matches 'Rocko's Happy Sack' at
        100% — a show-generic title must never score."""
        guide = self._guide()
        hit = _hit(title_candidates=["ROCKO"])
        score = score_episode_credits([hit], guide.episodes[1], series=guide.series)
        assert score.title_ratio == 0.0 and score.score == 0.0
        assert score.score == 0.0

    def test_next_episode_teaser_at_file_end_is_rejected(self) -> None:
        """A card at 23:05 of a ~24 min file is the next-episode teaser in the
        credits tail — its title must not score."""
        from episodematch.core.models import TitleCardHit, TitleCardKind

        guide = self._guide()
        teaser = _hit(timestamp_seconds=1385.0, title_candidates=["Trash-O-Madness"])
        score = score_episode_credits(
            [teaser], guide.episodes[2], series=guide.series, file_duration=1400.0
        )
        assert score.title_ratio == 0.0 and score.score == 0.0

    def test_real_title_mid_file_still_scores(self) -> None:
        """The genuine title card mid-file (Wallaby at 200 s of 1400 s) scores."""
        guide = self._guide()
        real = _hit(timestamp_seconds=200.0, title_candidates=["The Good, The Bad and The Wallaby"])
        score = score_episode_credits([real], guide.episodes[0], series=guide.series)
        assert score.title_ratio >= 80.0 and score.score >= 0.35 * 0.8
