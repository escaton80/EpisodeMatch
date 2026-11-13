"""Tests for the solver-independent matcher pieces: text scoring and credit scoring."""

from __future__ import annotations

import pytest
from rapidfuzz import fuzz

from episodematch.core.models import (
    EpisodeInfo,
    GuestStar,
    SubtitleLine,
    TitleCardHit,
    TitleCardKind,
    Transcript,
)
from episodematch.matcher.credit_scorer import credit_evidence, match_name, score_episode_credits
from episodematch.matcher.text_scorer import (
    align_dialogs,
    best_line_match,
    meaningful_lines,
    normalize_dialog_text,
    normalize_episode_text,
    score_episode,
    tfidf_cosine,
)


def transcript_with(lines: list[str]) -> Transcript:
    return Transcript(
        source="srt",
        lines=[SubtitleLine(start_seconds=float(i) * 2.0, end_seconds=float(i) * 2.0 + 1.5, text=text) for i, text in enumerate(lines)],
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_speaker_prefix_removed(self) -> None:
        assert normalize_dialog_text("WALTER: I am the one who knocks") == "i am the one who knocks"

    def test_bracketed_hi_removed(self) -> None:
        assert normalize_dialog_text("[door slams] hello there") == "hello there"

    def test_punctuation_removed(self) -> None:
        assert normalize_dialog_text("Wait... what?!") == "wait what"

    def test_episode_text_drops_stopwords(self) -> None:
        normalized = normalize_episode_text("The quick brown fox is on the hill")
        tokens = normalized.split()
        assert "the" not in tokens and "is" not in tokens and "on" not in tokens
        assert "quick" in tokens and "fox" in tokens

    def test_meaningful_lines_filter_short(self) -> None:
        lines = [SubtitleLine(start_seconds=0, end_seconds=1, text="Hi!"), SubtitleLine(start_seconds=2, end_seconds=3, text="This line is long enough")]
        assert meaningful_lines(lines) == ["this line is long enough"]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


class TestTfidf:
    def test_identical_documents_score_one(self) -> None:
        doc = "heisenberg operates a meth laboratory in albuquerque new mexico"
        scores = tfidf_cosine(doc, [doc])
        assert scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_documents_score_low(self) -> None:
        scores = tfidf_cosine(
            "the detective investigates a murder in the harbor district",
            ["quarterback throws a touchdown pass in the final seconds"],
        )
        assert scores[0] < 0.2

    def test_multiple_documents_ranked(self) -> None:
        query = "spaceship crew discovers alien artifact on mars"
        docs = [
            "spaceship crew explores the mars surface and finds a strange artifact",
            "the chef prepares a delicious pasta dinner for the family",
        ]
        scores = tfidf_cosine(query, docs)
        assert scores[0] > scores[1]

    def test_empty_documents(self) -> None:
        assert tfidf_cosine("anything", []) == []


# ---------------------------------------------------------------------------
# Dialog alignment
# ---------------------------------------------------------------------------


class TestDialogAlignment:
    def test_identical_dialog_scores_high(self) -> None:
        reference = ["we need to talk about the money you owe me", "i will get your money by friday"]
        extracted = ["We need to talk about the money you owe me!", "I will get your money by Friday."]
        alignment = align_dialogs(meaningful_lines(transcript_with(extracted).lines), reference)
        assert alignment.score > 0.85
        assert alignment.matched_pairs

    def test_unrelated_dialog_scores_low(self) -> None:
        reference = ["the atomic structure of helium includes two protons"]
        extracted = ["Welcome to the five star restaurant, your table is ready"]
        alignment = align_dialogs(meaningful_lines(transcript_with(extracted).lines), reference)
        assert alignment.score < 0.4

    def test_empty_inputs(self) -> None:
        assert align_dialogs([], ["reference"]).score == 0.0
        assert align_dialogs(["query"], []).score == 0.0
        assert align_dialogs(meaningful_lines(transcript_with(["a long enough line here"]).lines), []).score == 0.0

    def test_best_line_match_cutoff(self) -> None:
        reference = ["completely different subject matter entirely"]
        assert best_line_match("another unrelated sentence over here", reference, score_cutoff=90.0) is None
        match = best_line_match("another unrelated sentence over here", reference, score_cutoff=0.0)
        assert match is not None and match.score > 0


# ---------------------------------------------------------------------------
# Episode scoring
# ---------------------------------------------------------------------------


class TestScoreEpisode:
    def _episode(self, number: int, plot: str, title: str = "Episode") -> EpisodeInfo:
        return EpisodeInfo(season=1, episode=number, title=title, plot=plot)

    def test_matching_reference_scores_above_random(self) -> None:
        plot = "Walter and Jesse deal with the aftermath of the explosion in the desert"
        episode = self._episode(3, plot)
        reference = transcript_with(
            ["you and i are not done yet", "where is the money jesse", "we need to cook tonight"]
        )
        matching_transcript = transcript_with(
            ["You and I are not done yet", "Where is the money, Jesse?", "We need to cook tonight"]
        )
        other_transcript = transcript_with(
            ["The submarine surfaces at dawn near the harbor", "Admiral, radar shows contacts"]
        )
        matching = score_episode(matching_transcript, episode, reference_transcript=reference)
        other = score_episode(other_transcript, episode, reference_transcript=reference)
        assert matching.final_score > other.final_score
        assert matching.final_score > 0.5
        assert matching.matched_pairs

    def test_no_reference_uses_plot_only(self) -> None:
        episode = self._episode(1, "The crew lands on a snowy planet and meets the locals")
        transcript = transcript_with(["The crew lands on a snowy planet and meets the locals tonight"])
        score = score_episode(transcript, episode, reference_transcript=None)
        assert score.dialog_score == 0.0
        assert score.plot_score > 0.0
        assert score.final_score > 0.0

    def test_scores_bounded(self) -> None:
        episode = self._episode(1, "plot")
        score = score_episode(transcript_with(["x" * 40]), episode, reference_transcript=transcript_with(["y" * 40]))
        assert 0.0 <= score.final_score <= 1.0

    def test_conclusive_coverage_sets_confidence_floor(self) -> None:
        """100% reference coverage is near-proof: the final score must not be
        diluted below 0.93 by a weak plot overlap."""
        episode = self._episode(4, "A thin plot description that barely overlaps anything")
        dialog_lines = ["This exact dialog line appears in the episode", "Another unique line from the same scene"]
        extracted = transcript_with([line.upper() for line in dialog_lines])  # OCR-style case noise
        reference = transcript_with(dialog_lines)
        score = score_episode(extracted, episode, reference_transcript=reference)
        assert score.coverage == pytest.approx(1.0)
        assert score.dialog_score == pytest.approx(1.0)
        assert score.final_score >= 0.93

    def test_partial_coverage_gets_no_floor(self) -> None:
        episode = self._episode(1, "An unrelated plot for scoring")
        dialog_lines = ["some generic phrase number one here", "another generic phrase number two"]
        extracted = transcript_with([line.upper() for line in dialog_lines])
        reference = transcript_with(
            dialog_lines + [f"unique reference line number {i} with plenty of words" for i in range(30)]
        )
        score = score_episode(extracted, episode, reference_transcript=reference)
        assert score.coverage < 0.85
        assert score.final_score < 0.93


# ---------------------------------------------------------------------------
# Credit scoring
# ---------------------------------------------------------------------------


class TestCreditScorer:
    def _episode(self, **kwargs) -> EpisodeInfo:
        defaults = dict(
            season=1,
            episode=4,
            title="The Long Night",
            directors=["Michelle MacLaren"],
            writers=["Vince Gilligan"],
            guest_stars=[GuestStar(name="Gina Mangano", character="Deputy")],
        )
        defaults.update(kwargs)
        return EpisodeInfo(**defaults)

    def test_director_match(self) -> None:
        hits = [
            TitleCardHit(
                timestamp_seconds=100.0,
                kind=TitleCardKind.CREDITS,
                raw_lines=["Directed by Michelle McLaren"],
                directors=["Michelle McLaren"],
            )
        ]
        score = score_episode_credits(hits, self._episode())
        assert score.score > 0.3
        assert any(match.role == "director" and match.credited_name == "Michelle MacLaren" for match in score.matches)
        assert score.summaries

    def test_guest_star_match(self) -> None:
        # "Manganno" simulates a realistic OCR double-letter error.
        hits = [TitleCardHit(timestamp_seconds=800.0, raw_lines=["Guest starring Gina Manganno"], guest_stars=["Gina Manganno"])]
        score = score_episode_credits(hits, self._episode())
        assert any(match.role == "guest" for match in score.matches)

    def test_title_card_match(self) -> None:
        hits = [TitleCardHit(timestamp_seconds=50.0, raw_lines=["THE LONG NIGHT"], title_candidates=["THE LONG NIGHT"])]
        score = score_episode_credits(hits, self._episode())
        assert score.best_title_match == "THE LONG NIGHT"
        assert score.title_ratio >= 80.0

    def test_no_match_scores_zero(self) -> None:
        hits = [TitleCardHit(timestamp_seconds=10.0, raw_lines=["SOMETHING ELSE"], title_candidates=["SOMETHING ELSE"])]
        score = score_episode_credits(hits, self._episode())
        assert score.score == 0.0

    def test_wrong_episode_title_scores_low(self) -> None:
        hit = TitleCardHit(timestamp_seconds=50.0, title_candidates=["A Completely Different Movie"])
        episode = self._episode()
        correct = score_episode_credits([TitleCardHit(timestamp_seconds=50.0, title_candidates=["The Long Night"])], episode)
        assert correct.score > score_episode_credits([hit], episode).score

    def test_match_name_threshold(self) -> None:
        found = match_name("Michelle McLaren", ["Michelle MacLaren"], threshold=87.0)
        assert found is not None and found[1] >= 87.0
        assert match_name("Totally Unrelated Name Here", ["Michelle MacLaren"], threshold=87.0) is None

    def test_combined_credits(self) -> None:
        hits = [
            TitleCardHit(
                timestamp_seconds=40.0,
                kind=TitleCardKind.CREDITS,
                directors=["Michelle MacLaren"],
                writers=["Vince Gilligan"],
                guest_stars=["Gina Mangano"],
            )
        ]
        score = score_episode_credits(hits, self._episode())
        assert score.score > 0.7
        roles = {match.role for match in score.matches}
        assert roles == {"director", "writer", "guest"}

    def test_fuzz_sanity(self) -> None:
        # Sanity guard: the fuzzy backend behaves as expected for near-identical names.
        assert fuzz.token_sort_ratio("Vince Gilligan", "Gilligan Vince") == 100


class TestCreditEvidenceDisplayOnly:
    def test_credit_evidence_has_no_episode_hint(self) -> None:
        """Audit A4(a): per-hit evidence is display-only - the calibrated score
        comes from the summary evidence, so the aggregation must not
        double-count the observation."""
        hits = [TitleCardHit(timestamp_seconds=40.0, kind=TitleCardKind.CREDITS,
                             raw_lines=["Directed by Michelle MacLaren"],
                             directors=["Michelle MacLaren"])]
        episode = EpisodeInfo(season=1, episode=4, title="X",
                              directors=["Michelle MacLaren"])
        score = score_episode_credits(hits, episode)
        items = credit_evidence(hits, score)
        assert items
        assert all(item.episode_hint is None for item in items)
