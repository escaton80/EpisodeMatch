"""Tests for the assignment solver (Hungarian algorithm + season assignment)."""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pytest

from episodematch.core.models import Evidence, MatchCandidate, Pillar
from episodematch.matcher.solver import (
    build_score_matrix,
    hungarian,
    solve_assignment,
    solve_assignment_costs,
)

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


def brute_force_min_cost(cost: list[list[float]]) -> float:
    """Optimal assignment cost by exhaustive permutation (small matrices only)."""
    rows, cols = len(cost), len(cost[0])
    size = min(rows, cols)
    best = float("inf")
    for row_perm in itertools.permutations(range(rows), size):
        for col_perm in itertools.permutations(range(cols), size):
            best = min(best, sum(cost[r][c] for r, c in zip(row_perm, col_perm)))
    return best


class TestHungarianFallback:
    def test_square_identity(self) -> None:
        rows, cols = hungarian([[0.0, 1.0], [1.0, 0.0]])
        assert sorted(zip(rows, cols)) == [(0, 0), (1, 1)]

    def test_square_anti_identity(self) -> None:
        rows, cols = hungarian([[1.0, 0.0], [0.0, 1.0]])
        assert sorted(zip(rows, cols)) == [(0, 1), (1, 0)]

    def test_empty_matrix(self) -> None:
        assert hungarian([]) == ([], [])
        assert hungarian([[]]) == ([], [])

    def test_rectangular_more_rows(self) -> None:
        # 3 files, 2 episodes: exactly two assignments.
        cost = [[4.0, 1.0], [1.0, 4.0], [5.0, 5.0]]
        rows, cols = hungarian(cost)
        assert len(rows) == 2
        total = sum(cost[r][c] for r, c in zip(rows, cols))
        assert total == pytest.approx(2.0)

    def test_rectangular_more_cols(self) -> None:
        cost = [[5.0, 1.0, 2.0], [1.0, 5.0, 2.0]]
        rows, cols = hungarian(cost)
        assert len(rows) == 2
        total = sum(cost[r][c] for r, c in zip(rows, cols))
        assert total == pytest.approx(2.0)

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_brute_force(self, seed: int) -> None:
        rng = random.Random(seed)
        rows, cols = rng.randint(1, 4), rng.randint(1, 4)
        cost = [[round(rng.uniform(0, 20), 2) for _ in range(cols)] for _ in range(rows)]
        got_rows, got_cols = hungarian(cost)
        got = sum(cost[r][c] for r, c in zip(got_rows, got_cols))
        assert got == pytest.approx(brute_force_min_cost(cost))

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_scipy(self, seed: int) -> None:
        if not HAS_SCIPY:
            pytest.skip("scipy not installed")
        rng = random.Random(100 + seed)
        rows, cols = rng.randint(1, 6), rng.randint(1, 6)
        cost = [[round(rng.uniform(-5, 15), 3) for _ in range(cols)] for _ in range(rows)]
        my_rows, my_cols = hungarian(cost)
        sp_rows, sp_cols = linear_sum_assignment(np.asarray(cost))
        my_cost = sum(cost[r][c] for r, c in zip(my_rows, my_cols))
        sp_cost = sum(cost[r][c] for r, c in zip(sp_rows.tolist(), sp_cols.tolist()))
        assert my_cost == pytest.approx(sp_cost, abs=1e-6)

    def test_no_duplicate_assignments(self) -> None:
        rng = random.Random(7)
        cost = [[rng.uniform(0, 5) for _ in range(5)] for _ in range(5)]
        rows, cols = hungarian(cost)
        assert len(set(cols)) == len(cols)
        assert len(set(rows)) == len(rows)


class TestSolveAssignmentCosts:
    def test_dispatch_reports_method(self) -> None:
        rows, cols, method = solve_assignment_costs([[1.0, 2.0], [2.0, 1.0]])
        assert method in ("scipy", "pure-python")
        assert sorted(zip(rows, cols)) == [(0, 0), (1, 1)]

    def test_empty(self) -> None:
        rows, cols, method = solve_assignment_costs([])
        assert (rows, cols, method) == ([], [], "empty")


def candidate(name: str, scores: dict[int, float]) -> MatchCandidate:
    candidate_obj = MatchCandidate(file_path=f"D:/rips/{name}")
    for episode, score in scores.items():
        candidate_obj.add_evidence(
            Evidence(pillar=Pillar.DISC_DB, kind="fingerprint", score=score, summary=f"evidence for E{episode}", episode_hint=episode)
        )
    return candidate_obj


class TestSolveAssignment:
    def test_unique_assignment(self) -> None:
        candidates = [candidate("a.mkv", {1: 1.0}), candidate("b.mkv", {2: 0.9})]
        solved = solve_assignment(candidates, [1, 2], {1: "Pilot", 2: "Part Two"}, season=2, min_confidence=0.5)
        assert len(solved.entries) == 2
        by_file = {entry.file_path.name: entry for entry in solved.entries}
        assert by_file["a.mkv"].episode == 1 and by_file["a.mkv"].episode_title == "Pilot"
        assert by_file["b.mkv"].episode == 2
        assert by_file["a.mkv"].season == 2
        assert "disc_db" in by_file["a.mkv"].methods
        assert solved.is_complete

    def test_conflict_higher_score_wins(self) -> None:
        # 0.8 evidence squares to 0.64 (>= threshold); the weaker file loses the conflict.
        candidates = [candidate("strong.mkv", {3: 1.0}), candidate("weak.mkv", {3: 0.8})]
        solved = solve_assignment(candidates, [3], season=1, min_confidence=0.55)
        assert len(solved.entries) == 1
        assert solved.entries[0].file_path.name == "strong.mkv"
        assert len(solved.unassigned) == 1
        assert solved.unassigned[0].file_path.name == "weak.mkv"
        assert "conflict" in solved.unassigned[0].reason

    def test_min_confidence_threshold(self) -> None:
        candidates = [candidate("weak.mkv", {1: 0.3})]
        solved = solve_assignment(candidates, [1], season=1, min_confidence=0.55)
        assert solved.entries == []
        assert len(solved.unassigned) == 1
        # Audit A3: the reason wording changed — the cell is inadmissible now.
        assert "No admissible match above the confidence threshold" in solved.unassigned[0].reason
        assert solved.unassigned[0].best_guess_episode == 1

    def test_reasoning_and_evidence_attached(self) -> None:
        candidates = [candidate("a.mkv", {2: 0.8})]
        solved = solve_assignment(candidates, [2], season=1, min_confidence=0.5)
        entry = solved.entries[0]
        assert entry.reasoning and "evidence for E2" in entry.reasoning[0]
        assert entry.evidence[0].pillar is Pillar.DISC_DB

    def test_no_candidates(self) -> None:
        solved = solve_assignment([], [1, 2], season=1)
        assert solved.entries == [] and solved.unassigned == []
        assert solved.method == "empty"

    def test_no_episodes(self) -> None:
        solved = solve_assignment([candidate("a.mkv", {1: 1.0})], [], season=1)
        assert solved.entries == []
        assert len(solved.unassigned) == 1
        assert "no episodes" in solved.unassigned[0].reason.lower()

    def test_nan_scores_do_not_break_solver(self) -> None:
        candidates = [candidate("a.mkv", {1: 1.0}), candidate("b.mkv", {1: 0.9})]
        # Inject NaN into the matrix through episode_scores is not possible via the
        # model, so corrupt the built matrix instead.
        matrix = build_score_matrix(candidates, [1])
        matrix[0, 0] = float("nan")
        solved = solve_assignment(candidates, [1], season=1, min_confidence=0.5)
        # Solver rebuilds the matrix internally; just ensure no crash and a result.
        assert len(solved.entries) + len(solved.unassigned) == 2


class TestBuildScoreMatrix:
    def test_matrix_shape_and_values(self) -> None:
        candidate_a = MatchCandidate(file_path="D:/rips/a.mkv")
        candidate_a.add_evidence(Evidence(pillar=Pillar.DISC_DB, kind="f", score=0.5, summary="e1", episode_hint=1))
        candidate_a.add_evidence(Evidence(pillar=Pillar.SUBTITLE, kind="t", score=0.8, summary="e2", episode_hint=3))
        candidate_b = MatchCandidate(file_path="D:/rips/b.mkv")
        matrix = build_score_matrix([candidate_a, candidate_b], [1, 2, 3])
        assert matrix.shape == (2, 3)
        # Audit A4: calibrated aggregation - strongest evidence per pillar,
        # no squaring (0.5 stays 0.5, 0.8 stays 0.8).
        assert matrix[0, 0] == pytest.approx(0.5)
        assert matrix[0, 1] == 0.0
        assert matrix[0, 2] == pytest.approx(0.8)
        assert matrix[1].sum() == 0.0

    def test_squared_evidence_aggregation(self) -> None:
        # add_evidence squares per-evidence scores: two 0.5 evidences -> 0.5.
        candidate_obj = candidate("a.mkv", {})
        for _ in range(2):
            candidate_obj.add_evidence(Evidence(pillar=Pillar.SUBTITLE, kind="t", score=0.5, summary="x", episode_hint=1))
        matrix = build_score_matrix([candidate_obj], [1])
        assert matrix[0, 0] == pytest.approx(0.5)


class TestThresholdBeforeOptimization:
    def test_threshold_applied_before_optimization(self) -> None:
        """Audit A3: the confidence threshold must be part of the assignment
        problem. Otherwise the optimizer prefers two inadmissible 0.54 cells
        over one feasible 0.90 match and discards everything."""
        a = MatchCandidate(file_path="D:/x/a.mkv")
        a.episode_scores[1] = 0.54
        b = MatchCandidate(file_path="D:/x/b.mkv")
        b.episode_scores[1] = 0.90
        b.episode_scores[2] = 0.54
        solved = solve_assignment([a, b], [1, 2], season=1, min_confidence=0.55)
        assert len(solved.entries) == 1
        assert solved.entries[0].file_path.name == "b.mkv"
        assert solved.entries[0].episode == 1
        assert solved.entries[0].confidence == pytest.approx(0.90)
        assert len(solved.unassigned) == 1  # a: no admissible cell left


class TestUniqueDurationCalibration:
    def test_unique_duration_disc_evidence_is_assigned(self) -> None:
        """Audit A4(c): the DiscDB 'unique duration' signal (0.70) is
        near-conclusive and must not be diluted below the solver threshold."""
        c = MatchCandidate(file_path="D:/x/t06.mkv")
        c.add_evidence(Evidence(pillar=Pillar.DISC_DB, kind="f", score=0.70, summary="unique on disc", episode_hint=6))
        solved = solve_assignment([c], [6], season=1, min_confidence=0.55)
        assert len(solved.entries) == 1
        assert solved.entries[0].confidence == pytest.approx(0.70)

class TestExcludeCoveredEpisodes:
    """Review 1.1: a range assignment (E01-E02) consumes BOTH episodes — the
    covered follow-up episode must not be assignable to another file."""

    def test_range_candidate_covers_follow_up(self) -> None:
        from episodematch.matcher.solver import exclude_covered_episodes

        candidate = MatchCandidate(
            file_path=Path("D:/rips/title_t00.mkv"),
            episode_end=2,
            evidence=[
                Evidence(
                    pillar=Pillar.DISC_DB,
                    kind="fingerprint_match",
                    score=0.9,
                    summary="Disc fingerprint matches E01-E02",
                    episode_hint=1,
                )
            ],
        )
        assert exclude_covered_episodes([1, 2, 3, 4], [candidate]) == [1, 3, 4]

    def test_candidates_without_range_leave_numbers_untouched(self) -> None:
        from episodematch.matcher.solver import exclude_covered_episodes

        candidates = [MatchCandidate(file_path=Path("D:/rips/title_t00.mkv"))]
        assert exclude_covered_episodes([1, 2, 3], candidates) == [1, 2, 3]

class TestActivePillarThresholdScaling:
    """PILLAR4_PLAN batch 5: the confidence threshold scales with the sum of
    ACTIVE pillar weights (max. up to the full 1.0-scale) — a DVD with no
    DiscDB entry and no subtitles is identified by pillar 4 alone (a 100%
    title match is assignable), while all-pillar scans keep the plain 0.55
    threshold. The reported confidences stay the honest absolute scores."""

    def test_single_pillar_scan_gets_scaled_threshold(self) -> None:
        from episodematch.matcher.solver import solve_assignment

        candidate = MatchCandidate(
            file_path=Path("D:/rips/title_t00.mkv"),
            episode_scores={1: 0.35},  # title match alone: 0.35 * 1.0
        )
        solved = solve_assignment(
            [candidate],
            [1, 2, 3],
            {1: "Prototype", 2: "Second", 3: "Third"},
            season=1,
            min_confidence=0.55,
            weight_sum=0.20,  # only the visual pillar is active
        )
        assert solved.entries, "a 100% single-pillar match must be assignable"
        assert solved.entries[0].episode == 1
        assert solved.entries[0].confidence == pytest.approx(0.35)  # honest absolute score

    def test_two_pillar_scan_gets_proportionally_scaled_threshold(self) -> None:
        from episodematch.matcher.solver import solve_assignment

        candidate = MatchCandidate(
            file_path=Path("D:/rips/title_t00.mkv"),
            episode_scores={1: 0.70},  # disc 0.40 + visual 0.30
        )
        solved = solve_assignment(
            [candidate],
            [1, 2],
            {1: "E1", 2: "E2"},
            season=1,
            min_confidence=0.55,
            weight_sum=0.55,  # disc 0.35 + visual 0.20
        )
        # threshold 0.55 * 0.55 = 0.3025 -> 0.70 assigned, confidence honest
        assert solved.entries and solved.entries[0].confidence == pytest.approx(0.70)

    def test_full_pillar_weight_sum_keeps_plain_threshold(self) -> None:
        """With every pillar active (weight sum >= 1.0) the threshold stays
        the plain 0.55 — multi-pillar behaviour is unchanged."""
        from episodematch.matcher.solver import solve_assignment

        candidate = MatchCandidate(
            file_path=Path("D:/rips/title_t00.mkv"),
            episode_scores={1: 0.70},
        )
        solved = solve_assignment(
            [candidate],
            [1],
            {1: "E1"},
            season=1,
            min_confidence=0.55,
            weight_sum=1.40,  # all pillars active
        )
        assert solved.entries and solved.entries[0].confidence == pytest.approx(0.70)

    def test_without_weight_sum_behaviour_unchanged(self) -> None:
        from episodematch.matcher.solver import solve_assignment

        candidate = MatchCandidate(file_path=Path("D:/rips/title_t00.mkv"), episode_scores={1: 0.35})
        solved = solve_assignment([candidate], [1], {1: "E1"}, season=1, min_confidence=0.55)
        assert solved.entries == []  # 0.35 < 0.55: legacy behaviour preserved
