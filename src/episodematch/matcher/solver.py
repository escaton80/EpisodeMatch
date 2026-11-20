"""Assignment solver: unique file->episode mapping (Hungarian algorithm).

The core is :func:`solve_assignment`:

* builds a score matrix (files x episodes),
* maximizes the total score with ``scipy.optimize.linear_sum_assignment``
  (Jonker-Volgenant, i.e. the modern Hungarian method), falling back to a fully
  self-contained O(n^3) Hungarian implementation when SciPy is unavailable,
* drops assignments below ``min_confidence`` and reports them as unassigned with
  a reason, so no episode is ever assigned twice and every decision is explainable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from episodematch.core.exceptions import SolverError
from episodematch.core.models import AssignmentEntry, MatchCandidate, UnassignedFile

logger = logging.getLogger(__name__)

try:  # SciPy is a hard dependency, but the fallback keeps tests/environments resilient.
    from scipy.optimize import linear_sum_assignment as _scipy_lsa

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - depends on environment
    _scipy_lsa = None
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Pure-Python Hungarian algorithm (Kuhn-Munkres, O(n^3), potentials method)
# ---------------------------------------------------------------------------


def hungarian(cost_matrix: list[list[float]]) -> tuple[list[int], list[int]]:
    """Solve the *minimum-cost* assignment problem without SciPy.

    Returns ``(row_ind, col_ind)`` like ``scipy.optimize.linear_sum_assignment``.
    The implementation uses the classic potentials-based variant (O(n^3)),
    operating on a padded square matrix internally.
    """
    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [], []
    size = max(rows, cols)
    INF = float("inf")
    # Pad to square with large costs so pad cells are never chosen when avoidable.
    costs = [[0.0] * size for _ in range(size)]
    pad_value = 1e9
    for r in range(size):
        for c in range(size):
            if r < rows and c < cols:
                costs[r][c] = float(cost_matrix[r][c])
            else:
                costs[r][c] = pad_value

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)  # p[j] = row matched to column j (1-based), 0 = none
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j] = current
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind: list[int] = []
    col_ind: list[int] = []
    for j in range(1, size + 1):
        row = p[j] - 1
        col = j - 1
        if row < rows and col < cols and costs[row][col] < pad_value:
            row_ind.append(row)
            col_ind.append(col)
    # SciPy returns row_ind sorted ascending.
    pairs = sorted(zip(row_ind, col_ind), key=lambda pair: pair[0])
    return [r for r, _ in pairs], [c for _, c in pairs]


def solve_assignment_costs(
    cost_matrix: list[list[float]],
) -> tuple[list[int], list[int], str]:
    """Minimize the assignment cost; returns (rows, cols, method)."""
    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [], [], "empty"
    if _HAS_SCIPY:
        import numpy as _np  # noqa: PLC0415

        row_ind, col_ind = _scipy_lsa(_np.asarray(cost_matrix, dtype=float))
        return [int(r) for r in row_ind], [int(c) for c in col_ind], "scipy"
    row_ind, col_ind = hungarian(cost_matrix)
    return row_ind, col_ind, "pure-python"


# ---------------------------------------------------------------------------
# Season-level solver over MatchCandidates
# ---------------------------------------------------------------------------


@dataclass
class SolvedAssignment:
    """Result of the global assignment for one folder/disc."""

    entries: list[AssignmentEntry] = field(default_factory=list)
    unassigned: list[UnassignedFile] = field(default_factory=list)
    total_score: float = 0.0
    method: str = ""

    @property
    def is_complete(self) -> bool:
        return not self.unassigned


def exclude_covered_episodes(episode_numbers: list[int], candidates: list[MatchCandidate]) -> list[int]:
    """Remove episodes covered by multi-episode (range) candidates.

    Review 1.1: a candidate assigned to a disc entry published as "1-2" contains
    BOTH episodes — the follow-up episode (E02) must not be assignable to
    another file. The start episode stays in the list (the candidate scores on
    it); only strictly-covered follow-ups are excluded.
    """
    covered: set[int] = set()
    for candidate in candidates:
        end = candidate.episode_end
        hint = max(
            (e.episode_hint for e in candidate.evidence if e.episode_hint is not None),
            default=None,
        )
        if end is None or hint is None:
            continue
        for episode in range(hint + 1, end + 1):
            covered.add(episode)
    return [episode for episode in episode_numbers if episode not in covered]


def build_score_matrix(candidates: list[MatchCandidate], episode_numbers: list[int]) -> np.ndarray:
    """Create the (files x episodes) score matrix from accumulated evidence."""
    matrix = np.zeros((len(candidates), len(episode_numbers)), dtype=float)
    for row, candidate in enumerate(candidates):
        for col, episode in enumerate(episode_numbers):
            matrix[row, col] = float(candidate.episode_scores.get(episode, 0.0))
    return matrix


def solve_assignment(
    candidates: list[MatchCandidate],
    episode_numbers: list[int],
    episode_titles: dict[int, str] | None = None,
    *,
    season: int = 1,
    min_confidence: float = 0.55,
    max_confidence: float = 1.0,
) -> SolvedAssignment:
    """Assign every file at most one episode, and every episode at most one file.

    Scores above 1.0 (multiple corroborating pillars) are clamped to
    ``max_confidence`` for reporting but the raw matrix is used for optimization.
    """
    episode_titles = episode_titles or {}
    if not candidates:
        return SolvedAssignment(method="empty")
    if not episode_numbers:
        return SolvedAssignment(
            unassigned=[
                UnassignedFile(file_path=candidate.file_path, reason="Season guide has no episodes")
                for candidate in candidates
            ],
            method="no-episodes",
        )

    raw_matrix = build_score_matrix(candidates, episode_numbers)
    finite = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # Audit A3: the confidence threshold is part of the assignment problem.
    # Sub-threshold cells are inadmissible — they must not shape the global
    # assignment (otherwise the optimizer prefers two 0.54 cells over one
    # feasible 0.90 match and discards everything). The raw matrix is kept for
    # best-guess reporting.
    assignable = np.where(finite >= min_confidence, finite, 0.0)
    max_score = float(assignable.max()) or 1.0
    cost_matrix = (max_score - assignable).tolist()

    try:
        row_ind, col_ind, method = solve_assignment_costs(cost_matrix)
    except Exception as exc:  # noqa: BLE001 - solver failures must not crash the scan
        raise SolverError(f"Assignment failed: {exc}") from exc

    solved = SolvedAssignment(method=method)
    assigned_rows: set[int] = set()
    for row, col in zip(row_ind, col_ind):
        assigned_rows.add(row)
        score = float(assignable[row, col])
        candidate = candidates[row]
        episode = episode_numbers[col]
        if score < min_confidence:
            # The optimizer assigned an inadmissible (zeroed) cell — nothing
            # usable for this file.
            solved.unassigned.append(
                UnassignedFile(
                    file_path=candidate.file_path,
                    reason="No admissible match above the confidence threshold",
                    best_guess_episode=episode,
                    best_guess_confidence=score,
                )
            )
            continue
        evidence_sorted = sorted(candidate.evidence, key=lambda item: item.score, reverse=True)
        solved.entries.append(
                AssignmentEntry(
                    file_path=candidate.file_path,
                    season=season,
                    episode=episode,
                    episode_end=candidate.episode_end,
                    episode_title=episode_titles.get(episode),
                confidence=max(0.0, min(max_confidence, score)),
                methods=sorted({evidence.pillar.value for evidence in candidate.evidence}),
                reasoning=[evidence.summary for evidence in evidence_sorted[:8] if evidence.summary],
                evidence=list(candidate.evidence),
            )
        )
        solved.total_score += score

    # Files whose best cell was not chosen by the solver (lost conflicts). The
    # best guess is reported from the RAW matrix, so a sub-threshold best match
    # is still visible to the user.
    for row, candidate in enumerate(candidates):
        if row in assigned_rows:
            continue
        best_col = int(np.argmax(finite[row])) if finite.shape[1] else -1
        best_score = float(finite[row, best_col]) if best_col >= 0 else 0.0
        best_episode = episode_numbers[best_col] if best_col >= 0 else None
        below = best_score < min_confidence
        solved.unassigned.append(
            UnassignedFile(
                file_path=candidate.file_path,
                reason=(
                    f"Lost the global assignment conflict "
                    f"(best episode S{season:02d}E{best_episode:02d} at {best_score:.2f})"
                    if best_episode is not None and not below
                    else (
                        f"Best guess S{season:02d}E{best_episode:02d} at {best_score:.2f} "
                        f"is below the confidence threshold"
                        if best_episode is not None
                        else "No episodes available"
                    )
                ),
                best_guess_episode=best_episode,
                best_guess_confidence=best_score,
            )
        )

    solved.entries.sort(key=lambda entry: (entry.episode, entry.file_path.name))
    return solved
