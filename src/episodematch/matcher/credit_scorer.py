"""Credit scoring: match OCR'd names/titles against TMDB/IMDb credits (pillar 4).

Per candidate episode, the OCR'd title-card/credits hits are scored:

* directors on screen vs. the episode's TMDB directors,
* writers on screen vs. TMDB writers,
* guest stars on screen vs. TMDB guest stars,
* detected title-card text vs. the episode title.

Every contributing match becomes human-readable evidence ("Director 'Michelle
MacLaren' detected at 01:42 is credited for S01E03").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from episodematch.core.models import (
    EpisodeInfo,
    Evidence,
    Pillar,
    TitleCardHit,
)

_DEFAULT_NAME_THRESHOLD = 87.0


@dataclass
class NameMatch:
    """One fuzzy name match between OCR text and provider credits."""

    ocr_name: str
    credited_name: str
    ratio: float
    role: str  # "director" | "writer" | "guest" | "title"


@dataclass
class CreditScore:
    """Pillar-4 score of one file against one candidate episode."""

    episode_number: int
    score: float
    matches: list[NameMatch] = field(default_factory=list)
    best_title_match: str | None = None
    title_ratio: float = 0.0

    @property
    def summaries(self) -> list[str]:
        lines: list[str] = []
        for match in self.matches:
            lines.append(
                f"{match.role.capitalize()} {match.ocr_name!r} on screen matches {match.credited_name!r} "
                f"({match.ratio:.0f}%)"
            )
        if self.best_title_match:
            lines.append(f"Title card {self.best_title_match!r} matches episode title ({self.title_ratio:.0f}%)")
        return lines


def match_name(
    ocr_name: str,
    candidates: list[str],
    *,
    threshold: float = _DEFAULT_NAME_THRESHOLD,
) -> tuple[str, float] | None:
    """Best fuzzy match of an OCR'd name against credited names (or None)."""
    if not candidates:
        return None
    best_name: str | None = None
    best_ratio = 0.0
    for candidate in candidates:
        ratio = max(
            fuzz.token_sort_ratio(ocr_name, candidate),
            fuzz.ratio(ocr_name, candidate) * 0.98,
        )
        if ratio > best_ratio:
            best_name, best_ratio = candidate, ratio
    if best_name is None or best_ratio < threshold:
        return None
    return best_name, best_ratio


def score_episode_credits(
    hits: list[TitleCardHit],
    episode: EpisodeInfo,
    *,
    weights: dict[str, float] | None = None,
    name_threshold: float = _DEFAULT_NAME_THRESHOLD,
) -> CreditScore:
    """Score OCR'd title-card/credits hits against one candidate episode."""
    weights = weights or {"director": 0.40, "writer": 0.35, "guest": 0.15, "title": 0.35}
    credited_directors = list(episode.directors)
    credited_writers = list(episode.writers)
    credited_guests = [guest.name for guest in episode.guest_stars]

    matches: list[NameMatch] = []
    for hit in hits:
        for name in hit.directors:
            found = match_name(name, credited_directors, threshold=name_threshold)
            if found:
                matches.append(NameMatch(name, found[0], found[1], "director"))
        for name in hit.writers:
            found = match_name(name, credited_writers, threshold=name_threshold)
            if found:
                matches.append(NameMatch(name, found[0], found[1], "writer"))
        for name in hit.guest_stars:
            found = match_name(name, credited_guests, threshold=name_threshold)
            if found:
                matches.append(NameMatch(name, found[0], found[1], "guest"))

    director_score = 0.0
    writer_score = 0.0
    guest_score = 0.0
    for match in matches:
        ratio_factor = match.ratio / 100.0
        if match.role == "director":
            director_score = max(director_score, ratio_factor)
        elif match.role == "writer":
            writer_score = max(writer_score, ratio_factor)
        elif match.role == "guest":
            guest_score = max(guest_score, ratio_factor)

    title_score = 0.0
    best_title: str | None = None
    best_title_ratio = 0.0
    normalized_title = _normalize_title(episode.title)
    for hit in hits:
        for candidate in hit.title_candidates:
            normalized_candidate = _normalize_title(candidate)
            ratio = max(
                fuzz.token_sort_ratio(normalized_candidate, normalized_title),
                fuzz.partial_ratio(normalized_candidate, normalized_title) * 0.97,
            )
            if ratio > best_title_ratio:
                best_title, best_title_ratio = candidate, ratio
    title_score = (best_title_ratio / 100.0) if best_title_ratio >= 80.0 else 0.0
    if best_title is not None and title_score > 0.0:
        matches.append(NameMatch(best_title, episode.title, best_title_ratio, "title"))

    final = (
        weights.get("director", 0.4) * director_score
        + weights.get("writer", 0.35) * writer_score
        + weights.get("guest", 0.15) * guest_score
        + weights.get("title", 0.35) * title_score
    )
    return CreditScore(
        episode_number=episode.episode,
        score=max(0.0, min(1.0, final)),
        matches=matches,
        best_title_match=best_title if title_score > 0.0 else None,
        title_ratio=best_title_ratio,
    )


def _normalize_title(text: str) -> str:
    import re

    return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()


def credit_evidence(hits: list[TitleCardHit], score: CreditScore) -> list[Evidence]:
    """Build user-facing evidence entries from a credit score."""
    evidence: list[Evidence] = []
    for match in score.matches:
        hit = _find_hit_for_name(hits, match.ocr_name)
        evidence.append(
            Evidence(
                pillar=Pillar.VISUAL,
                kind=f"{match.role}_ocr",
                score=max(0.0, min(1.0, match.ratio / 100.0)),
                summary=(
                    f"{match.role.capitalize()} {match.ocr_name!r} recognized on screen at "
                    f"{_format_timestamp(hit.timestamp_seconds) if hit else '?'} matches {match.credited_name!r}"
                ),
                details={
                    "role": match.role,
                    "ocr_name": match.ocr_name,
                    "credited_name": match.credited_name,
                    "ratio": match.ratio,
                },
                timestamp_seconds=hit.timestamp_seconds if hit else None,
                episode_hint=score.episode_number,
            )
        )
    return evidence


def _find_hit_for_name(hits: list[TitleCardHit], name: str) -> TitleCardHit | None:
    for hit in hits:
        if name in hit.directors or name in hit.writers or name in hit.guest_stars or name in hit.title_candidates:
            return hit
    return None


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"
