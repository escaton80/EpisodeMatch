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

import math
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from episodematch.core.models import (
    EpisodeInfo,
    Evidence,
    Pillar,
    SeasonGuide,
    SeriesInfo,
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


def build_credit_idf(guide: SeasonGuide) -> dict[tuple[str, str], float] | None:
    """Normalized season-IDF for credit names (PILLAR4_PLAN batch 3).

    A name credited in k of N season episodes gets the multiplier
    ``log(N / k) / log(N)`` — always in ``[0.0, 1.0]`` (k=1 -> 1.0, k=N -> 0.0),
    so long seasons cannot inflate weights above 1.0 and a showrunner credited
    in every episode fades to zero. Requires >= 3 episodes; below that ``None``
    (the caller keeps the coverage unweighted and logs it). Keys are
    ``(role, credited_name.lower())``.
    """
    episodes = [episode for episode in guide.episodes]
    if len(episodes) < 3:
        return None
    counts: Counter[tuple[str, str]] = Counter()
    for episode in episodes:
        for name in episode.directors:
            counts[("director", name.lower())] += 1
        for name in episode.writers:
            counts[("writer", name.lower())] += 1
        for guest in episode.guest_stars:
            counts[("guest", guest.name.lower())] += 1
    total = len(episodes)
    return {key: math.log(total / count) / math.log(total) for key, count in counts.items()}


def score_episode_credits(
    hits: list[TitleCardHit],
    episode: EpisodeInfo,
    *,
    weights: dict[str, float] | None = None,
    name_threshold: float = _DEFAULT_NAME_THRESHOLD,
    credit_idf: dict[tuple[str, str], float] | None = None,
    series: SeriesInfo | None = None,
    file_duration: float | None = None,
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
        # PILLAR4_PLAN batch 3: a name credited in many season episodes fades
        # via normalized IDF (log(N/k)/log(N)); names absent from the IDF map
        # keep full weight.
        if credit_idf is not None:
            ratio_factor *= credit_idf.get((match.role, match.credited_name.lower()), 1.0)
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
    # PILLAR4_PLAN batch 3.6 (Rockos calibration): two show-generic traps
    # measured on Rocko's Modern Life S1 —
    #  a) the series logo on EVERY card partial-matches episode titles that
    #     start with it ("ROCKO" -> "Rocko's Happy Sack" at 100%),
    #  b) the NEXT-EPISODE teaser card sits in the credits tail (23:05 of a
    #     ~24 min file) and matches the NEXT episode at 85%.
    series_normalized = _normalize_title(series.name) if series and series.name else ""
    for hit in hits:
        candidates_for_hit = list(hit.title_candidates)
        # b) teaser zone: cards in the credits tail belong to the NEXT episode
        if file_duration is not None and file_duration > 0 and hit.timestamp_seconds > file_duration * 0.85:
            candidates_for_hit = []
        for candidate in candidates_for_hit:
            normalized_candidate = _normalize_title(candidate)
            # a) show-generic filter: the candidate IS (or is embedded in) the
            # series name -> the series logo, never an episode discriminator.
            if series_normalized and normalized_candidate:
                if (
                    fuzz.partial_ratio(series_normalized, normalized_candidate) >= 80
                    or fuzz.partial_ratio(normalized_candidate, series_normalized) >= 80
                ):
                    continue
            # PILLAR4_PLAN batch 3 (short-title protection): below 5 characters
            # partial_ratio is disabled entirely — "Home" must not hit
            # "Welcome Home" at 100%; only a full-line match >= 95 confirms.
            if len(normalized_candidate) < 5 or len(normalized_title) < 5:
                ratio = fuzz.token_sort_ratio(normalized_candidate, normalized_title)
                confirmed = ratio >= 95
            else:
                ratio = max(
                    fuzz.token_sort_ratio(normalized_candidate, normalized_title),
                    fuzz.partial_ratio(normalized_candidate, normalized_title) * 0.97,
                )
                confirmed = ratio >= 80.0
            if confirmed and ratio > best_title_ratio:
                best_title, best_title_ratio = candidate, ratio
    title_score = (best_title_ratio / 100.0) if best_title is not None else 0.0
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

    normalized = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    # Voyager case: OCR reads 'THE37\'s' (no space between letters and digits),
    # the TMDB title is 'The 37\'s' — split letter/digit boundaries so both
    # forms normalize identically ('the37s' -> 'the 37 s').
    normalized = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", normalized)
    normalized = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def credit_evidence(hits: list[TitleCardHit], score: CreditScore) -> list[Evidence]:
    """Build user-facing evidence entries from a credit score.

    Per-hit items are display-only (``episode_hint`` stays unset) — the
    calibrated contribution to the assignment comes from the summary evidence,
    so the aggregation never double-counts one observation (audit A4a).
    """
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
