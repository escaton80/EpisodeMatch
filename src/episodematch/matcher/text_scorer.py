"""Text scoring: fuzzy dialog comparison and TF-IDF similarity (pillar 2 helpers).

The dialog measure is **reference-coverage based** (not top-K best matches):
for a candidate episode, we count how many *distinct* reference lines are
confirmed (fuzzy >= strong cutoff) by some extracted line, divided by the
reference size. This is discriminative — a wrong episode only produces a
handful of generic collisions, while the true episode confirms a large share
of its own reference dialog. Top-K "best match" scoring was removed because a
large transcript always finds a dozen fuzzy hits in ANY reference, flattening
every episode to 60–70%.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process

from episodematch.core.languages import languages_match
from episodematch.core.models import EpisodeInfo, SubtitleLine, Transcript

_SPEAKER_PREFIX = re.compile(r"^\s*[-–—]?\s*[A-Z][A-Z0-9 .'\-]{1,24}:\s+")
_BRACKETED = re.compile(r"[\[\(][^\]\)]{0,80}[\]\)]")
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    he her him his how i if in into is it its me my not of on or our she so than that
    the their them then there these they this to too up us was we were what when where
    which who why will with would you your just know like get got going gonna really
    okay yeah yes no oh well now here them they them what when""".split()
)

_MIN_DIALOG_LINE_LENGTH = 12
_TOP_K_LINES = 12


def normalize_dialog_text(text: str) -> str:
    """Normalize one dialog line for fuzzy comparison (speaker labels, HI, case)."""
    cleaned = text
    cleaned = _BRACKETED.sub(" ", cleaned)
    cleaned = _SPEAKER_PREFIX.sub("", cleaned)
    cleaned = cleaned.lower()
    cleaned = _PUNCTUATION.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return cleaned.strip()


def normalize_episode_text(text: str) -> str:
    """Normalize longer prose (plots, synopses) and drop stopwords for TF-IDF."""
    cleaned = _PUNCTUATION.sub(" ", (text or "").lower())
    tokens = [token for token in _WHITESPACE.split(cleaned) if token and token not in _STOPWORDS]
    return " ".join(tokens)


def meaningful_lines(lines: list[SubtitleLine]) -> list[str]:
    """Extract normalized dialog lines that carry enough content to match on."""
    result: list[str] = []
    for line in lines:
        normalized = normalize_dialog_text(line.text)
        if len(normalized) >= _MIN_DIALOG_LINE_LENGTH:
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (self-contained)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [token for token in _WHITESPACE.split(normalize_episode_text(text)) if token]


def _term_frequencies(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def tfidf_cosine(query: str, documents: list[str]) -> list[float]:
    """Cosine similarity between ``query`` and each document using TF-IDF weights.

    IDF is computed over the provided document collection (plus the query as one
    extra pseudo-document so unseen query terms are not infinitely weighted).
    Returns one similarity in [0, 1] per document.
    """
    if not documents:
        return []
    query_tokens = _tokenize(query)
    doc_tokens = [_tokenize(doc) for doc in documents]
    corpus = doc_tokens + [query_tokens]

    document_frequency: dict[str, int] = {}
    for tokens in corpus:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    total_documents = float(len(corpus))

    def idf(token: str) -> float:
        return math.log((1.0 + total_documents) / (1.0 + document_frequency.get(token, 0))) + 1.0

    query_tf = _term_frequencies(query_tokens)
    query_vec = {token: count * idf(token) for token, count in query_tf.items()}
    query_norm = math.sqrt(sum(weight * weight for weight in query_vec.values())) or 1.0

    similarities: list[float] = []
    for tokens in doc_tokens:
        doc_tf = _term_frequencies(tokens)
        doc_vec = {token: count * idf(token) for token, count in doc_tf.items()}
        doc_norm = math.sqrt(sum(weight * weight for weight in doc_vec.values())) or 1.0
        dot = sum(weight * query_vec.get(token, 0.0) for token, weight in doc_vec.items())
        similarities.append(max(0.0, min(1.0, dot / (query_norm * doc_norm))))
    return similarities


# ---------------------------------------------------------------------------
# Dialog alignment via RapidFuzz
# ---------------------------------------------------------------------------


@dataclass
class DialogMatch:
    """Best fuzzy match for one extracted dialog line."""

    query_line: str
    reference_line: str
    score: float  # 0..100 (RapidFuzz scale)
    method: str


def best_line_match(
    query_line: str,
    reference_lines: list[str],
    *,
    score_cutoff: float = 40.0,
) -> DialogMatch | None:
    """Find the best-matching reference line for one query line."""
    if not reference_lines:
        return None
    best: DialogMatch | None = None
    for reference in reference_lines:
        token_set = fuzz.token_set_ratio(query_line, reference)
        partial = fuzz.partial_ratio(query_line, reference)
        combined = 0.7 * token_set + 0.3 * partial
        method = "token_set+partial"
        if best is None or combined > best.score:
            best = DialogMatch(query_line=query_line, reference_line=reference, score=combined, method=method)
    if best is None or best.score < score_cutoff:
        return None
    return best


@dataclass
class DialogAlignment:
    """Coverage-based alignment of extracted lines against reference lines."""

    score: float = 0.0  # 0..1
    matched: list[DialogMatch] = field(default_factory=list)
    unmatched_count: int = 0
    coverage: float = 0.0  # share of unique reference lines confirmed (>= strong)
    confirmed_refs: int = 0
    reference_size: int = 0

    @property
    def matched_pairs(self) -> list[tuple[str, str, float]]:
        return [(match.query_line, match.reference_line, match.score) for match in self.matched]


def align_dialogs(
    extracted_lines: list[str],
    reference_lines: list[str],
    *,
    score_cutoff: float = 40.0,
    strong_cutoff: float = 80.0,
    coverage_target: float = 0.40,
) -> DialogAlignment:
    """Coverage-based dialog scoring (the discriminative pillar-2 measure).

    How many DISTINCT reference lines of the candidate episode are confirmed
    (fuzzy >= ``strong_cutoff``) by at least one extracted line, relative to the
    full unique reference size? A wrong episode only produces generic collisions
    (~15-20%, recap lines etc.); the true episode confirms most of its own
    dialog. Matches below the cutoff are ignored entirely.

    The similarity matrix is computed with ``rapidfuzz.process.cdist`` (C++,
    parallel) — a full film (~500 lines) against a full reference (~500 lines)
    stays fast.
    """
    if not extracted_lines or not reference_lines:
        return DialogAlignment(score=0.0, unmatched_count=len(extracted_lines))

    unique_refs: list[str] = list(dict.fromkeys(reference_lines))
    denominator = len(unique_refs)
    if denominator <= 0:
        return DialogAlignment(score=0.0, unmatched_count=len(extracted_lines))

    matrix = process.cdist(
        extracted_lines,
        unique_refs,
        scorer=fuzz.token_set_ratio,
        score_cutoff=score_cutoff,
        workers=-1,
    )
    best_per_ref = np.asarray(matrix).max(axis=0)
    covered_mask = best_per_ref >= strong_cutoff
    confirmed = int(covered_mask.sum())
    coverage = confirmed / denominator
    mean_score = (float(best_per_ref[covered_mask].mean()) / 100.0) if confirmed else 0.0
    score = min(1.0, coverage / coverage_target) * (0.7 + 0.3 * mean_score)

    # Evidence pairs: strongest extracted line per confirmed reference.
    pairs: list[DialogMatch] = []
    if confirmed:
        best_per_query = np.asarray(matrix).max(axis=1)
        order = np.argsort(best_per_query)[::-1][:6]
        for query_index in order:
            ref_index = int(np.asarray(matrix)[query_index].argmax())
            pairs.append(
                DialogMatch(
                    query_line=extracted_lines[int(query_index)],
                    reference_line=unique_refs[ref_index],
                    score=float(matrix[int(query_index), ref_index]),
                    method="coverage",
                )
            )
    return DialogAlignment(
        score=max(0.0, min(1.0, score)),
        matched=pairs,
        unmatched_count=len(extracted_lines),
        coverage=coverage,
        confirmed_refs=confirmed,
        reference_size=len(unique_refs),
    )


# ---------------------------------------------------------------------------
# Episode scoring facade
# ---------------------------------------------------------------------------


@dataclass
class TextScore:
    """Complete pillar-2 score of one file against one candidate episode."""

    episode_number: int
    dialog_score: float
    plot_score: float
    final_score: float
    matched_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    language_mismatch: str | None = None  # e.g. "de vs en" when dialog scoring was skipped
    coverage: float = 0.0  # share of the candidate's reference dialog confirmed

    @property
    def summary(self) -> str:
        percent = int(round(self.final_score * 100))
        coverage = int(round(self.coverage * 100)) if self.coverage else 0
        pairs = len(self.matched_pairs)
        text = f"{percent}% text match ({pairs} strongest matches, {coverage}% reference dialog covered, plot overlap {self.plot_score:.2f})"
        if self.language_mismatch:
            text += f" — dialog skipped, language mismatch ({self.language_mismatch})"
        return text


def score_episode(
    transcript: Transcript,
    episode: EpisodeInfo,
    *,
    reference_transcript: Transcript | None,
    dialog_cutoff: float = 40.0,
) -> TextScore:
    """Score one transcript against one candidate episode.

    Weights: dialog alignment 0.65 (when a reference exists), plot overlap 0.35.
    Without a reference transcript — or when the reference's language differs
    from the extracted track's language (fuzzy-matching German against English
    would only produce noise) — only the plot overlap is used and the skip is
    reported via ``language_mismatch``.

    Conclusive reference coverage (>= 0.85 of the candidate's own reference
    dialog confirmed) sets a confidence floor of 0.93: the extracted dialog
    literally matches the episode's subtitle — stronger than any keyword overlap.
    """
    plot_blob = " ".join(
        part for part in (episode.title, episode.plot or "", episode.long_synopsis or "") if part
    )
    transcript_text = transcript.text
    plot_scores = tfidf_cosine(transcript_text, [plot_blob]) if plot_blob.strip() else [0.0]
    plot_score = plot_scores[0] if plot_scores else 0.0
    # TF-IDF plot overlap alone rarely exceeds ~0.35; rescale for a useful signal.
    plot_score = min(1.0, plot_score / 0.35)

    language_mismatch: str | None = None
    if reference_transcript is not None and not languages_match(
        transcript.language, reference_transcript.language
    ):
        from episodematch.core.languages import to_osdt_language

        language_mismatch = (
            f"{to_osdt_language(transcript.language) or '?'} vs "
            f"{to_osdt_language(reference_transcript.language) or '?'}"
        )

    extracted = meaningful_lines(transcript.lines)
    if reference_transcript is None or language_mismatch:
        dialog_score = 0.0
        pairs: list[tuple[str, str, float]] = []
        coverage = 0.0
        final = 0.35 * plot_score
    else:
        reference = meaningful_lines(reference_transcript.lines)
        alignment = align_dialogs(extracted, reference, score_cutoff=dialog_cutoff)
        dialog_score = alignment.score
        pairs = alignment.matched_pairs
        coverage = alignment.coverage
        final = 0.65 * dialog_score + 0.35 * plot_score
        if coverage >= 0.85 and dialog_score >= 0.9:
            final = max(final, 0.93)
    return TextScore(
        episode_number=episode.episode,
        dialog_score=dialog_score,
        plot_score=plot_score,
        final_score=max(0.0, min(1.0, final)),
        matched_pairs=pairs[:6],
        language_mismatch=language_mismatch,
        coverage=coverage,
    )
