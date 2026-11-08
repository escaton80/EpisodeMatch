"""Language code helpers: MKV/ISO-639-2 codes -> OpenSubtitles (ISO-639-1).

Matroska language tags use ISO 639-2 (``ger``/``deu``, ``eng``, ...), while the
OpenSubtitles REST API expects ISO 639-1 (``de``, ``en``, ...). This module maps
between them and builds the candidate chain used when requesting reference
subtitles (extracted track language first, then the user's preferences, then a
fallback).
"""

from __future__ import annotations

from typing import Sequence

#: ISO 639-2 (B and T variants) -> ISO 639-1 for languages commonly found on
#: Blu-ray/DVD subtitle tracks. Two-letter codes pass through unchanged.
_ISO_639_2_TO_1: dict[str, str] = {
    "ger": "de", "deu": "de",
    "eng": "en",
    "fre": "fr", "fra": "fr",
    "spa": "es",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "rus": "ru",
    "dut": "nl", "nld": "nl",
    "por": "pt",
    "chi": "zh", "zho": "zh",
    "swe": "sv",
    "nor": "no", "nob": "nb",
    "dan": "da",
    "fin": "fi",
    "pol": "pl",
    "cze": "cs", "ces": "cs",
    "hun": "hu",
    "tur": "tr",
    "gre": "el", "ell": "el",
    "ara": "ar",
    "heb": "he",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "ind": "id",
    "ron": "ro", "rum": "ro",
    "ukr": "uk",
    "bul": "bg",
    "hrv": "hr", "scr": "hr",
    "srp": "sr",
    "slk": "sk", "slo": "sk",
    "slv": "sl",
    "cat": "ca",
    "epo": "eo",
    "est": "et",
    "lav": "lv",
    "lit": "lt",
    "mkd": "mk", "mac": "mk",
    "sqi": "sq", "alb": "sq",
}


def normalize_language(code: str | None) -> str:
    """Lowercase/strip a language code; empty string for None/blank."""
    return (code or "").strip().lower()


def to_osdt_language(code: str | None) -> str:
    """Convert any language tag to the OpenSubtitles (ISO 639-1) form.

    Two-letter codes pass through; known ISO-639-2 variants are mapped; unknown
    codes are returned lowercased unchanged (the API may still understand them).
    """
    normalized = normalize_language(code)
    if not normalized:
        return ""
    if len(normalized) == 2:
        return normalized
    return _ISO_639_2_TO_1.get(normalized, normalized)


def language_candidates(
    primary: str | None,
    preferred: Sequence[str] | None = None,
    *,
    fallback: str = "en",
    limit: int = 4,
) -> list[str]:
    """Build the ordered OpenSubtitles language chain for reference lookups.

    Order: the extracted track's language first (dialog must match dialog!),
    then the user's preferred languages, then the fallback. Duplicates removed,
    capped at ``limit`` to bound API usage.
    """
    candidates: list[str] = []
    for code in (primary, *(preferred or []), fallback):
        converted = to_osdt_language(code)
        if converted and converted not in candidates:
            candidates.append(converted)
        if len(candidates) >= limit:
            break
    return candidates or [to_osdt_language(fallback)]


def languages_match(a: str | None, b: str | None) -> bool:
    """Compare two language tags on their 2-letter form (None = unknown)."""
    first = to_osdt_language(a)
    second = to_osdt_language(b)
    if not first or not second:
        return True  # unknown languages never count as a mismatch
    return first[:2] == second[:2]


__all__ = [
    "language_candidates",
    "languages_match",
    "normalize_language",
    "to_osdt_language",
]
