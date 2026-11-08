"""Exception hierarchy for EpisodeMatch.

Every module raises subclasses of :class:`EpisodeMatchError` so callers can catch a single
base class while still being able to react to specific failure modes.
"""

from __future__ import annotations


class EpisodeMatchError(Exception):
    """Base class for all EpisodeMatch errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        text = super().__str__()
        if self.hint:
            return f"{text} (hint: {self.hint})"
        return text


class ConfigError(EpisodeMatchError):
    """Raised when the user configuration is invalid or cannot be persisted."""


class BinaryNotFoundError(EpisodeMatchError):
    """Raised when a required external tool (ffmpeg, mkvmerge, ...) cannot be located."""


class BinaryDownloadError(EpisodeMatchError):
    """Raised when downloading or extracting an external tool fails."""


class ExtractionError(EpisodeMatchError):
    """Raised when data cannot be extracted from an MKV file."""


class PgsParseError(ExtractionError):
    """Raised when a Blu-ray PGS (.sup) stream is malformed."""

    def __init__(self, message: str, *, offset: int | None = None, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.offset = offset


class OcrUnavailableError(EpisodeMatchError):
    """Raised when no OCR backend is usable on the current system."""


class WhisperUnavailableError(EpisodeMatchError):
    """Raised when faster-whisper is not installed or no model is available."""


class ProviderError(EpisodeMatchError):
    """Raised when an external metadata provider (TMDB, IMDb, ...) fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.provider = provider
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised when a provider rejects requests due to rate limiting."""


class LlmError(EpisodeMatchError):
    """Raised when the LLM arbiter request or its response fails."""

    def __init__(self, message: str, *, status_code: int | None = None, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.status_code = status_code


class LlmResponseFormatError(LlmError):
    """Raised when the LLM returned data that is not the requested JSON verdict."""


class SolverError(EpisodeMatchError):
    """Raised when the assignment solver receives an unsolvable problem."""


class RenameError(EpisodeMatchError):
    """Raised when renaming files fails or a rename plan is unsafe."""


class TaggingError(EpisodeMatchError):
    """Raised when writing MKV header metadata fails."""


class StreamCleanError(EpisodeMatchError):
    """Raised when track cleanup / remuxing fails."""


class EncodingError(EpisodeMatchError):
    """Raised when the AV1 encoding queue hits a hard failure."""
