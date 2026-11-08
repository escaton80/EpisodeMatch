"""Pydantic data models shared across EpisodeMatch.

All timestamps/durations are expressed in *seconds* (``float``) unless noted otherwise.
All scores are expressed in the range ``0.0``–``1.0`` unless noted otherwise.
"""

from __future__ import annotations

from datetime import date as _date
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StreamType(str, Enum):
    """MKV stream (track) type."""

    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    OTHER = "other"


class SubtitleFormat(str, Enum):
    """Concrete codec/format of a subtitle track."""

    SRT = "srt"
    ASS = "ass"
    VTT = "vtt"
    PGS = "pgs"
    VOB = "vob"
    OTHER = "other"


class ToolSource(str, Enum):
    """Where a required external binary was found."""

    LOCAL_BIN = "local_bin"
    SYSTEM_PATH = "system_path"
    PROGRAM_FILES = "program_files"
    NOT_FOUND = "not_found"


class Pillar(str, Enum):
    """The five identification pillars plus the fast metadata pre-pass."""

    METADATA = "metadata"
    DISC_DB = "disc_db"
    SUBTITLE = "subtitle"
    AUDIO = "audio"
    VISUAL = "visual"
    LLM = "llm"


class TitleCardKind(str, Enum):
    """Classification of an OCR'd keyframe."""

    TITLE_CARD = "title_card"
    CREDITS = "credits"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    """Ordered pipeline stages, from fast metadata to the LLM arbiter."""

    PROBE = "probe"
    FAST = "fast"
    SUBTITLES = "subtitles"
    AUDIO = "audio"
    VISUAL = "visual"
    LLM = "llm"
    SOLVE = "solve"


# ---------------------------------------------------------------------------
# Tools / environment
# ---------------------------------------------------------------------------


class ToolInfo(BaseModel):
    """Location and version of one external binary."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: Path | None = None
    version: str | None = None
    source: ToolSource = ToolSource.NOT_FOUND

    @property
    def is_available(self) -> bool:
        return self.path is not None and self.source is not ToolSource.NOT_FOUND

    @field_validator("version", mode="before")
    @classmethod
    def _clean_version(cls, value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class ToolStatus(BaseModel):
    """Snapshot of every tool EpisodeMatch knows about."""

    tools: dict[str, ToolInfo] = Field(default_factory=dict)

    def get(self, name: str) -> ToolInfo:
        return self.tools.get(name) or ToolInfo(name=name)

    @property
    def missing(self) -> list[str]:
        return sorted(name for name, info in self.tools.items() if not info.is_available)

    @property
    def all_available(self) -> bool:
        return bool(self.tools) and not self.missing


# ---------------------------------------------------------------------------
# MKV container data
# ---------------------------------------------------------------------------


class StreamInfo(BaseModel):
    """One stream/track inside an MKV container."""

    index: int
    stream_type: StreamType = StreamType.OTHER
    codec_name: str | None = None
    codec_long_name: str | None = None
    language: str | None = None
    title: str | None = None
    default: bool = False
    forced: bool = False
    commentary: bool = False
    hearing_impaired: bool = False
    # Audio specifics
    channels: int | None = None
    sample_rate: int | None = None
    # Video specifics
    width: int | None = None
    height: int | None = None
    frame_rate: str | None = None
    # Subtitle specifics
    subtitle_format: SubtitleFormat | None = None

    @property
    def is_subtitle(self) -> bool:
        return self.stream_type is StreamType.SUBTITLE

    @property
    def is_audio(self) -> bool:
        return self.stream_type is StreamType.AUDIO


class ChapterInfo(BaseModel):
    """One chapter entry."""

    index: int
    start_seconds: float = 0.0
    end_seconds: float | None = None
    title: str | None = None


class MkvFile(BaseModel):
    """Everything EpisodeMatch knows about one ripped .mkv file."""

    path: Path
    size_bytes: int = 0
    duration_seconds: float | None = None
    container_name: str | None = None
    # MakeMKV stores the source playlist (e.g. "00800.mpls") in the MKV title tag.
    title_tag: str | None = None
    source_id: str | None = None
    segment_uid: str | None = None
    overall_bit_rate: int | None = None
    streams: list[StreamInfo] = Field(default_factory=list)
    chapters: list[ChapterInfo] = Field(default_factory=list)

    @property
    def file_name(self) -> str:
        return self.path.name

    @property
    def video_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.stream_type is StreamType.VIDEO]

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.stream_type is StreamType.AUDIO]

    @property
    def subtitle_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.stream_type is StreamType.SUBTITLE]

    @property
    def text_subtitle_streams(self) -> list[StreamInfo]:
        return [s for s in self.subtitle_streams if s.subtitle_format in (SubtitleFormat.SRT, SubtitleFormat.ASS, SubtitleFormat.VTT)]

    @property
    def pgs_subtitle_streams(self) -> list[StreamInfo]:
        return [s for s in self.subtitle_streams if s.subtitle_format is SubtitleFormat.PGS]

    def subtitle_stream_by_index(self, index: int) -> StreamInfo | None:
        for stream in self.streams:
            if stream.index == index:
                return stream
        return None

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value) if not isinstance(value, Path) else value


class DiscFingerprint(BaseModel):
    """Technical fingerprint of one ripped playlist, used for TheDiscDB matching."""

    mpls_name: str | None = None
    duration_seconds: float | None = None
    segment_count: int | None = None
    chapter_count: int | None = None
    size_bytes: int | None = None
    # MakeMKV title number parsed from the output file name ("…_t07.mkv" -> 7).
    # TheDiscDB Titles[].Index is exactly this number for the matching disc, so
    # with a pinned disc URL the index alone identifies the title.
    make_mkv_index: int | None = None

    @classmethod
    def from_mkv(cls, mkv: MkvFile, *, segment_count: int | None = None) -> DiscFingerprint:
        """Derive a fingerprint from probed MKV metadata + the file name."""
        import re

        mpls = mkv.source_id or (mkv.title_tag if mkv.title_tag and mkv.title_tag.lower().endswith(".mpls") else None)
        # MakeMKV output names end in _tNN.mkv (plain or prefixed by the disc
        # label). The number is the title index on the source disc.
        make_mkv_index: int | None = None
        index_match = re.search(r"(?:^|[_\-. ])t(\d{1,4})$", mkv.path.stem, re.IGNORECASE)
        if index_match:
            make_mkv_index = int(index_match.group(1))
        return cls(
            mpls_name=mpls,
            duration_seconds=mkv.duration_seconds,
            segment_count=segment_count,
            chapter_count=len(mkv.chapters) or None,
            size_bytes=mkv.size_bytes or None,
            make_mkv_index=make_mkv_index,
        )


# ---------------------------------------------------------------------------
# Extracted text (subtitles, transcripts, OCR)
# ---------------------------------------------------------------------------


class SubtitleLine(BaseModel):
    """One timed text line (from SRT/ASS/VTT, OCR'd PGS or Whisper)."""

    start_seconds: float
    end_seconds: float | None = None
    text: str
    confidence: float | None = None  # OCR/ASR confidence when known

    @property
    def duration(self) -> float:
        if self.end_seconds is None:
            return 0.0
        return max(0.0, self.end_seconds - self.start_seconds)


class Transcript(BaseModel):
    """Timed text extracted from one source of one file."""

    source: str  # "srt", "ass", "vtt", "pgs-ocr", "whisper"
    language: str | None = None
    track_index: int | None = None
    lines: list[SubtitleLine] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)

    @property
    def character_count(self) -> int:
        return sum(len(line.text) for line in self.lines)


class TranscriptSegment(BaseModel):
    """One Whisper output segment."""

    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None


class OcrLine(BaseModel):
    """One recognized line of text with its bounding box (pixels)."""

    text: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float | None = None


class OcrResult(BaseModel):
    """Result of OCR over a single image."""

    lines: list[OcrLine] = Field(default_factory=list)
    backend: str = "unknown"
    language: str | None = None

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


class TitleCardHit(BaseModel):
    """An OCR'd keyframe that looks like a title card or credits frame."""

    timestamp_seconds: float
    kind: TitleCardKind = TitleCardKind.UNKNOWN
    raw_lines: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    guest_stars: list[str] = Field(default_factory=list)
    title_candidates: list[str] = Field(default_factory=list)
    ocr_confidence: float | None = None
    frame_path: Path | None = None

    @field_validator("frame_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path | None:
        return Path(value) if value is not None and not isinstance(value, Path) else value


# ---------------------------------------------------------------------------
# Providers: series/season/episode metadata
# ---------------------------------------------------------------------------


class GuestStar(BaseModel):
    name: str
    character: str | None = None


class CrewMember(BaseModel):
    name: str
    job: str  # "Director", "Writer", "Story", "Teleplay", ...


class SeriesInfo(BaseModel):
    """Identified series."""

    name: str
    tmdb_id: int | None = None
    imdb_id: str | None = None
    first_air_year: int | None = None
    overview: str | None = None


class EpisodeInfo(BaseModel):
    """One episode of a season, with everything the scorers need."""

    season: int
    episode: int
    title: str
    plot: str | None = None
    long_synopsis: str | None = None
    air_date: _date | None = None
    runtime_seconds: int | None = None
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    guest_stars: list[GuestStar] = Field(default_factory=list)
    tmdb_id: int | None = None
    imdb_id: str | None = None

    @property
    def code(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"

    @property
    def full_name(self) -> str:
        return f"{self.code} - {self.title}"

    def dialog_blob(self) -> str:
        """Concatenated text used by dialog/plot scoring."""
        parts = [self.title or "", self.plot or "", self.long_synopsis or ""]
        parts.extend(guest.name for guest in self.guest_stars)
        return " ".join(p for p in parts if p)

    @field_validator("air_date", mode="before")
    @classmethod
    def _parse_air_date(cls, value: Any) -> Any:
        if isinstance(value, str) and value:
            try:
                return _date.fromisoformat(value[:10])
            except ValueError:
                return None
        return value


class SeasonGuide(BaseModel):
    """Complete episode guide for one season (from TMDB, optionally enriched by IMDb)."""

    series: SeriesInfo
    season: int
    episodes: list[EpisodeInfo] = Field(default_factory=list)

    def by_number(self, episode: int) -> EpisodeInfo | None:
        for ep in self.episodes:
            if ep.episode == episode:
                return ep
        return None


# ---------------------------------------------------------------------------
# Matching: evidence, candidates, assignments
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """One piece of evidence produced by a pillar.

    ``score`` is the pillar-local strength of this single observation (0..1).
    ``summary`` is a short human-readable sentence (shown in the UI evidence card).
    """

    pillar: Pillar
    kind: str  # e.g. "mpls_match", "director_ocr", "dialog_fuzzy", "runtime_match"
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp_seconds: float | None = None  # media time the evidence was observed at
    season_hint: int | None = None
    episode_hint: int | None = None


class MatchCandidate(BaseModel):
    """Per-file accumulated matching state."""

    file_path: Path
    evidence: list[Evidence] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # pillar value -> score
    # episode_number -> accumulated score for that episode of the season
    episode_scores: dict[int, float] = Field(default_factory=dict)

    @field_validator("file_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value) if not isinstance(value, Path) else value

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)
        key = evidence.pillar.value
        # Keep the strongest score per pillar.
        self.scores[key] = max(self.scores.get(key, 0.0), evidence.score)
        if evidence.episode_hint is not None:
            episode = evidence.episode_hint
            contribution = evidence.score * evidence.score  # square: reward sharp evidence
            self.episode_scores[episode] = self.episode_scores.get(episode, 0.0) + contribution


class ArbiterAssignment(BaseModel):
    """One file->episode verdict produced by the LLM arbiter."""

    file: str  # file name as presented to the LLM
    episode: int | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str = ""


class ArbiterVerdict(BaseModel):
    """Full arbiter response for one disc/folder."""

    assignments: list[ArbiterAssignment] = Field(default_factory=list)
    notes: str = ""
    model: str | None = None
    raw_response: str | None = None


class AssignmentEntry(BaseModel):
    """Final, solved assignment of one file."""

    file_path: Path
    season: int
    episode: int
    episode_title: str | None = None
    confidence: float = 0.0
    methods: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("file_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value) if not isinstance(value, Path) else value

    @property
    def code(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"


class UnassignedFile(BaseModel):
    """A file that could not be assigned with sufficient confidence."""

    file_path: Path
    reason: str
    best_guess_episode: int | None = None
    best_guess_confidence: float = 0.0

    @field_validator("file_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value) if not isinstance(value, Path) else value


class SeasonAssignment(BaseModel):
    """The solved assignment for a whole folder/disc."""

    series: SeriesInfo
    season: int
    entries: list[AssignmentEntry] = Field(default_factory=list)
    unassigned: list[UnassignedFile] = Field(default_factory=list)

    @property
    def assigned_count(self) -> int:
        return len(self.entries)

    @property
    def average_confidence(self) -> float:
        if not self.entries:
            return 0.0
        return sum(entry.confidence for entry in self.entries) / len(self.entries)


# ---------------------------------------------------------------------------
# TheDiscDB structures
# ---------------------------------------------------------------------------


class DiscPlaylistClip(BaseModel):
    """One episode reference inside a TheDiscDB playlist."""

    season: int
    episode: int
    order: int = 0  # position of this episode within the playlist
    title: str | None = None


class DiscPlaylist(BaseModel):
    """One MPLS playlist of a disc as known by TheDiscDB."""

    playlist_id: str | None = None  # e.g. "00800.mpls"
    duration_seconds: float | None = None
    segment_count: int | None = None
    chapter_count: int | None = None
    size_bytes: int | None = None
    clips: list[DiscPlaylistClip] = Field(default_factory=list)


class DiscEntry(BaseModel):
    """A disc entry imported from TheDiscDB (HTTP or local JSON)."""

    series_title: str
    disc_title: str | None = None
    upc: str | None = None
    tmdb_id: int | None = None
    playlists: list[DiscPlaylist] = Field(default_factory=list)
    source: str = "unknown"  # "thediscdb" or a local file path


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class ProgressEvent(BaseModel):
    """Emitted by the pipeline to report progress (consumed by UI/CLI)."""

    stage: PipelineStage
    message: str
    file_name: str | None = None
    percent: float | None = Field(default=None, ge=0.0, le=100.0)
