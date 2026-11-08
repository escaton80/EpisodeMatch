"""Application configuration: settings, API keys, naming templates.

Configuration is loaded from (highest priority last):

1. built-in defaults,
2. ``data/config.json`` (or the path given explicitly),
3. environment variables (``EPISODEMATCH_*``).

The :class:`AppConfig` model is a plain pydantic model so it can be embedded into other
models and validated. Persistence is handled by :class:`ConfigStore` with atomic writes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from episodematch.core.exceptions import ConfigError

logger = logging.getLogger(__name__)

ENV_PREFIX = "EPISODEMATCH_"

#: Environment variable -> config field, applied after loading the JSON file.
_ENV_OVERRIDES: dict[str, str] = {
    "EPISODEMATCH_TMDB_API_KEY": "tmdb_api_key",
    "EPISODEMATCH_OSDT_API_KEY": "opensubtitles_api_key",
    "EPISODEMATCH_SUBDL_API_KEY": "subdl_api_key",
    "EPISODEMATCH_SUBTITLE_SOURCE": "subtitle_source",
    "EPISODEMATCH_LLM_API_KEY": "llm_api_key",
    "EPISODEMATCH_LLM_BASE_URL": "llm_base_url",
    "EPISODEMATCH_LLM_MODEL": "llm_model",
    "EPISODEMATCH_BIN_DIR": "bin_dir",
    "EPISODEMATCH_DATA_DIR": "data_dir",
}


class PillarWeights(BaseModel):
    """Relative weights of the five pillars when computing the final confidence."""

    model_config = ConfigDict(frozen=True)

    metadata: float = 0.0  # free pre-pass, not part of the weighted sum
    disc_db: float = 0.35
    subtitle: float = 0.30
    audio: float = 0.15
    visual: float = 0.20
    llm: float = 0.40  # the arbiter can confirm/override; applied as a booster

    @model_validator(mode="after")
    def _check_positive(self) -> PillarWeights:
        for name in ("disc_db", "subtitle", "audio", "visual"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"pillar weight {name!r} must be >= 0.0")
        return self

    def as_dict(self) -> dict[str, float]:
        return {
            "disc_db": self.disc_db,
            "subtitle": self.subtitle,
            "audio": self.audio,
            "visual": self.visual,
            "llm": self.llm,
        }


class NamingTemplate(BaseModel):
    """Plex/Jellyfin/Emby-friendly naming configuration."""

    model_config = ConfigDict(frozen=True)

    episode_template: str = "{series}/Season {season:02d}/{series} - S{season:02d}E{episode:02d} - {title}"
    season_folder: bool = True
    lowercase_series: bool = False
    replace_spaces_with_dots: bool = False
    extension: str = ".mkv"

    @field_validator("episode_template")
    @classmethod
    def _validate_template(cls, value: str) -> str:
        if not value or "{series}" not in value:
            raise ValueError("episode_template must contain the {series} placeholder")
        if "{episode:" not in value and "{episode}" not in value:
            raise ValueError("episode_template must contain an {episode} placeholder")
        return value


class ExtractionSettings(BaseModel):
    """Knobs for the extractors (windows, OCR throttling, whisper model)."""

    model_config = ConfigDict(frozen=True)

    # Subtitle dialog window: first/last N minutes of each episode (pillar 2).
    subtitle_window_minutes: float = 10.0
    # Prefer the SubtitleEdit CLI (./bin/SubtitleEdit.exe) over the Python PGS
    # pipeline: native conversion to SRT avoids rendering bitmaps in Python.
    use_subtitle_edit_cli: bool = True
    # Seconds before a SubtitleEdit/seconv CLI conversion is killed. The GUI
    # SubtitleEdit.exe opens a manual OCR dialog for PGS on some versions —
    # the timeout detects that and falls back to the Python pipeline.
    subtitle_edit_timeout: float = 180.0
    # seconv OCR engine (nocr/binaryocr use the DB the SE GUI already downloaded;
    # "tesseract" needs the tesseract binary on PATH).
    seconv_ocr_engine: str = "nocr"
    # Explicit path to the .nocr/.db database; empty = auto-probe %APPDATA%\Subtitle Edit\OCR\.
    seconv_ocr_db: str = ""
    # Python-OCR fallback: hard cap on decoded/OCR'd PGS events (first N only).
    pgs_max_events: int = Field(default=40, ge=1, le=500)
    # Audio windows (pillar 3): start/end in minutes, plus end-relative window.
    audio_window_1_start_min: float = 2.5
    audio_window_1_end_min: float = 4.5
    audio_window_2_minutes_before_end: float = 5.0
    audio_window_2_length_min: float = 2.0
    whisper_model: str = "small"
    whisper_language: str | None = None
    # Frame scanning (pillar 4): first N minutes + credits window.
    title_card_window_minutes: float = 5.0
    credits_window_fraction: float = 0.12
    scene_change_threshold: float = 0.30
    max_frames_per_window: int = 12


class MatchingSettings(BaseModel):
    """Thresholds and weights for the matching engine."""

    model_config = ConfigDict(frozen=True)

    weights: PillarWeights = Field(default_factory=PillarWeights)
    # Confidence at which a match is auto-accepted and deep scanning stops early.
    auto_accept_confidence: float = 0.95
    # Minimum confidence for a solved assignment; below -> unassigned.
    min_assignment_confidence: float = 0.55
    # Fuzzy thresholds.
    name_match_threshold: float = 87.0
    dialog_match_threshold: float = 60.0
    # Credit scorer weights (visual pillar).
    credit_weight_director: float = 0.40
    credit_weight_writer: float = 0.35
    credit_weight_guest: float = 0.15
    credit_weight_title: float = 0.35
    # TheDiscDB matching tolerances.
    discdb_duration_tolerance_seconds: float = 2.0
    # LLM arbiter blending: final = max(local, arbiter) where arbiter weight applies.
    llm_boost: float = 0.15


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk upwards from ``start`` looking for ``pyproject.toml`` (dev checkouts).

    Used to anchor relative ``bin/``/``data/`` paths when the app is started from
    a different working directory (e.g. double-clicking the venv console-script
    shim, whose CWD is ``.venv\\Scripts``). Returns None for frozen/packaged runs
    where no checkout marker exists — those keep CWD-relative behavior.
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _dev_anchor() -> Path | None:
    """Project root when running from a source checkout (venv included).

    Detects the checkout via ``sys.prefix`` (the venv directory in development,
    the bundle directory in frozen builds) and disables itself for frozen
    executables, whose documented behavior is CWD-relative (a copied ``dist``
    folder carries its own ``bin``/``data``).
    """
    if getattr(sys, "frozen", False):
        return None
    prefix = Path(sys.prefix).resolve()
    for candidate in (prefix, *prefix.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _resolve_relative(path: Path) -> Path:
    """Resolve a relative path; the dev checkout outranks the CWD.

    Development (venv/checkout): the project root's ``bin``/``data`` are used
    even when the CWD contains same-named folders — this keeps double-clicked
    shims and accidental CWDs from silently forking the configuration and cache.
    Frozen executables: plain CWD-relative resolution (portable dist layout).
    """
    anchor = _dev_anchor()
    if anchor is not None:
        from_anchor = anchor / path
        if from_anchor.exists():
            return from_anchor
    from_cwd = Path.cwd() / path
    if from_cwd.exists():
        return from_cwd
    return from_anchor / path if anchor is not None else from_cwd


class AppConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(validate_assignment=True)

    # --- API keys ----------------------------------------------------------
    tmdb_api_key: str = ""
    opensubtitles_api_key: str = ""
    subdl_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-4-flash"

    # --- Reference subtitle source ------------------------------------------
    # "opensubtitles" (precise API, free tier: 5 downloads / 24h) or "subdl"
    # (free tier: 2,000 requests + 300 downloads per day) — both implement the
    # same reference_transcript contract and can be swapped freely.
    subtitle_source: str = "opensubtitles"

    # --- Paths -------------------------------------------------------------
    bin_dir: Path = Path("bin")
    data_dir: Path = Path("data")

    # --- Behaviour ---------------------------------------------------------
    naming: NamingTemplate = Field(default_factory=NamingTemplate)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)

    # Preferred subtitle/audio languages, ISO-639-2 codes, most preferred first.
    preferred_languages: list[str] = Field(default_factory=lambda: ["ger", "deu", "eng"])
    # Keep commentary tracks when cleaning streams.
    keep_commentary: bool = False
    # AV1 encoder defaults (SVT-AV1).
    av1_preset: int = Field(default=6, ge=0, le=13)
    av1_crf: int = Field(default=30, ge=0, le=63)

    @field_validator("bin_dir", "data_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Any:
        return Path(value) if isinstance(value, str) else value

    @field_validator("subtitle_source")
    @classmethod
    def _validate_subtitle_source(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ("opensubtitles", "subdl"):
            raise ValueError("subtitle_source must be 'opensubtitles' or 'subdl'")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def _validate_llm_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("llm_base_url must start with http:// or https://")
        return value

    def resolve_path(self, path: Path | str) -> Path:
        """Resolve a possibly relative path against the working directory.

        When the CWD-relative target does not exist (typical when the app was
        started from an unexpected directory, e.g. a double-clicked venv shim),
        the path falls back to the detected project root — see
        :func:`find_project_root`. Absolute paths pass through unchanged.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return _resolve_relative(candidate)

    @property
    def resolved_bin_dir(self) -> Path:
        return self.resolve_path(self.bin_dir)

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve_path(self.data_dir)

    @property
    def database_path(self) -> Path:
        return self.resolved_data_dir / "episodematch.db"

    @property
    def config_path(self) -> Path:
        return self.resolved_data_dir / "config.json"

    @property
    def undo_dir(self) -> Path:
        return self.resolved_data_dir / "undo"

    @property
    def subtitle_cache_dir(self) -> Path:
        return self.resolved_data_dir / "subtitles"

    @property
    def frame_cache_dir(self) -> Path:
        return self.resolved_data_dir / "frames"

    @property
    def audio_cache_dir(self) -> Path:
        return self.resolved_data_dir / "audio"

    def masked(self) -> dict[str, Any]:
        """Dict representation with API keys masked (for logging/UI)."""
        data = self.model_dump(mode="json")
        for key in ("tmdb_api_key", "opensubtitles_api_key", "llm_api_key"):
            value = data.get(key) or ""
            data[key] = (value[:4] + "…" + value[-2:]) if len(value) > 8 else ("set" if value else "")
        return data


class ConfigStore:
    """Loads/saves :class:`AppConfig` with atomic writes and env overrides."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._explicit_path = Path(config_path) if config_path is not None else None

    @property
    def path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        # Anchor the default config to the resolved data dir so that a UI started
        # from an unexpected CWD (double-clicked shim) still reads/writes the
        # project's data folder — same resolution rules as bin/data themselves.
        return AppConfig().resolved_data_dir / "config.json"

    def load(self) -> AppConfig:
        """Load config from disk + env overrides. Missing file -> defaults."""
        data: dict[str, Any] = {}
        path = self.path
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"Cannot read configuration file {path}: {exc}",
                    hint="Fix or delete the file to fall back to defaults.",
                ) from exc
            if not isinstance(data, dict):
                raise ConfigError(f"Configuration file {path} must contain a JSON object.")
        config = AppConfig.model_validate(data)
        return apply_env_overrides(config)

    def save(self, config: AppConfig) -> Path:
        """Atomically persist the config; returns the written path."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = config.model_dump(mode="json")
        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".config-", suffix=".tmp")
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp_path, path)
        except OSError as exc:
            raise ConfigError(f"Cannot write configuration file {path}: {exc}") from exc
        logger.info("Configuration saved to %s", path)
        return path


def apply_env_overrides(config: AppConfig, env: dict[str, str] | None = None) -> AppConfig:
    """Apply ``EPISODEMATCH_*`` environment variables onto the config."""
    source = os.environ if env is None else env
    updates: dict[str, Any] = {}
    for env_name, field_name in _ENV_OVERRIDES.items():
        value = source.get(env_name)
        if value is None or value == "":
            continue
        current = getattr(config, field_name, None)
        if isinstance(current, Path):
            updates[field_name] = Path(value)
        else:
            updates[field_name] = value
    if updates:
        # Re-validate so field validators (URL normalization, path coercion) run.
        config = config.__class__.model_validate({**config.model_dump(), **updates})
        logger.debug("Applied env overrides: %s", ", ".join(sorted(updates)))
    return config


# ---------------------------------------------------------------------------
# Naming template engine
# ---------------------------------------------------------------------------

_TEMPLATE_TOKEN = re.compile(r"\{(\w+)(?::(\d+)d)?\}")
_ILLEGAL_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(name: str, *, max_length: int = 180) -> str:
    """Sanitize a single path component for Windows filesystems."""
    cleaned = _ILLEGAL_FILENAME.sub("_", name)
    cleaned = cleaned.strip().lstrip(".").rstrip(". ").strip()
    if not cleaned:
        cleaned = "_"
    stem = cleaned.split(".")[0].upper()
    if stem in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(". ")
    return cleaned


def format_episode_name(
    template: str,
    *,
    series: str,
    season: int,
    episode: int,
    title: str,
    extension: str = ".mkv",
    **extra: str,
) -> str:
    """Render a naming template into a relative path (or file name).

    Supported placeholders: ``{series}``, ``{season}``, ``{episode}``, ``{title}``,
    zero-padded variants ``{season:2d}`` / ``{episode:2d}`` / ``{episode:3d}`` and any
    extra keyword placeholders. Each rendered *component* is sanitized for Windows.
    """
    values: dict[str, str] = {
        "series": series,
        "title": title,
        "season": str(season),
        "episode": str(episode),
        **{key: str(value) for key, value in extra.items()},
    }

    def _sub(match: re.Match[str]) -> str:
        token, padding = match.group(1), match.group(2)
        if padding is not None and token in ("season", "episode"):
            number = season if token == "season" else episode
            return str(number).zfill(int(padding))
        if token not in values:
            raise ConfigError(f"Unknown placeholder {{{token}}} in naming template")
        return values[token]

    rendered = _TEMPLATE_TOKEN.sub(_sub, template)
    # Append extension when the template does not already contain one.
    if extension and not rendered.lower().endswith(extension.lower()):
        rendered = rendered + extension
    parts = [sanitize_filename(part) for part in rendered.split("/") if part]
    return "/".join(parts)
