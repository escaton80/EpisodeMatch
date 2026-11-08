"""Safe file renaming in Plex/Jellyfin/Emby layout with a full undo journal.

Features:

* naming templates (``{series} - S{season:02d}E{episode:02d} - {title}``),
* optional season folders, lowercasing and dot-replacement,
* Windows-safe sanitizing (reserved names, illegal characters),
* collision handling (`` (2)`` suffixes) — never overwrites silently,
* dry-run plans,
* an append-only **undo journal** (``data/undo/*.json``) recording every applied
  batch so ``undo_batch`` can restore the previous state reliably.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from episodematch.core.config import format_episode_name, sanitize_filename
from episodematch.core.exceptions import RenameError
from episodematch.core.models import AssignmentEntry

logger = logging.getLogger(__name__)

_RENAME_SUFFIX_TEMPLATE = " ({count})"


class RenamePlanItem(BaseModel):
    """One source -> destination mapping (relative directory preserved)."""

    source: Path
    destination: Path

    @property
    def file_name(self) -> str:
        return self.source.name


class RenamePlan(BaseModel):
    """A full rename batch (validated as a whole before anything is touched)."""

    items: list[RenamePlanItem] = Field(default_factory=list)
    dry_run: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.items

    def conflicts_within_plan(self) -> list[str]:
        """Destination paths that appear twice inside this plan."""
        seen: dict[str, int] = {}
        for item in self.items:
            key = str(item.destination).lower()
            seen[key] = seen.get(key, 0) + 1
        return [path for path, count in seen.items() if count > 1]


def build_rename_plan(
    entries: list[AssignmentEntry],
    *,
    series: str,
    template: str,
    extension: str = ".mkv",
    season_folder: bool = True,
    lowercase_series: bool = False,
    replace_spaces_with_dots: bool = False,
    dry_run: bool = True,
) -> RenamePlan:
    """Build a validated rename plan from solved assignments.

    ``series`` is the canonical show name (from TMDB) — deriving it from folder
    names is unreliable, so the caller passes it explicitly.
    """
    items: list[RenamePlanItem] = []
    used_destinations: set[str] = set()
    for entry in entries:
        title = entry.episode_title or f"Episode {entry.episode}"
        rendered = format_episode_name(
            template,
            series=series,
            season=entry.season,
            episode=entry.episode,
            title=title,
            extension="",
        )
        parts = rendered.split("/")
        if not season_folder:
            parts = [part for part in parts if not part.lower().startswith("season ")]
        parts = [_apply_style(part, lowercase_series=lowercase_series, dots=replace_spaces_with_dots) for part in parts]
        relative = Path(*parts) if parts else Path(entry.file_path.stem)
        destination = entry.file_path.parent / (str(relative) + extension)
        destination = _resolve_collision(destination, used_destinations)
        used_destinations.add(str(destination).lower())
        items.append(RenamePlanItem(source=entry.file_path, destination=destination))
    return RenamePlan(items=items, dry_run=dry_run)


def _apply_style(part: str, *, lowercase_series: bool, dots: bool) -> str:
    cleaned = part
    if dots:
        cleaned = cleaned.replace(" ", ".")
    if lowercase_series:
        cleaned = cleaned.lower()
    return cleaned


def _resolve_collision(destination: Path, used: set[str]) -> Path:
    """Return a unique destination by appending ' (2)', ' (3)', ... when needed."""
    if not destination.exists() and str(destination).lower() not in used:
        return destination
    counter = 2
    stem = destination.stem
    suffix = destination.suffix
    while True:
        candidate = destination.with_name(f"{stem}{_RENAME_SUFFIX_TEMPLATE.format(count=counter)}{suffix}")
        if not candidate.exists() and str(candidate).lower() not in used:
            return candidate
        counter += 1
        if counter > 999:
            raise RenameError(f"Cannot find a free name for {destination}")


class Renamer:
    """Applies rename plans and journals them for undo."""

    def __init__(self, undo_dir: Path | str) -> None:
        self._undo_dir = Path(undo_dir)
        self._undo_dir.mkdir(parents=True, exist_ok=True)

    # -- validation -----------------------------------------------------------

    @staticmethod
    def validate(plan: RenamePlan) -> None:
        """Raise :class:`RenameError` when the plan is unsafe to apply."""
        internal = plan.conflicts_within_plan()
        if internal:
            raise RenameError(f"Plan contains duplicate destinations: {internal[:3]}")
        for item in plan.items:
            if not item.source.is_file():
                raise RenameError(f"Source missing: {item.source}")
            if item.source == item.destination:
                continue
            if item.destination.exists():
                raise RenameError(f"Destination already exists: {item.destination}")
            if item.source.parent != item.destination.parent:
                item.destination.parent.mkdir(parents=True, exist_ok=True)

    # -- apply / undo ------------------------------------------------------------

    def apply(self, plan: RenamePlan) -> list[RenamePlanItem]:
        """Apply the plan (or nothing when ``dry_run``) and journal it."""
        self.validate(plan)
        if plan.dry_run:
            logger.info("Dry run: %d files would be renamed", len(plan.items))
            return list(plan.items)
        applied: list[RenamePlanItem] = []
        for item in plan.items:
            if item.source == item.destination:
                continue
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            # Rename via a temp name on the same volume to survive case-only renames.
            temp = item.destination.with_name(f".{item.destination.stem}.emtmp{item.destination.suffix}")
            item.source.rename(temp)
            temp.rename(item.destination)
            applied.append(item)
            logger.info("Renamed %s -> %s", item.source.name, item.destination)
        if applied:
            self._write_journal(applied)
        return applied

    def _write_journal(self, applied: list[RenamePlanItem]) -> Path:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        # Two batches can finish within the same microsecond; the unique suffix
        # guarantees the older journal is never overwritten (undo history loss).
        journal_path = self._undo_dir / f"rename-{timestamp}-{uuid4().hex[:8]}.json"
        payload = {
            "timestamp": _dt.datetime.now().isoformat(),
            "operations": [
                {"source": str(item.source), "destination": str(item.destination)} for item in applied
            ],
        }
        journal_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Undo journal written: %s", journal_path)
        return journal_path

    def list_journals(self) -> list[Path]:
        return sorted(self._undo_dir.glob("rename-*.json"), reverse=True)

    def undo_latest(self) -> list[tuple[Path, Path]]:
        """Undo the most recent applied batch; returns (restored_from, restored_to) pairs."""
        journals = self.list_journals()
        if not journals:
            raise RenameError("No rename journal found to undo")
        return self.undo_journal(journals[0])

    def undo_journal(self, journal_path: Path) -> list[tuple[Path, Path]]:
        """Undo one specific batch and delete its journal afterwards."""
        try:
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RenameError(f"Cannot read journal {journal_path}: {exc}") from exc
        restored: list[tuple[Path, Path]] = []
        operations = payload.get("operations") or []
        # Reverse order so season folders empty out before being removed.
        for operation in reversed(operations):
            destination = Path(operation["destination"])
            source = Path(operation["source"])
            if not destination.is_file():
                logger.warning("Undo: destination missing (already moved?): %s", destination)
                continue
            if source.exists():
                raise RenameError(f"Undo conflict: original path exists again: {source}")
            source.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(source)
            restored.append((destination, source))
            # Prune now-empty season folders created by the batch.
            parent = destination.parent
            try:
                next(parent.iterdir())
            except StopIteration:
                parent.rmdir()
            except OSError:
                pass
        journal_path.unlink(missing_ok=True)
        logger.info("Undo complete: %d files restored", len(restored))
        return restored
