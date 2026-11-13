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


def repoint_entries(entries: list[AssignmentEntry], applied: list[RenamePlanItem]) -> int:
    """Update solved entries to the NEW paths after renames were applied.

    Audit B4: ``--rename --tag`` in one CLI run must tag the renamed files, not
    the vanished originals. Mutates ``entries`` in place; returns the number of
    entries that were repointed.
    """
    source_to_destination = {item.source: item.destination for item in applied if item.source != item.destination}
    if not source_to_destination:
        return 0
    repointed = 0
    for entry in entries:
        destination = source_to_destination.get(entry.file_path)
        if destination is not None:
            entry.file_path = destination
            repointed += 1
    return repointed


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
            # Audit B1: a stale temporary file from an earlier crashed run would
            # abort the batch *mid-loop* (after earlier renames) — fail fast here.
            temp = item.destination.with_name(f".{item.destination.stem}.emtmp{item.destination.suffix}")
            if temp.exists():
                raise RenameError(f"Temporary rename target already exists: {temp}")
            if item.source.parent != item.destination.parent:
                item.destination.parent.mkdir(parents=True, exist_ok=True)

    # -- apply / undo ------------------------------------------------------------

    def apply(self, plan: RenamePlan) -> list[RenamePlanItem]:
        """Apply the plan and journal every completed rename incrementally.

        Audit B1: the journal is written *while* the batch runs, so a mid-batch
        failure (disk error, vanished source) leaves an undo journal for the
        renames that already happened — and OSError is wrapped into
        :class:`RenameError` so the CLI reports it properly.
        """
        self.validate(plan)
        if plan.dry_run:
            logger.info("Dry run: %d files would be renamed", len(plan.items))
            return list(plan.items)
        applied: list[RenamePlanItem] = []
        operations: list[dict[str, str]] = []
        journal_path = self._new_journal_path()
        try:
            for item in plan.items:
                if item.source == item.destination:
                    continue
                item.destination.parent.mkdir(parents=True, exist_ok=True)
                temp = item.destination.with_name(f".{item.destination.stem}.emtmp{item.destination.suffix}")
                if temp.exists():
                    raise RenameError(f"Temporary rename target already exists: {temp}")
                try:
                    item.source.rename(temp)
                    temp.rename(item.destination)
                except OSError as exc:
                    raise RenameError(f"Rename failed for {item.source.name}: {exc}") from exc
                applied.append(item)
                operations.append({"source": str(item.source), "destination": str(item.destination)})
                # Incremental journal: every completed rename is undoable, even
                # if the batch is interrupted right after it.
                self._write_journal_file(journal_path, operations)
                logger.info("Renamed %s -> %s", item.source.name, item.destination)
        finally:
            if not operations:
                journal_path.unlink(missing_ok=True)  # nothing applied -> drop empty journal
        return applied

    def _new_journal_path(self) -> Path:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return self._undo_dir / f"rename-{timestamp}-{uuid4().hex[:8]}.json"

    @staticmethod
    def _write_journal_file(journal_path: Path, operations: list[dict[str, str]]) -> None:
        payload = {
            "timestamp": _dt.datetime.now().isoformat(),
            "operations": [dict(op) for op in operations],
        }
        journal_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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
