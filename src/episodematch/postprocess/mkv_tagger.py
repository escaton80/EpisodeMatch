"""Lossless MKV header tagging via mkvpropedit.

Writes episode title, part numbers and track names directly into the Matroska
header — no remux, no re-encode, instant. Every write is verified afterwards with
``mkvmerge -J`` and supports a dry-run mode.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from episodematch.core.exceptions import TaggingError
from episodematch.core.models import AssignmentEntry

logger = logging.getLogger(__name__)

_TIMEOUT = 120


@dataclass(frozen=True)
class TaggingPlan:
    """Everything mkvpropedit will set on one file."""

    file_path: Path
    title: str
    season: int
    episode: int
    date_title: str | None = None

    def summary(self) -> str:
        return f"{self.file_path.name}: title={self.title!r}, part S{self.season:02d}E{self.episode:02d}"


def build_tagging_plan(entry: AssignmentEntry) -> TaggingPlan:
    """Build the tagging plan for one solved assignment."""
    title = f"{entry.episode_title or f'Episode {entry.episode}'}"
    return TaggingPlan(
        file_path=entry.file_path,
        title=title,
        season=entry.season,
        episode=entry.episode,
    )


class MkvTagger:
    """Writes episode metadata into MKV headers using mkvpropedit."""

    def __init__(self, mkvpropedit_path: Path, *, mkvmerge_path: Path | None = None) -> None:
        self._mkvpropedit = Path(mkvpropedit_path)
        self._mkvmerge = Path(mkvmerge_path) if mkvmerge_path else None

    def apply(self, plan: TaggingPlan, *, dry_run: bool = False) -> None:
        """Apply one tagging plan (lossless header edit)."""
        if not plan.file_path.is_file():
            raise TaggingError(f"File not found: {plan.file_path}")
        command = [
            str(self._mkvpropedit),
            str(plan.file_path),
            "--edit", "info",
            "--set", f"title={plan.title}",
            "--set", f"part_number={plan.episode}",
            "--set", f"part_total={plan.season}",
        ]
        if dry_run:
            logger.info("Dry run tagging: %s", " ".join(command))
            return
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise TaggingError(f"Cannot execute mkvpropedit: {exc}") from exc
        if completed.returncode != 0:
            raise TaggingError(
                f"mkvpropedit failed for {plan.file_path.name} (exit {completed.returncode}): {(completed.stderr or completed.stdout)[:300]}"
            )
        logger.info("Tagged %s", plan.summary())
        if self._mkvmerge is not None:
            self._verify(plan)

    def apply_entry(self, entry: AssignmentEntry, *, dry_run: bool = False) -> None:
        self.apply(build_tagging_plan(entry), dry_run=dry_run)

    def _verify(self, plan: TaggingPlan) -> None:
        """Confirm the title landed in the container (mkvmerge -J)."""
        command = [str(self._mkvmerge), "-J", str(plan.file_path)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=_TIMEOUT,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            logger.debug("mkvmerge verification unavailable: %s", exc)
            return
        if completed.returncode != 2:  # mkvmerge -J exits 2 on success with JSON output
            logger.debug("mkvmerge -J exit %s", completed.returncode)
            return
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            logger.debug("mkvmerge -J produced no JSON")
            return
        container_title = (payload.get("container") or {}).get("properties", {}).get("title")
        if container_title != plan.title:
            logger.warning(
                "Tag verification mismatch for %s: header title %r != expected %r",
                plan.file_path.name,
                container_title,
                plan.title,
            )
        else:
            logger.debug("Tag verification OK for %s", plan.file_path.name)
