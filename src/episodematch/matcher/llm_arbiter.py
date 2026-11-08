"""LLM deep-context arbiter (pillar 5) via any OpenAI-compatible chat API.

All evidence of *all* files of a folder is bundled with the full season episode
guide and sent to the configured model (default: GLM Flash at BigModel's
OpenAI-compatible endpoint). The model returns a strict JSON assignment verdict
which is validated with pydantic, retried on malformed output, and merged into
the local score matrix as an additional, strongly weighted pillar.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from episodematch.core.exceptions import LlmError, LlmResponseFormatError
from episodematch.core.models import (
    ArbiterAssignment,
    ArbiterVerdict,
    EpisodeInfo,
    Evidence,
    MatchCandidate,
    Pillar,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 90.0
_MAX_EVIDENCE_PER_FILE = 12
_MAX_PLOT_CHARS = 400


@dataclass(frozen=True)
class LlmSettings:
    """Connection settings for the OpenAI-compatible endpoint."""

    api_key: str
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-4-flash"
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout: float = _DEFAULT_TIMEOUT


def test_connection(
    settings: LlmSettings,
    *,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> tuple[bool, str]:
    """Send a minimal chat request and report whether the endpoint works.

    Used by the Settings view's "Test LLM connection" button (runs off the UI
    thread via a worker). Returns ``(ok, message)`` — the message is user-facing
    and includes the model's reply on success or the failure reason (bad key,
    HTTP status, network error). Accepts an injected client for offline tests.
    """
    if not settings.api_key:
        return False, "No API key configured"
    own_client = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"Authorization": f"Bearer {settings.api_key}"},
    )
    try:
        url = f"{settings.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": settings.model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "temperature": 0,
            "max_tokens": 10,
        }
        response = http.post(url, json=body)
        if response.status_code == 200:
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                return True, f"{settings.model} answered, but with an unexpected response structure"
            reply = str(content or "").strip()
            return True, f"Connected — {settings.model} replied: {reply[:80]!r}"
        if response.status_code == 401:
            return False, "API key rejected (HTTP 401) — check llm_api_key"
        if response.status_code == 404:
            return False, f"Model or endpoint not found (HTTP 404) — check llm_base_url/llm_model ({url})"
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    except httpx.HTTPError as exc:
        return False, f"Connection failed: {exc}"
    finally:
        if own_client:
            http.close()


# ---------------------------------------------------------------------------
# Prompt building (pure functions)
# ---------------------------------------------------------------------------


def build_dossier(
    candidates: Sequence[MatchCandidate],
    guide_episodes: Sequence[EpisodeInfo],
    *,
    season: int,
) -> dict[str, Any]:
    """Build the JSON dossier sent to the LLM (files + evidence + episode guide)."""
    files: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        evidence_items: list[dict[str, Any]] = []
        for evidence in sorted(candidate.evidence, key=lambda item: item.score, reverse=True)[:_MAX_EVIDENCE_PER_FILE]:
            evidence_items.append(
                {
                    "pillar": evidence.pillar.value,
                    "kind": evidence.kind,
                    "score": round(evidence.score, 3),
                    "summary": evidence.summary[:300],
                    "at_seconds": evidence.timestamp_seconds,
                    "episode_hint": evidence.episode_hint,
                }
            )
        files.append(
            {
                "file": candidate.file_path.name,
                "index": index,
                "evidence": evidence_items,
                "episode_scores": {str(k): round(v, 3) for k, v in sorted(candidate.episode_scores.items())},
            }
        )
    episodes: list[dict[str, Any]] = []
    for episode in guide_episodes:
        episodes.append(
            {
                "episode": episode.episode,
                "title": episode.title,
                "plot": (episode.plot or "")[:_MAX_PLOT_CHARS],
                "directors": episode.directors,
                "writers": episode.writers,
                "guest_stars": [guest.name for guest in episode.guest_stars][:8],
                "runtime_minutes": (episode.runtime_seconds // 60) if episode.runtime_seconds else None,
            }
        )
    return {"season": season, "files": files, "episodes": episodes}


def build_prompt(dossier: dict[str, Any]) -> str:
    """Build the arbiter system+user prompt requesting a strict JSON verdict."""
    system = (
        "You are a TV episode identification arbiter. You receive technical and textual "
        "evidence for several ripped MKV files (from one Blu-ray/DVD season disc) and the "
        "complete episode guide of that season. Assign every file to exactly one episode "
        "of the season, using ALL evidence (playlists, durations, dialog, OCR'd credits, "
        "titles). Episodes must not be assigned twice unless evidence overwhelmingly "
        "supports a duplicate. Answer with STRICT JSON only, no markdown fences, matching:\n"
        "{\"assignments\": [{\"file\": \"<file name>\", \"episode\": <int>, "
        "\"confidence\": <0.0..1.0>, \"reason\": \"<one short sentence>\"}], \"notes\": \"<optional>\"}"
    )
    user = json.dumps(dossier, ensure_ascii=False, indent=1)
    return f"{system}\n\nDATA:\n{user}"


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from an LLM response (fence-tolerant)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise LlmResponseFormatError(f"LLM response contains no JSON object: {cleaned[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise LlmResponseFormatError(f"LLM JSON is malformed: {exc}") from exc
    raise LlmResponseFormatError("LLM JSON object is not closed")


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------


class LlmArbiter:
    """Calls an OpenAI-compatible chat completions API for the global assignment."""

    def __init__(
        self,
        settings: LlmSettings,
        *,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if not settings.api_key:
            raise LlmError(
                "LLM API key is missing",
                hint="Set EPISODEMATCH_LLM_API_KEY (BigModel/GLM keys work out of the box).",
            )
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                follow_redirects=True,
                timeout=self._settings.timeout,
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
            )
        return self._client

    def arbitrate(
        self,
        candidates: Sequence[MatchCandidate],
        guide_episodes: Sequence[EpisodeInfo],
        *,
        season: int,
    ) -> ArbiterVerdict:
        """Ask the LLM for a global file->episode assignment."""
        if not candidates:
            return ArbiterVerdict(model=self._settings.model)
        dossier = build_dossier(candidates, guide_episodes, season=season)
        prompt = build_prompt(dossier)
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                raw = self._chat(prompt)
                payload = extract_json_object(raw)
                verdict = self._validate(payload, raw)
                return verdict
            except (LlmError, LlmResponseFormatError, httpx.HTTPError) as exc:
                last_error = exc
                logger.warning("LLM arbiter attempt %d/%d failed: %s", attempt, self._max_retries, exc)
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * attempt)
        raise LlmError(
            f"LLM arbiter failed after {self._max_retries} attempts: {last_error}",
            hint="Check llm_base_url/llm_model/llm_api_key in the configuration.",
        )

    def _chat(self, prompt: str) -> str:
        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        try:
            response = self.client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise LlmError(f"LLM request failed: {exc}") from exc
        if response.status_code == 401:
            raise LlmError("LLM API key rejected (HTTP 401)", status_code=401)
        if response.status_code != 200:
            raise LlmError(
                f"LLM endpoint returned HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
            )
        payload = response.json()
        try:
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"Unexpected LLM response structure: {json.dumps(payload)[:300]}") from exc
        return str(content or "")

    def _validate(self, payload: dict[str, Any], raw: str) -> ArbiterVerdict:
        assignments: list[ArbiterAssignment] = []
        raw_assignments = payload.get("assignments")
        if not isinstance(raw_assignments, list):
            raise LlmResponseFormatError("LLM verdict misses the 'assignments' list")
        for item in raw_assignments:
            if not isinstance(item, dict):
                continue
            episode = item.get("episode")
            confidence = item.get("confidence", 0.0)
            try:
                confidence_value = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence_value = 0.0
            episode_value: int | None
            try:
                episode_value = int(episode) if episode is not None else None
            except (TypeError, ValueError):
                episode_value = None
            assignments.append(
                ArbiterAssignment(
                    file=str(item.get("file") or ""),
                    episode=episode_value,
                    confidence=confidence_value,
                    reason=str(item.get("reason") or ""),
                )
            )
        if not assignments:
            raise LlmResponseFormatError("LLM verdict contains no assignments")
        return ArbiterVerdict(
            assignments=assignments,
            notes=str(payload.get("notes") or ""),
            model=self._settings.model,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Merging into local candidates
# ---------------------------------------------------------------------------


def merge_verdict_into_candidates(
    verdict: ArbiterVerdict,
    candidates: list[MatchCandidate],
    *,
    boost: float = 0.15,
) -> None:
    """Fold the arbiter verdict into the local score matrix (in place).

    The arbiter acts as pillar 5: its confidence (plus an optional boost) is added
    to the candidate's episode score for the suggested episode, and its reasoning
    is attached as evidence.
    """
    by_name = {candidate.file_path.name: candidate for candidate in candidates}
    lowered = {name.lower(): candidate for name, candidate in by_name.items()}
    for assignment in verdict.assignments:
        candidate = by_name.get(assignment.file) or lowered.get(assignment.file.lower())
        if candidate is None or assignment.episode is None:
            continue
        contribution = assignment.confidence + boost
        candidate.episode_scores[assignment.episode] = (
            candidate.episode_scores.get(assignment.episode, 0.0) + contribution
        )
        candidate.evidence.append(
            Evidence(
                pillar=Pillar.LLM,
                kind="arbiter_assignment",
                score=assignment.confidence,
                summary=(
                    f"LLM arbiter ({verdict.model or 'model'}) assigns {assignment.file} to "
                    f"E{assignment.episode:02d} at {assignment.confidence:.0%}: {assignment.reason}"
                ),
                details={"reason": assignment.reason, "notes": verdict.notes},
                episode_hint=assignment.episode,
            )
        )
