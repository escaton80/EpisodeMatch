"""SQLite-backed cache for scans and provider data.

A single database file (default ``data/episodematch.db``) stores:

* a generic TTL key/value JSON cache (used by all providers),
* probed MKV scans keyed by path+size+mtime,
* TheDiscDB disc entries,
* OpenSubtitles reference subtitle metadata.

The class is thread-safe (single lock, WAL journal mode) and creates its schema
idempotently. Schema versioning uses ``PRAGMA user_version``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS cache (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        expires_at  REAL,
        created_at  REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mkv_scans (
        path        TEXT NOT NULL,
        size_bytes  INTEGER NOT NULL,
        mtime       REAL NOT NULL,
        payload     TEXT NOT NULL,
        created_at  REAL NOT NULL,
        PRIMARY KEY (path, size_bytes, mtime)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS thediscdb (
        cache_key   TEXT PRIMARY KEY,
        series      TEXT,
        payload     TEXT NOT NULL,
        fetched_at  REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS osdb_subtitles (
        subtitle_id TEXT PRIMARY KEY,
        meta        TEXT NOT NULL,
        srt_path    TEXT,
        fetched_at  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache (expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_thediscdb_series ON thediscdb (series)",
)


class CacheDatabase:
    """Thread-safe SQLite cache with JSON payload helpers."""

    def __init__(self, db_path: Path | str, *, read_only: bool = False) -> None:
        self._path = Path(db_path)
        self._read_only = read_only
        self._lock = threading.RLock()
        self._conn = self._open()
        self._migrate()

    # -- lifecycle -----------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        if not self._read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=15.0,
        )
        conn.row_factory = sqlite3.Row
        with self._lock:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _migrate(self) -> None:
        with self._lock, self._conn:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if current == _SCHEMA_VERSION:
                return
            if not self._read_only:
                for statement in _SCHEMA_STATEMENTS:
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                logger.debug("Database schema at v%d (%s)", _SCHEMA_VERSION, self._path)
            elif current != _SCHEMA_VERSION:
                logger.warning("Database %s has schema v%s (expected v%s)", self._path, current, _SCHEMA_VERSION)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> CacheDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- generic JSON cache ---------------------------------------------------

    def get_json(self, key: str) -> Any | None:
        """Return the cached JSON value, or ``None`` when absent/expired."""
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        if row["expires_at"] is not None and row["expires_at"] < now:
            self.delete(key)
            return None
        return json.loads(row["value"])

    def put_json(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        """Store a JSON-serializable value, optionally expiring after ``ttl_seconds``."""
        expires_at = (time.time() + ttl_seconds) if ttl_seconds is not None else None
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO cache (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    created_at = excluded.created_at
                """,
                (key, payload, expires_at, time.time()),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def purge_expired(self) -> int:
        """Remove expired cache rows; returns the number of deleted rows."""
        now = time.time()
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
            return cursor.rowcount

    # -- typed helpers ---------------------------------------------------------

    def get_model(self, key: str, model_type: type[BaseModel]) -> BaseModel | None:
        value = self.get_json(key)
        if value is None:
            return None
        return model_type.model_validate(value)

    def put_model(self, key: str, model: BaseModel, *, ttl_seconds: float | None = None) -> None:
        self.put_json(key, model.model_dump(mode="json"), ttl_seconds=ttl_seconds)

    # -- MKV scan cache ----------------------------------------------------------

    def get_mkv_scan(self, path: Path, size_bytes: int, mtime: float) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM mkv_scans WHERE path = ? AND size_bytes = ? AND mtime = ?",
                (str(path), size_bytes, mtime),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def put_mkv_scan(self, path: Path, size_bytes: int, mtime: float, payload: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO mkv_scans (path, size_bytes, mtime, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(path), size_bytes, mtime, json.dumps(payload, ensure_ascii=False), time.time()),
            )

    # -- TheDiscDB cache ----------------------------------------------------------

    def get_disc(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM thediscdb WHERE cache_key = ?", (cache_key,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def put_disc(self, cache_key: str, series: str, payload: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO thediscdb (cache_key, series, payload, fetched_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, series, json.dumps(payload, ensure_ascii=False), time.time()),
            )

    def find_discs_by_series(self, series_query: str) -> list[dict[str, Any]]:
        """Case-insensitive substring search over cached disc entries."""
        pattern = f"%{series_query}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM thediscdb WHERE series LIKE ? ORDER BY fetched_at DESC",
                (pattern,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # -- OpenSubtitles cache --------------------------------------------------------

    def get_osdb_subtitle(self, subtitle_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT meta, srt_path FROM osdb_subtitles WHERE subtitle_id = ?", (subtitle_id,)).fetchone()
        if row is None:
            return None
        meta = json.loads(row["meta"])
        meta["_srt_path"] = row["srt_path"]
        return meta

    def put_osdb_subtitle(self, subtitle_id: str, meta: dict[str, Any], srt_path: Path | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO osdb_subtitles (subtitle_id, meta, srt_path, fetched_at)
                VALUES (?, ?, ?, ?)
                """,
                (subtitle_id, json.dumps(meta, ensure_ascii=False), str(srt_path) if srt_path else None, time.time()),
            )

    # -- maintenance ------------------------------------------------------------------

    def vacuum(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("VACUUM")

    @property
    def path(self) -> Path:
        return self._path
