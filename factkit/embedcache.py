"""Persistent, best-effort cache for embedding vectors.

The cache is a separate SQLite file (never the fact store), so a corrupt or
locked cache can be deleted without touching facts. Every read path is soft:
:meth:`EmbedCache.get` returns ``None`` on any storage problem and
:func:`cached_embed` falls back to the real embedder, because recall degrading
to BM25 is acceptable while a failed turn is not.

Storage format
--------------
One table, one row per ``(model, normalized text)`` pair::

    embedding_cache(key TEXT PRIMARY KEY, model TEXT, dim INTEGER,
                    vector BLOB, created_at TEXT, hits INTEGER)

``vector`` is a raw little-endian ``float32`` array — exactly
``array.array("f", vec).tobytes()``, so 768 dims occupy 3 072 bytes and a
5000-entry cache stays around 15 MB. ``dim`` is the number of floats and is
checked against the BLOB length on read, which is how truncated or foreign
BLOBs turn into misses instead of nonsense vectors. ``created_at`` is a UTC
ISO-8601 string with second precision; pruning and ``oldest`` compare it
lexicographically, which is safe because every row is written in that format.

Keys are ``sha256(model + "\\x00" + normalized_text)``. ``embed.py`` sends the
text to the endpoint verbatim, so normalization only strips and collapses
whitespace (the same text embedded twice should not pay twice); case is
**not** folded, matching the embedder, which treats "Atlas Server" and
"atlas server" as different inputs.

The file is created with mode 0600, and the parent directory with 0700, so
cached vectors never become group- or world-readable.
"""
from __future__ import annotations

import array
import hashlib
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 5000
DEFAULT_TTL_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    key        TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
)
"""

INDEX = "CREATE INDEX IF NOT EXISTS embedding_cache_created ON embedding_cache(created_at)"


# --------------------------------------------------------------- key / text
def normalize_text(text: str) -> str:
    """Strip and collapse runs of whitespace; no case folding.

    ``embed.py`` forwards text to the endpoint unchanged, so this is only a
    cache-key normalization: " Atlas\\n Server " and "Atlas Server" are the
    same input for retrieval purposes, while letter case is preserved.
    """
    return " ".join(str(text or "").split())


def cache_key(text: str, model: str) -> str:
    """Stable cache key: sha256 over model and whitespace-normalized text."""
    payload = f"{model}\x00{normalize_text(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pack(vector: Sequence[float]) -> tuple[bytes, int]:
    """Return ``(blob, dim)`` as float32, or raise ``ValueError``."""
    try:
        values = [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector must be a sequence of numbers") from exc
    if not values:
        raise ValueError("vector must not be empty")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("vector must contain only finite numbers")
    try:
        packed = array.array("f", values)
    except (OverflowError, TypeError) as exc:
        raise ValueError("vector is not representable as float32") from exc
    if not all(math.isfinite(v) for v in packed):
        raise ValueError("vector overflows float32")
    return packed.tobytes(), len(packed)


def _unpack(blob, dim, model: str, expected_model: str) -> list[float] | None:
    """Decode a stored vector, or ``None`` when the row is unusable."""
    if model != expected_model:
        return None
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    raw = bytes(blob)
    if not isinstance(dim, int) or dim <= 0 or len(raw) != dim * 4:
        return None
    values = array.array("f")
    values.frombytes(raw)
    if len(values) != dim or not all(math.isfinite(v) for v in values):
        return None
    return [float(v) for v in values]


# ------------------------------------------------------------------- cache
class EmbedCache:
    """SQLite-backed embedding cache, created on first use.

    Parameters are deliberately dumb: ``max_entries`` and ``ttl_days`` are only
    enforced when :meth:`prune` runs, so a long-lived process decides when to
    pay for maintenance. The connection is lazy, thread-safe and WAL-mode so a
    reader never blocks the process that is writing new vectors.
    """

    def __init__(self, path: str | Path, *, max_entries: int = DEFAULT_MAX_ENTRIES,
                 ttl_days: int = DEFAULT_TTL_DAYS):
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        if ttl_days < 0:
            raise ValueError("ttl_days must be >= 0")
        self.path = Path(path).expanduser()
        self.max_entries = int(max_entries)
        self.ttl_days = int(ttl_days)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # ---------------------------------------------------------- connection
    def _ensure_file(self) -> bool:
        """Create the file 0600 before SQLite can apply the umask."""
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        os.close(fd)
        return True

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            created = self._ensure_file()
            conn = sqlite3.connect(str(self.path), timeout=5.0, check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute(SCHEMA)
                conn.execute(INDEX)
                conn.commit()
            except Exception:
                conn.close()
                raise
            if created:
                try:
                    os.chmod(self.path, 0o600)
                except OSError:  # pragma: no cover - exotic filesystems
                    pass
            self._conn = conn
            return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "EmbedCache":
        self._connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------------------------------------------------------------- read
    def get(self, text: str, model: str) -> list[float] | None:
        """Return the cached vector, or ``None`` on a miss or any cache fault."""
        key = cache_key(text, model)
        try:
            with self._lock:
                conn = self._connect()
                row = conn.execute(
                    "SELECT model, dim, vector FROM embedding_cache WHERE key = ?",
                    (key,)).fetchone()
                if row is None:
                    return None
                vector = _unpack(row[2], row[1], row[0], model)
                if vector is None:
                    # corrupt/foreign row: drop it so the next put can heal it
                    conn.execute("DELETE FROM embedding_cache WHERE key = ?", (key,))
                    conn.commit()
                    return None
                conn.execute(
                    "UPDATE embedding_cache SET hits = hits + 1 WHERE key = ?", (key,))
                conn.commit()
                return vector
        except (sqlite3.Error, OSError, ValueError) as exc:
            logger.debug("embedding cache read failed: %s", exc)
            return None

    # --------------------------------------------------------------- write
    def put(self, text: str, model: str, vector: Sequence[float]) -> str:
        """Store ``vector`` under ``(model, text)``; return the cache key.

        Raises ``ValueError`` for an unusable vector and propagates storage
        errors; :func:`cached_embed` is the wrapper that swallows those.
        """
        blob, dim = _pack(vector)
        key = cache_key(text, model)
        with self._lock:
            conn = self._connect()
            with conn:  # commit on success, roll back on error
                conn.execute(
                    "INSERT INTO embedding_cache(key, model, dim, vector, created_at, hits) "
                    "VALUES (?,?,?,?,?,0) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "model = excluded.model, dim = excluded.dim, "
                    "vector = excluded.vector, created_at = excluded.created_at",
                    (key, model, dim, blob, _now()))
        return key

    # --------------------------------------------------------------- stats
    def stats(self) -> dict:
        """``{"entries": int, "hits": int, "bytes": int, "oldest": str | None}``.

        ``bytes`` is the payload size of the stored BLOBs (768 dims = 3 072 B
        per entry), not the on-disk file size, which includes SQLite overhead.
        """
        try:
            with self._lock:
                conn = self._connect()
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(hits), 0), "
                    "COALESCE(SUM(LENGTH(vector)), 0), MIN(created_at) "
                    "FROM embedding_cache").fetchone()
        except (sqlite3.Error, OSError) as exc:
            logger.debug("embedding cache stats failed: %s", exc)
            return {"entries": 0, "hits": 0, "bytes": 0, "oldest": None}
        return {"entries": int(row[0] or 0), "hits": int(row[1] or 0),
                "bytes": int(row[2] or 0), "oldest": row[3]}

    # --------------------------------------------------------------- prune
    def prune(self) -> dict:
        """Drop expired rows, then the oldest rows above ``max_entries``.

        Returns ``{"expired", "overflow", "removed", "entries"}``. TTL is
        skipped when ``ttl_days`` is 0 and the cap when ``max_entries`` is 0
        (which then means "keep nothing"). Storage errors are logged and the
        counts come back as zeros.
        """
        expired = overflow = 0
        try:
            with self._lock:
                conn = self._connect()
                with conn:
                    if self.ttl_days > 0:
                        cutoff = (datetime.now(timezone.utc)
                                  - timedelta(days=self.ttl_days)
                                  ).isoformat(timespec="seconds")
                        cur = conn.execute(
                            "DELETE FROM embedding_cache WHERE created_at < ?",
                            (cutoff,))
                        expired = max(cur.rowcount, 0)
                    total = int(conn.execute(
                        "SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
                    excess = total - self.max_entries
                    if excess > 0:
                        cur = conn.execute(
                            "DELETE FROM embedding_cache WHERE key IN ("
                            "  SELECT key FROM embedding_cache"
                            "  ORDER BY created_at ASC, rowid ASC LIMIT ?)",
                            (excess,))
                        overflow = max(cur.rowcount, 0)
                    remaining = int(conn.execute(
                        "SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
        except (sqlite3.Error, OSError) as exc:
            logger.warning("embedding cache prune failed: %s", exc)
            return {"expired": 0, "overflow": 0, "removed": 0, "entries": 0}
        return {"expired": expired, "overflow": overflow,
                "removed": expired + overflow, "entries": remaining}


# -------------------------------------------------------------- read-through
def cached_embed(cache: EmbedCache | None, embed_fn: Callable[[str], list[float]],
                 text: str, model: str) -> list[float]:
    """Read through ``cache``; on any cache fault, just call ``embed_fn``.

    ``embed_fn`` failures are the caller's business and propagate unchanged.
    Pass ``cache=None`` to bypass caching entirely.
    """
    if cache is not None:
        try:
            hit = cache.get(text, model)
        except Exception as exc:  # noqa: BLE001 - a cache must never break a turn
            logger.debug("embedding cache lookup skipped: %s", exc)
            hit = None
        if hit is not None:
            return hit
    vector = embed_fn(text)
    if cache is not None:
        try:
            cache.put(text, model, vector)
        except Exception as exc:  # noqa: BLE001 - write-back is best effort
            logger.debug("embedding cache write skipped: %s", exc)
    return vector
