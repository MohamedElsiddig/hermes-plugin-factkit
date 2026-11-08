"""Maintenance operations over a live holographic store.

All writes go through the holographic plugin's own ``MemoryStore`` API
(``add_fact_status`` / ``update_fact`` / ``replace_fact`` / ``remove_fact``), so
FTS, entity links, HRR vectors, trust scores and bank invalidation stay
consistent — the same path the agent's ``fact_store`` tool uses.

Detection is read-only. Every operation supports ``dry_run`` and records what it
did in the ``audit`` table, written by factkit with actor ``factkit``.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from hermes_constants import get_hermes_home  # type: ignore
    _HERMES_HOME = get_hermes_home()
except Exception:  # pragma: no cover - outside Hermes
    _HERMES_HOME = Path.home() / ".hermes"

PLUGIN_DIR = _HERMES_HOME / "plugins" / "holographic"


# --------------------------------------------------------------------- store
class ReadOnlyStore:
    """Read-only view of a store, for callers that cannot take a write lock.

    The gateway's plugin host can refuse a write-open of the live file, so the
    recall path opens it read-only. A WAL database still needs its -shm/-wal
    side files, which may also be off limits; in that case the database and its
    side files are copied to a private snapshot and read from there. Snapshots
    are cached by (mtime, size) so a turn does not re-copy the file.
    """

    _snapshots: dict = {}

    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).expanduser())
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = self._open_direct(self.db_path)
        except sqlite3.OperationalError:
            self._conn = self._open_snapshot(self.db_path)
        self._conn.row_factory = sqlite3.Row

    @staticmethod
    def _open_direct(path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        conn.execute("SELECT COUNT(*) FROM facts").fetchone()   # prove it reads
        return conn

    @staticmethod
    def _snapshot_key(path: str) -> tuple:
        """Cache key: main db plus WAL, so uncheckpointed writes invalidate it."""
        source = Path(path)
        main = source.stat()
        key = (str(source), main.st_mtime_ns, main.st_size)
        wal = source.with_name(source.name + "-wal")
        try:
            info = wal.stat()
            key += (info.st_mtime_ns, info.st_size)
        except OSError:
            key += (None, None)
        return key

    @staticmethod
    def _backup(source: Path, target: Path) -> None:
        """Copy the live database with SQLite's backup API, WAL contents included."""
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(str(target))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

    @classmethod
    def _open_snapshot(cls, path: str, *, refresh: bool = False) -> sqlite3.Connection:
        import shutil
        import tempfile
        import time

        source = Path(path)
        stamp = cls._snapshot_key(path)
        if refresh:
            cls._snapshots.pop(stamp, None)
        cached = cls._snapshots.get(stamp)
        if cached:
            return sqlite3.connect(cached, check_same_thread=False)

        target_dir = Path(tempfile.mkdtemp(prefix="factkit-ro-"))
        target = target_dir / source.name
        try:
            cls._backup(source, target)
            conn = sqlite3.connect(str(target), check_same_thread=False)
            conn.execute("SELECT COUNT(*) FROM facts").fetchone()
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

        # keep the cache small; snapshots are per-process and disposable
        for old, _ in list(cls._snapshots.items())[:-2]:
            cls._snapshots.pop(old, None)
        for stale in target_dir.parent.glob("factkit-ro-*"):
            try:
                if stale != target_dir and time.time() - stale.stat().st_mtime > 3600:
                    shutil.rmtree(stale, ignore_errors=True)
            except OSError:
                pass
        cls._snapshots[stamp] = str(target)
        return conn

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise sqlite3.ProgrammingError("ReadOnlyStore is closed")
        return self._conn

    def read(self, sql: str, params=()) -> list[dict]:
        try:
            return [dict(r) for r in self._connection().execute(sql, params)]
        except sqlite3.OperationalError:
            # opening can succeed and the first read still fail when the WAL
            # side files are off limits; retry once from a fresh snapshot
            self.close()
            self._conn = self._open_snapshot(self.db_path, refresh=True)
            self._conn.row_factory = sqlite3.Row
            return [dict(r) for r in self._conn.execute(sql, params)]

    def count_facts(self) -> int:
        row = self._connection().execute("SELECT COUNT(*) AS n FROM facts").fetchone()
        return int(row["n"] if row else 0)

    def close(self) -> None:
        """Close the open connection; safe to call more than once."""
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


# Provider-compatible schema subset for the standalone store. It deliberately
# omits the maintenance tables (tombstones, audit, proposals): those are created
# on demand by ``factkit.maintenance.ensure_tombstones``.
_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class StandaloneStore:
    """Writable store used when the holographic provider is not installed.

    Implements the subset of the provider's ``MemoryStore`` contract that
    factkit's maintenance and CLI paths use, on the same schema shape
    (AUTOINCREMENT fact ids, external-content FTS, entity tables). It is
    stdlib-only: nothing here imports the provider.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - filesystem without WAL
            pass
        conn.executescript(_STORE_SCHEMA)
        self._conn: sqlite3.Connection | None = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise sqlite3.ProgrammingError("StandaloneStore is closed")
        return self._conn

    def read(self, sql: str, params=()) -> list:
        return self._connection().execute(sql, params).fetchall()

    def _one(self, sql: str, params=()):
        return self._connection().execute(sql, params).fetchone()

    def _write(self, sql: str, params=()) -> sqlite3.Cursor:
        """One write statement: autocommitted on its own, or part of an enclosing _tx()."""
        return self._connection().execute(sql, params)

    @contextlib.contextmanager
    def _tx(self):
        """One write transaction; nested calls join the outer one."""
        conn = self._connection()
        if conn.in_transaction:
            yield
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def count_facts(self) -> int:
        row = self._connection().execute("SELECT COUNT(*) FROM facts").fetchone()
        return int(row[0]) if row else 0

    def _entity(self, name: str, entity_type: str) -> int:
        row = self._connection().execute(
            "SELECT entity_id FROM entities WHERE name = ?", (name,)).fetchone()
        if row is not None:
            if entity_type == "declared":
                self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?",
                            (row[0],))
            return int(row[0])
        cur = self._write("INSERT INTO entities(name, entity_type) VALUES (?, ?)",
                          (name, entity_type))
        return int(cur.lastrowid)

    def _link(self, fact_id: int, content: str, declared=()) -> None:
        """Link declared names, else conservative multi-word extracted candidates."""
        names = [(str(name).strip(), "declared") for name in declared or ()
                 if isinstance(name, str) and str(name).strip()]
        if not names:
            try:
                from . import entities as entmod
                names = [(name, "extracted") for name in entmod.candidates(content)
                         if len(str(name).split()) > 1]
            except Exception:  # noqa: BLE001 - extraction is optional
                names = []
        for name, entity_type in names:
            entity_id = self._entity(name, entity_type)
            self._write("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?,?)",
                        (fact_id, entity_id))

    def add_fact_status(self, content: str, category: str = "general", tags: str = "", *,
                        trust: float | None = None, entities=()) -> tuple[int, bool]:
        """(fact_id, created); duplicate content reuses the existing row."""
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        with self._tx():
            try:
                cur = self._write(
                    "INSERT INTO facts(content, category, tags, trust_score) VALUES (?,?,?,?)",
                    (content, category, tags, 0.5 if trust is None else trust))
                fact_id = int(cur.lastrowid)
            except sqlite3.IntegrityError:
                return int(self._one("SELECT fact_id FROM facts WHERE content = ?",
                                     (content,))["fact_id"]), False
            self._link(fact_id, content, entities)
            return fact_id, True

    def replace_fact(self, old_id: int, content: str, category: str = "general", tags: str = "",
                     *, entities=()) -> tuple[int, bool]:
        """Supersede a fact in one transaction; KeyError when it does not exist."""
        with self._tx():
            old = self._one("SELECT fact_id, content FROM facts WHERE fact_id = ?", (old_id,))
            if old is None:
                raise KeyError(f"fact_id {old_id} not found")
            if old["content"] == (content or "").strip():
                return int(old_id), False
            fact_id, created = self.add_fact_status(
                content, category=category, tags=tags, entities=entities)
            self.remove_fact(old_id)
            return fact_id, created

    def update_fact(self, fact_id: int, content: str | None = None,
                    trust_delta: float | None = None, tags: str | None = None,
                    category: str | None = None, *, entities=(), touch: bool = True) -> bool:
        """Partially update a fact; returns True when the row existed."""
        if content is not None:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
        with self._tx():
            row = self._one("SELECT fact_id, trust_score, category FROM facts WHERE fact_id = ?",
                            (fact_id,))
            if row is None:
                return False
            changes = {col: val for col, val in {
                "content": content, "tags": tags, "category": category,
                "trust_score": (min(1.0, max(0.0, float(row["trust_score"] or 0.0) + trust_delta))
                                if trust_delta is not None else None),
            }.items() if val is not None}
            assignments = ", ".join((["updated_at = CURRENT_TIMESTAMP"] if touch else [])
                                    + [f"{col} = ?" for col in changes])
            if not assignments:
                return True
            try:
                self._write(f"UPDATE facts SET {assignments} WHERE fact_id = ?",
                            [*changes.values(), fact_id])
            except sqlite3.IntegrityError:
                other = self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))
                owner = other["fact_id"] if other else "?"
                raise ValueError(f"content already belongs to fact {owner}") from None
            if content is not None:
                kept = [r[0] for r in self._connection().execute(
                    "SELECT e.name FROM fact_entities fe JOIN entities e "
                    "ON e.entity_id = fe.entity_id "
                    "WHERE fe.fact_id = ? AND e.entity_type = 'declared'", (fact_id,))]
                self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
                self._link(fact_id, content, [*kept, *(entities or ())])
            elif entities:
                self._link(fact_id, "", entities)
            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links; returns True when the row existed."""
        with self._tx():
            row = self._one("SELECT fact_id FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
            self._write("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            return True

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def _provider_db_path(db_path: str | Path | None = None) -> str:
    """Path of the provider's live store (or the explicit path the caller gave)."""
    if db_path:
        return str(Path(db_path).expanduser())
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        return str(get_hermes_home() / "memory_store.db")
    except Exception:  # pragma: no cover - outside Hermes
        return str(_HERMES_HOME / "memory_store.db")


def _factkit_db_path() -> Path:
    """factkit's own writable database, used when no provider is installed."""
    p = _HERMES_HOME / "plugins" / "factkit" / "facts.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_store_file(p: Path) -> "StandaloneStore":
    """Create the database with a usable schema, not a bare 0-byte file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    return StandaloneStore(str(p))


def load_plugin_store(db_path: str | Path | None = None):
    """Return a read-only handle to the provider store.

    Never imports the provider and never raises at import time: a missing file
    is bootstrapped with the provider-compatible schema so the read-only probe
    has a ``facts`` table to read.
    """
    path = _provider_db_path(db_path)
    p = Path(path)
    if not p.exists():
        _ensure_store_file(p).close()
    return ReadOnlyStore(path)


def _provider_memory_store():
    """The provider's writable ``MemoryStore`` class, or None when unavailable.

    This is the only place allowed to register ``holographic`` in
    ``sys.modules``; the read-only path never does.
    """
    # 1) direct import (Hermes gateway)
    try:
        from holographic.store import MemoryStore  # type: ignore
        return MemoryStore
    except Exception:
        pass
    # 2) CLI with the provider plugin installed but not on sys.path
    try:
        if "holographic" not in sys.modules and (PLUGIN_DIR / "store.py").exists():
            pkg = types.ModuleType("holographic")
            pkg.__path__ = [str(PLUGIN_DIR)]
            pkg.__package__ = "holographic"
            sys.modules["holographic"] = pkg
            try:
                from . import standalone  # type: ignore
                standalone.ensure()
            except Exception:
                pass
    except Exception:
        pass
    try:
        from holographic.store import MemoryStore  # type: ignore
        return MemoryStore
    except Exception:
        return None


def get_memory_store(db_path: str | Path | None = None):
    """Get a writable store: the provider's when installed, else a standalone one."""
    path = _provider_db_path(db_path)
    cls = _provider_memory_store()
    if cls is not None:
        return cls(path)
    if db_path is None:
        path = str(_factkit_db_path())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return StandaloneStore(path)


_FACTKIT_VECTORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id INTEGER,
    model TEXT,
    dims INTEGER,
    vec BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, model)
);
CREATE INDEX IF NOT EXISTS idx_factkit_emb_model ON fact_embeddings(model);
"""


def _factkit_vectors_db_path() -> Path:
    try:
        from plugins import plugin_storage  # type: ignore
        try:
            conn = plugin_storage.plugin_db("factkit")
            dbp = Path(conn.execute("PRAGMA database_list").fetchone()[2])
            conn.close()
            if dbp and str(dbp) != ":memory:":
                return dbp
        except Exception:
            pass
        try:
            return plugin_storage.plugin_data_dir("factkit") / "facts.db"
        except Exception:
            pass
    except Exception:
        pass
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        base = get_hermes_home() / "plugins" / "factkit"
    except Exception:
        base = Path.home() / ".hermes" / "plugins" / "factkit"
    base.mkdir(parents=True, exist_ok=True)
    return base / "facts.db"


def _factkit_vectors_conn(readonly: bool = False):
    dbp = _factkit_vectors_db_path()
    if readonly:
        try:
            conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True, check_same_thread=False)
        except Exception:
            dbp.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(dbp), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_FACTKIT_VECTORS_SCHEMA)
            conn.close()
            conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(dbp), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_FACTKIT_VECTORS_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn




def _vectors_conn(store, *, readonly: bool = False):
    """Connection for fact_embeddings: Store's own DB, else factkit's plugin DB.

    Only a ``factkit.store.Store`` keeps ``fact_embeddings`` in its own file.
    Provider handles (``holo.ReadOnlyStore``, the provider's ``MemoryStore``)
    always use factkit's plugin DB: provider vectors are copied there once by
    ``_maybe_migrate_vectors`` instead of being read in place.
    """
    from .store import Store as FactkitStore

    if isinstance(store, FactkitStore):
        if not getattr(store, "readonly", False):
            try:
                store.db.executescript(_FACTKIT_VECTORS_SCHEMA)
            except Exception:
                pass
        return store.db, False

    conn = _factkit_vectors_conn(readonly=readonly)
    return conn, True


def _is_provider_store(store) -> bool:
    """True for the provider's handles (read-only view, standalone, MemoryStore).

    Those must never supply vectors directly: factkit keeps them in its own DB.
    Unknown duck-typed adapters keep the legacy read-through fallback.
    """
    if isinstance(store, (ReadOnlyStore, StandaloneStore)):
        return True
    module = getattr(type(store), "__module__", "") or ""
    return module == "holographic" or module.startswith("holographic.")

def migrate_vectors_from_provider(provider_db_path: str | Path | None = None) -> dict:
    """One-time migration: move vectors from provider's fact_embeddings to factkit's own DB."""
    try:
        target = _factkit_vectors_conn(readonly=False)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    try:
        # Find provider DB
        provider_path = None
        if provider_db_path:
            provider_path = Path(provider_db_path).expanduser()
        else:
            try:
                from hermes_constants import get_hermes_home  # type: ignore
                provider_path = get_hermes_home() / "memory_store.db"
            except Exception:
                provider_path = Path.home() / ".hermes" / "memory_store.db"
        if not provider_path.exists():
            return {"ok": True, "migrated": 0, "reason": "no provider db"}
        try:
            src = sqlite3.connect(f"file:{provider_path}?mode=ro", uri=True)
            src.row_factory = sqlite3.Row
        except Exception:
            return {"ok": True, "migrated": 0, "reason": "cannot open provider"}
        try:
            cols = {r[1] for r in src.execute("PRAGMA table_info(fact_embeddings)")}
        except Exception:
            cols = set()
        if not cols or "fact_id" not in cols or "vec" not in cols:
            rows = []
        else:
            model_sql = "model" if "model" in cols else "NULL AS model"
            dims_sql = "dims" if "dims" in cols else "NULL AS dims"
            created_sql = "created_at" if "created_at" in cols else "NULL AS created_at"
            try:
                rows = src.execute(
                    f"SELECT fact_id, {model_sql}, {dims_sql}, vec, {created_sql} "
                    "FROM fact_embeddings"
                ).fetchall()
            except Exception:
                rows = []
        moved = 0
        for r in rows:
            try:
                target.execute(
                    "INSERT OR REPLACE INTO fact_embeddings(fact_id, model, dims, vec, created_at) VALUES (?,?,?,?,?)",
                    (r["fact_id"], r["model"], r["dims"], r["vec"], r["created_at"]),
                )
                moved += 1
            except Exception:
                continue
        target.commit()
        target.close()
        src.close()
        return {"ok": True, "migrated": moved}
    except Exception as exc:
        try:
            target.close()
        except Exception:
            pass
        return {"ok": False, "error": str(exc)}


_VECTOR_MIGRATION_ATTEMPTED: set[str] = set()


def _factkit_vectors_row_count() -> int:
    """Rows in factkit's own vectors DB; 0 on any storage fault."""
    try:
        conn = _factkit_vectors_conn(readonly=False)
    except Exception:
        return 0
    try:
        return int(conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0])
    except Exception:
        return 0
    finally:
        conn.close()


def _maybe_migrate_vectors(store) -> None:
    """Copy provider vectors into factkit's own DB once, when it is still empty.

    Best effort: a missing provider, an unreadable file or a concurrent writer
    only logs and leaves recall on the legacy path. Never raises.
    """
    from .store import Store as FactkitStore

    if isinstance(store, FactkitStore) or not _is_provider_store(store):
        return
    provider = getattr(store, "db_path", None)
    if not provider:
        return
    try:
        provider_path = Path(provider).expanduser()
    except Exception:
        return
    key = str(provider_path)
    if key in _VECTOR_MIGRATION_ATTEMPTED:
        return
    try:
        if provider_path.resolve() == _factkit_vectors_db_path().resolve():
            _VECTOR_MIGRATION_ATTEMPTED.add(key)
            return
    except Exception:
        pass
    if not provider_path.exists():
        return
    if _factkit_vectors_row_count():
        _VECTOR_MIGRATION_ATTEMPTED.add(key)
        return
    try:
        result = migrate_vectors_from_provider(key)
    except Exception as exc:  # noqa: BLE001 - migration must never break recall
        logger.warning("factkit: vector migration from %s failed: %s", key, exc)
        return
    if not result.get("ok"):
        logger.warning("factkit: vector migration from %s failed: %s",
                       key, result.get("error") or result.get("reason"))
        return
    _VECTOR_MIGRATION_ATTEMPTED.add(key)
    if result.get("migrated"):
        logger.info("factkit: migrated %s vectors from provider DB %s",
                    result["migrated"], key)


def _ro_conn(path: str | Path):
    conn = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------ conflict scan
STOP = {
    "the", "and", "for", "with", "that", "this", "from", "has", "have", "was",
    "are", "its", "his", "her", "their", "not", "but", "you", "what", "when",
    "which", "how", "there", "these", "those", "into", "about", "will", "can",
    "full", "specs", "more", "info", "details", "note", "notes", "file",
}

# Attribute = (name, value). Only equal attribute names can contradict: two
# different paths are not a disagreement, two different RAM sizes are.
ATTR_PATTERNS = [
    # 16 GB RAM / 512 gb ssd / 6 cores
    ("unit", re.compile(r"\b(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s+(ram|memory|ssd|storage|disk|swap)\b")),
    ("cores", re.compile(r"\b(\d+)\s*(?:cores?|threads?)\b")),
    # 2.4 GHz
    ("clock", re.compile(r"\b(\d+(?:\.\d+)?)\s*ghz\b")),
    # port 8080 / port=8080 / :8080
    ("port", re.compile(r"\bport\s*[=:]?\s*(\d{2,5}(?:\s*-\s*\d{2,5})?)")),
    ("port", re.compile(r"(?<![\d.]):(\d{2,5})\b")),
    # 192.0.2.5 (documentation range)
    ("ip", re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")),
    # caddy 2.7.6 / python3.12 / v1.2.3
    ("version", re.compile(r"\b([a-z][a-z0-9_-]{2,})\s+v?(\d+\.\d+(?:\.\d+)?)\b")),
]

# Paths, dates and generic locations are context, not attributes.
VOLATILE = ("port", "version", "ip", "clock", "unit", "cores")


def _tokens(text: str) -> set[str]:
    t = set(re.findall(r"[a-z0-9_-]+", (text or "").lower()))
    return {x for x in t if len(x) > 2 and x not in STOP and not x.isdigit()}


def _attributes(text: str) -> dict[str, set[str]]:
    low = (text or "").lower()
    out: dict[str, set[str]] = {}

    def put(name: str, value: str) -> None:
        out.setdefault(name, set()).add(value)

    for name, rx in ATTR_PATTERNS:
        for m in rx.finditer(low):
            if name == "unit":
                put(m.group(3), f"{m.group(1)}{m.group(2)}")
            elif name == "version":
                put(m.group(1), m.group(2))
            else:
                put(name, m.group(1))
    # a bare "16 GB" with no noun is storage/memory only when another fact
    # names the same attribute; do not guess here.
    return out


GENERIC_ATTRS = {"port", "ip", "clock", "cores"}


def _value_conflict(a: dict, b: dict, shared=()) -> list[tuple[str, set, set]]:
    aa, ab = _attributes(a["content"]), _attributes(b["content"])
    out = []
    for k in set(aa) & set(ab):
        # A bare port or IP is only a conflict when both facts are about the
        # same declared subject; otherwise two services may legitimately differ.
        if k in GENERIC_ATTRS and not shared:
            continue
        # Values are compared canonically, so 16GB vs 16384MB or 3.6GHz vs
        # 3600MHz are agreement, not a conflict.
        differing = [v for v in aa[k] if not any(_same(k, v, w) for w in ab[k])]
        differing_b = [w for w in ab[k] if not any(_same(k, w, v) for v in aa[k])]
        if not differing and not differing_b:
            continue
        out.append((k, aa[k], ab[k]))
    return out


def _same(attr: str, a: str, b: str) -> bool:
    """Equality under unit normalisation.

    Ports and IPs are identity values: on the same declared subject, 7000 and
    7001 are a disagreement, not two acceptable truths. Capacities, clocks and
    versions use ``units.conflicts`` so 16GB == 16384MB.
    """
    try:
        from .units import canonical, conflicts
        if attr in ("port", "ip"):
            return canonical(attr, a) == canonical(attr, b)
        return not conflicts(attr, a, b)
    except Exception:  # noqa: BLE001 - never let unit parsing break detection
        return str(a).strip().lower() == str(b).strip().lower()


def find_conflicts(db_path: str | Path, overlap_min: float = 0.35,
                   entity_min: float = 0.15) -> list[dict]:
    """Same declared subject + same attribute + different value = conflict.

    Two gates, both required: the facts must share a *declared* entity (or be
    near-duplicates by content), and they must disagree on the same named
    attribute. Similarity alone never creates a conflict.
    """
    conn = _ro_conn(db_path)
    try:
        facts = [dict(r) for r in conn.execute(
            "SELECT fact_id, content, category, trust_score, created_at, updated_at FROM facts")]
        links: dict[int, set[str]] = {}
        try:
            erows = conn.execute(
                "SELECT fe.fact_id, e.name FROM fact_entities fe "
                "JOIN entities e ON e.entity_id = fe.entity_id "
                "WHERE e.entity_type = 'declared'").fetchall()
        except sqlite3.OperationalError:
            erows = conn.execute(
                "SELECT fe.fact_id, e.name FROM fact_entities fe "
                "JOIN entities e ON e.entity_id = fe.entity_id "
                "WHERE e.kind = 'declared'").fetchall()
        for r in erows:
            links.setdefault(int(r["fact_id"]), set()).add(r["name"].lower())
    finally:
        conn.close()

    toks = {f["fact_id"]: _tokens(f["content"]) for f in facts}
    out = []
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            ta, tb = toks[a["fact_id"]], toks[b["fact_id"]]
            jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
            shared = set(links.get(a["fact_id"], ())) & set(links.get(b["fact_id"], ()))
            if not (jac >= overlap_min or (shared and jac >= entity_min)):
                continue
            newer, older = (a, b) if (str(a["updated_at"]), a["fact_id"]) >= (str(b["updated_at"]), b["fact_id"]) else (b, a)
            diffs = _value_conflict(newer, older, shared)
            if not diffs:
                continue
            out.append({
                "keep": newer["fact_id"], "retire": older["fact_id"],
                "keep_text": newer["content"], "retire_text": older["content"],
                "keep_updated": newer["updated_at"], "retire_updated": older["updated_at"],
                "shared_entities": sorted(shared), "jaccard": round(jac, 3),
                "diffs": [{"slot": s_, "keep": sorted(k), "retire": sorted(r)} for s_, k, r in diffs],
            })
    return out


# ------------------------------------------------------------------ apply
def apply_conflict(store, pair: dict, *, dry_run: bool = True, batch_id: str | None = None) -> dict:
    """Retire the older fact, keeping the newer text as the single truth."""
    keep_id, retire_id = int(pair["keep"]), int(pair["retire"])
    keep = store._one("SELECT fact_id, content, category, tags FROM facts WHERE fact_id = ?", (keep_id,))
    retire = store._one("SELECT fact_id, content FROM facts WHERE fact_id = ?", (retire_id,))
    if keep is None or retire is None:
        return {"ok": False, "reason": "fact disappeared"}
    if keep["content"] == retire["content"]:
        return {"ok": False, "reason": "identical content"}

    if dry_run:
        return {"ok": True, "dry_run": True, "keep": keep_id, "retire": retire_id,
                "action": "update keep + remove retire"}

    # 1. keep the authoritative text (identical => already correct, no touch)
    store.update_fact(keep_id, content=keep["content"], category=keep["category"],
                      tags=keep["tags"], touch=False)
    # 2. tombstone first, so the retirement is undoable even if step 3 fails
    from . import maintenance
    maintenance.ensure_tombstones(store)
    maintenance.tombstone_fact(store, retire_id, reason="conflict", batch_id=batch_id)
    # 3. retire the stale one through the plugin's own delete path
    store.remove_fact(retire_id)
    # 3. provenance
    with store._tx():
        store._write(
            "INSERT INTO audit(action, fact_id, actor, reason, before, after) VALUES (?,?,?,?,?,?)",
            ("conflict-apply", keep_id, "factkit",
             f"retired #{retire_id} on slot {','.join(d['slot'] for d in pair['diffs'])}",
             retire["content"], keep["content"]))
    return {"ok": True, "keep": keep_id, "retire": retire_id, "removed": True}


# --------------------------------------------------------------- TTL flags
def find_stale(db_path: str | Path, days: int = 90, slot_days: int = 180) -> list[dict]:
    """Flag old facts, and older facts that hold volatile (rot-prone) slots."""
    conn = _ro_conn(db_path)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT fact_id, content, category, trust_score, created_at, updated_at FROM facts")]
    finally:
        conn.close()
    out = []
    for r in rows:
        try:
            from datetime import datetime as _dt
            age = (_dt.now() - _dt.fromisoformat(str(r["updated_at"]))).days
        except ValueError:
            continue
        slots = _attributes(r["content"])
        volatile = [name for name in VOLATILE if name in slots]
        if age >= days:
            out.append({"fact_id": r["fact_id"], "age_days": age, "reason": "age",
                        "slots": volatile, "content": r["content"]})
        elif volatile and age >= slot_days:
            out.append({"fact_id": r["fact_id"], "age_days": age, "reason": "stale-slot",
                        "slots": volatile, "content": r["content"]})
    return sorted(out, key=lambda x: -x["age_days"])


def touch_fact(store, fact_id: int, *, dry_run: bool = True, reason: str = "reviewed, still correct") -> dict:
    """Confirm a fact is still true: bump updated_at through the store API."""
    if dry_run:
        return {"ok": True, "dry_run": True, "fact_id": fact_id}
    ok = store.update_fact(int(fact_id), content=None, touch=True)
    with store._tx():
        store._write("INSERT INTO audit(action, fact_id, actor, reason) VALUES (?,?,?,?)",
                     ("review-touch", int(fact_id), "factkit", reason))
    return {"ok": bool(ok), "fact_id": int(fact_id)}


# ------------------------------------------------------------------ merge
def merge_duplicates(db_path: str | Path, threshold: float = 0.92) -> list[dict]:
    """Near-identical facts: keep the highest trust (then newest), retire the rest."""
    conn = _ro_conn(db_path)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT fact_id, content, trust_score, updated_at FROM facts")]
    finally:
        conn.close()
    from difflib import SequenceMatcher
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            ratio = SequenceMatcher(None, a["content"].lower(), b["content"].lower()).ratio()
            if ratio < threshold:
                continue
            # Same sentence with a different number is not a duplicate: "filler
            # fact 1" and "filler fact 2" score ~0.95 character similarity but
            # describe different things. Numeric identity is part of the claim.
            nums_a = set(re.findall(r"\d+(?:\.\d+)?", a["content"]))
            nums_b = set(re.findall(r"\d+(?:\.\d+)?", b["content"]))
            if nums_a != nums_b:
                continue
            keep, drop = sorted([a, b], key=lambda r: (r["trust_score"], r["updated_at"]), reverse=True)
            out.append({"keep": keep["fact_id"], "retire": drop["fact_id"],
                        "ratio": round(ratio, 3), "keep_text": keep["content"],
                        "retire_text": drop["content"]})
    return out


def apply_merge(store, pair: dict, *, dry_run: bool = True, batch_id: str | None = None) -> dict:
    keep_id, retire_id = int(pair["keep"]), int(pair["retire"])
    if dry_run:
        return {"ok": True, "dry_run": True, "keep": keep_id, "retire": retire_id}
    keep = store._one("SELECT content FROM facts WHERE fact_id = ?", (keep_id,))
    retire = store._one("SELECT content FROM facts WHERE fact_id = ?", (retire_id,))
    if keep is None or retire is None:
        return {"ok": False, "reason": "fact disappeared"}
    from . import maintenance
    maintenance.ensure_tombstones(store)
    maintenance.tombstone_fact(store, retire_id, reason="duplicate", batch_id=batch_id)
    store.remove_fact(retire_id)
    with store._tx():
        store._write("INSERT INTO audit(action, fact_id, actor, reason, before, after) VALUES (?,?,?,?,?,?)",
                     ("merge", keep_id, "factkit", f"merged duplicate #{retire_id}",
                      retire["content"], keep["content"]))
    return {"ok": True, "keep": keep_id, "retire": retire_id}


# ---------------------------------------------------- entity-aware scoring
# Nomic compresses paraphrases: a correct-but-generic fact sits around 0.51
# while an unrelated neighbour sits at 0.58, so a cosine floor cannot separate
# them. Ranking therefore gets a small additive overlap bonus. The bonus never
# admits a fact (``min_score`` is still checked against the raw cosine) and
# never outvotes a provider hit; it only reorders facts that are already close.
ENTITY_BOOST = 0.04       # the query names an entity this fact is about
ATTRIBUTE_BOOST = 0.02    # the query names an attribute this fact reports
MAX_ENTITY_BOOST = 0.08   # ceiling on the total additive bonus per fact

# Sentence-initial/question words are not entity names even when capitalised.
ENTITY_STOP = {
    "how", "what", "when", "where", "which", "who", "why", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "the", "a", "an", "my", "our", "your",
    "i", "we", "you", "it", "this", "that", "these", "those",
    "tell", "show", "give", "list", "find", "get", "please",
}

# Query attribute -> words that evidence that attribute in a fact. The first
# entry is the query word itself, so "port" matches "port" literally while
# "memory" also matches a bare "RAM" claim.
ATTRIBUTE_ALIASES = {
    # capacity aliases include the unit words, so "how much memory" reaches a
    # fact written as "1 GB" even when it never says RAM
    "memory": ("memory", "ram", "gb", "mb", "tb", "gib"),
    "ram": ("ram", "memory", "gb", "mb", "tb", "gib"),
    "storage": ("storage", "disk", "ssd", "nvme", "drive", "gb", "tb"),
    "disk": ("disk", "storage", "ssd", "nvme"),
    "cpu": ("cpu", "processor", "cores", "core"),
    "port": ("port",),
    "ip": ("ip", "address"),
    "version": ("version", "release"),
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*")
_WORDS_RE = re.compile(r"[a-z0-9]+")


def _norm_words(text: str) -> list[str]:
    return _WORDS_RE.findall((text or "").lower())


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    """True when ``needle`` appears as a contiguous token run in ``haystack``."""
    n = len(needle)
    if not n or n > len(haystack):
        return False
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _query_attributes(query: str) -> list[str]:
    out: list[str] = []
    for word in _norm_words(query):
        if word in ATTRIBUTE_ALIASES and word not in out:
            out.append(word)
    return out


def query_entity_tokens(query: str, known_entities=()) -> list[str]:
    """Deterministic entity-ish phrases named by ``query``.

    Three cheap sources, no LLM: capitalised runs ("Beacon Mini"), known entity
    names that appear anywhere in the query even when typed in lower case, and
    the facts' own ``entities`` column values passed in via ``known_entities``.
    """
    text = query or ""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(" ".join(run).lower())
            run.clear()

    for raw in _WORD_RE.findall(text):
        word = raw.strip("._-")
        if not word:
            continue
        low = word.lower()
        if (word[0].isupper() and low not in ENTITY_STOP
                and low not in ATTRIBUTE_ALIASES and not word.isdigit()):
            run.append(word)
        else:
            flush()
    flush()

    qwords = _norm_words(text)
    for name in sorted({str(n).strip() for n in known_entities if str(n).strip()}):
        nwords = _norm_words(name)
        if not nwords or len("".join(nwords)) < 3:
            continue
        if len(nwords) == 1 and (nwords[0] in ENTITY_STOP or nwords[0] in ATTRIBUTE_ALIASES):
            continue
        if _contains_sequence(qwords, nwords):
            out.append(name.lower())

    return list(dict.fromkeys(out))


def _entity_cell_names(value) -> list[str]:
    """Names held in a facts.entities cell: JSON list or comma/semicolon text."""
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except (ValueError, TypeError):
            pass
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def _known_entities(store, rows) -> list[str]:
    """Entity names known to the store: the facts.entities column plus the table."""
    names: set[str] = set()
    for row in rows:
        names.update(_entity_cell_names(dict(row).get("entities")))
    try:
        for row in store.read("SELECT name FROM entities"):
            name = str(dict(row).get("name") or "").strip()
            if name:
                names.add(name)
    except Exception:  # noqa: BLE001 - the table is optional on older stores
        pass
    return sorted(names)


def _overlap_boost(entity_phrases, attributes, content: str,
                   entities: str | None = "") -> float:
    """The scoring core, with query extraction already done once per request."""
    if not entity_phrases and not attributes:
        return 0.0

    haystack = _norm_words(f"{content or ''} {entities or ''}")
    haywords = set(haystack)
    boost = 0.0
    for phrase in entity_phrases:
        if _contains_sequence(haystack, phrase.split()):
            boost += ENTITY_BOOST
    for attribute in attributes:
        if haywords & set(ATTRIBUTE_ALIASES[attribute]):
            boost += ATTRIBUTE_BOOST
    return round(min(boost, MAX_ENTITY_BOOST), 6)


def entity_boost(query: str, content: str, entities: str | None = "",
                 known_entities=()) -> float:
    """Additive, capped boost for entity + attribute overlap.

    ``ENTITY_BOOST`` is granted once per distinct entity phrase the fact is
    about, ``ATTRIBUTE_BOOST`` once per distinct attribute word in the query,
    and the sum never exceeds ``MAX_ENTITY_BOOST``. Facts that match neither
    get exactly ``0.0``, so their cosine score is untouched.
    """
    return _overlap_boost(query_entity_tokens(query, known_entities),
                          _query_attributes(query), content, entities)


# --------------------------------------------------------------- prefetch
_QUERY_CACHE = None


def _query_cache():
    """Process-wide embedding cache; a cache fault must never break recall."""
    global _QUERY_CACHE
    if _QUERY_CACHE is None:
        import os
        from .embedcache import EmbedCache
        default_cache = None
        try:
            from hermes_constants import get_hermes_home  # type: ignore
            default_cache = str(get_hermes_home() / "embed_cache.db")
        except Exception:
            default_cache = str(Path.home() / ".hermes" / "embed_cache.db")
        path = os.environ.get("FACTKIT_EMBED_CACHE", default_cache)
        try:
            _QUERY_CACHE = EmbedCache(path)
        except Exception:  # noqa: BLE001
            _QUERY_CACHE = False
    return _QUERY_CACHE or None


def prefetch(store, message: str, *, limit: int = 5, min_trust: float = 0.3,
             min_score: float = 0.35, timeout: float = 1.5) -> list[dict] | None:
    """Semantic recall for a user message.

    Reads facts from the read-only provider store and vectors from factkit's own DB.
    """
    from .config import Config
    from .embed import from_config
    from .embedcache import cached_embed

    _maybe_migrate_vectors(store)

    cfg = Config.load(str(getattr(store, "db_path", "") or "") or None)
    client = from_config(cfg)
    client.timeout, client.retries = timeout, 0
    try:
        qvec = cached_embed(_query_cache(), client.embed_one, message,
                            str(getattr(cfg, "embed_model", "") or "default"))
    except Exception:  # noqa: BLE001 - endpoint down: caller decides
        return None
    if not qvec:
        return None

    import math
    import struct

    try:
        columns = {str(dict(r).get("name") or "") for r in store.read("PRAGMA table_info(facts)")}
    except Exception:  # noqa: BLE001 - older stores may not expose PRAGMA
        columns = set()
    entity_column = "f.entities" if "entities" in columns else "NULL"

    model = getattr(cfg, "embed_model", None)
    # Get vectors from factkit's own DB (or the store's own DB for Store)
    vec_rows = {}
    try:
        vconn, vclose = _vectors_conn(store, readonly=True)
        try:
            for r in vconn.execute(
                    "SELECT fact_id, model, dims, vec FROM fact_embeddings WHERE model = ?",
                    (model,)):
                d = dict(r) if not isinstance(r, dict) else r
                vec_rows[int(d["fact_id"])] = d
        finally:
            if vclose:
                vconn.close()
    except Exception:
        vec_rows = {}

    # Legacy duck-typed adapters still work; provider handles must read only
    # factkit's own DB so recall never silently depends on provider storage.
    if not vec_rows and not _is_provider_store(store):
        try:
            sread = getattr(store, "read", None)
            if sread:
                for r in sread(
                        "SELECT fact_id, vec FROM fact_embeddings WHERE model = ?",
                        (model,)):
                    d = dict(r) if not isinstance(r, dict) else r
                    vec_rows[int(d["fact_id"])] = d
        except Exception:
            pass

    if not vec_rows:
        return []

    fact_ids = list(vec_rows.keys())
    # Read facts from provider store
    try:
        placeholders = ",".join("?" for _ in fact_ids)
        rows = [dict(r) for r in store.read(
            f"SELECT f.fact_id, f.content, f.trust_score, {entity_column} AS entities "
            f"FROM facts f WHERE f.fact_id IN ({placeholders})",
            tuple(fact_ids))]
    except Exception:
        return []

    by_id = {int(r["fact_id"]): r for r in rows}
    known_entities = _known_entities(store, rows)
    entity_phrases = query_entity_tokens(message, known_entities)
    attributes = _query_attributes(message)

    out = []
    for fid, frow in by_id.items():
        vrow = vec_rows.get(fid)
        if not vrow:
            continue
        blob = vrow["vec"]
        if blob is None:
            continue
        try:
            n = len(blob) // 4
            vec = struct.unpack(f"<{n}f", blob)
        except Exception:
            continue
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        na = math.sqrt(sum(a * a for a in qvec))
        nb = math.sqrt(sum(b * b for b in vec))
        if not na or not nb:
            continue
        cosine = dot / (na * nb)
        if cosine < min_score or float(frow.get("trust_score") or 0) < min_trust:
            continue
        boost = _overlap_boost(entity_phrases, attributes, frow["content"],
                               frow.get("entities"))
        score = cosine + boost
        hit = {"fact_id": int(fid), "content": frow["content"],
               "score": round(score, 4), "trust": round(float(frow.get("trust_score") or 0), 2)}
        if boost:
            hit["boost"] = round(boost, 4)
        out.append(hit)
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


def index_pending(store, *, batch: int = 32, force: bool = False) -> dict:
    """Embed facts that have no vector for the configured model (writes to factkit's own DB)."""
    from .config import Config
    from .embed import from_config
    from .vectors import pack

    _maybe_migrate_vectors(store)

    cfg = Config.load(str(getattr(store, "db_path", "") or "") or None)
    client = from_config(cfg)
    model = cfg.embed_model
    have = set()
    try:
        vconn, vclose = _vectors_conn(store, readonly=False)
        try:
            for r in vconn.execute("SELECT fact_id FROM fact_embeddings WHERE model = ?", (model,)):
                have.add(int(r["fact_id"] if not isinstance(r, dict) else r["fact_id"]))
        finally:
            if vclose:
                vconn.close()
    except Exception:
        pass
    # Legacy adapters only: provider handles must count factkit's own rows so
    # migration + indexing converge instead of treating provider rows as done.
    if not _is_provider_store(store):
        try:
            sread = getattr(store, "read", None)
            if sread:
                for r in sread("SELECT fact_id FROM fact_embeddings WHERE model = ?", (model,)):
                    have.add(int(dict(r)["fact_id"] if not isinstance(r, dict) else r["fact_id"]))
        except Exception:
            pass
    try:
        sread = getattr(store, "read", None)
        if sread:
            rows = [dict(r) for r in sread("SELECT fact_id, content FROM facts ORDER BY fact_id")]
        else:
            rows = [dict(r) for r in store.db.execute("SELECT fact_id, content FROM facts ORDER BY fact_id")]
    except Exception:
        rows = []
    todo = [r for r in rows if force or int(r["fact_id"]) not in have]
    if not todo:
        return {"model": model, "indexed": 0, "pending": 0}

    done = 0
    try:
        vconn, vclose = _vectors_conn(store, readonly=False)
        try:
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                vecs = client.embed([r["content"] for r in chunk])
                if not vecs:
                    break
                for row, vec in zip(chunk, vecs):
                    vconn.execute(
                        "INSERT OR REPLACE INTO fact_embeddings(fact_id, model, dims, vec) "
                        "VALUES (?,?,?,?)", (row["fact_id"], model, len(vec), pack(vec)))
                if vclose:
                    vconn.commit()
                else:
                    try: vconn.commit()
                    except Exception: pass
                done += len(chunk)
        finally:
            if vclose:
                vconn.close()
    except Exception as exc:  # noqa: BLE001 - endpoint down: report, do not raise
        return {"model": model, "indexed": done, "pending": len(todo) - done, "error": str(exc)}
    return {"model": model, "indexed": done, "pending": len(todo) - done}


# ------------------------------------------------------------ rank fusion
def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse_results(store, query: str, provider_facts: list[dict], *,
                 semantic_weight: float = 0.5, limit: int = 10,
                 min_trust: float = 0.3, timeout: float = 1.5) -> list[dict] | None:
    """Re-rank provider hits with an embedding channel (RRF), keeping every provider fact.

    Returns the merged list, or None when embeddings are unavailable so the caller
    can return the provider's result untouched.
    """
    # Score the whole embedding channel: within-query margins are ~0.03 between
    # neighbours, so a global similarity floor cannot separate signal from noise.
    hits = prefetch(store, query, limit=1_000_000, min_trust=min_trust,
                    min_score=0.0, timeout=timeout)
    if hits is None:
        return None
    sem_rank = {h["fact_id"]: i for i, h in enumerate(hits, 1)}
    sem_score = {h["fact_id"]: h["score"] for h in hits}

    by_id: dict[int, dict] = {}
    for i, fact in enumerate(provider_facts, 1):
        fact = dict(fact)
        score = _rrf(i) + (semantic_weight * _rrf(sem_rank[fact["fact_id"]])
                           if fact["fact_id"] in sem_rank else 0.0)
        fact["score"] = round(score, 6)
        if fact["fact_id"] in sem_rank:
            fact["semantic_score"] = sem_score[fact["fact_id"]]
        by_id[fact["fact_id"]] = fact

    # Provider rows own the slots: never drop one in favour of a semantic-only
    # row. With more provider rows than slots, truncate in provider order.
    if limit and limit < len(provider_facts):
        return [by_id[fact["fact_id"]] for fact in provider_facts[:limit]]

    # Reserve only the slots the provider channel left empty. A fixed cosine
    # floor misbehaves: nomic puts weak matches at ~0.51, barely below strong
    # ones, so rank-relative gating is used instead.
    if limit:
        spare = max(0, limit - len(provider_facts))
        cap = min(max(1, limit // 2), spare)
    else:
        cap = 1 if limit in (None, 0) else min(max(1, limit // 2), max(0, limit - len(provider_facts)))
    reserve: list[dict] = []
    cutoff = max(1, min(len(hits), 5))   # top-5 of the embedding channel
    for hit in hits[:cutoff]:
        if hit["fact_id"] in by_id:
            continue
        if len(reserve) >= cap:
            break
        score = semantic_weight * _rrf(sem_rank[hit["fact_id"]])
        row = store.read("SELECT fact_id, content, category, tags, trust_score FROM facts "
                         "WHERE fact_id = ?", (hit["fact_id"],))
        if not row:
            continue
        fact = dict(row[0])
        fact["score"] = round(score, 6)
        fact["semantic_score"] = hit["score"]
        fact["source"] = "semantic"
        reserve.append(fact)

    ordered = sorted(by_id.values(), key=lambda f: -f["score"])
    return (ordered + reserve)[:limit] if limit else ordered + reserve




# ------------------------------------------------------- maintenance batches
def _enrich(store, pair: dict) -> dict:
    """Attach the fact metadata the policy engine needs."""
    out = dict(pair)
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(facts)")}
    source = "source" if "source" in cols else "NULL AS source"
    for side in ("keep", "retire"):
        row = store._one(f"SELECT category, trust_score, {source} FROM facts WHERE fact_id = ?",
                         (int(pair[side]),))
        if row is not None:
            out[f"{side}_category"] = row["category"]
            out[f"{side}_trust"] = row["trust_score"]
            out[f"{side}_source"] = row["source"]
    return out


def _ensure_proposal_columns(store) -> None:
    """Add the ``batch_id`` / ``evidence`` columns to an older ``conflict_proposals``.

    Additive only: ``ALTER TABLE ... ADD COLUMN`` when missing, never a rewrite or
    drop, so existing rows and their values survive the migration.
    """
    cols = {str(r[1]) for r in store._conn.execute("PRAGMA table_info(conflict_proposals)")}
    for name, ddl in (("batch_id", "batch_id TEXT"), ("evidence", "evidence TEXT")):
        if name in cols:
            continue
        with store._tx():
            store._write(f"ALTER TABLE conflict_proposals ADD COLUMN {ddl}")


def _proposal_evidence(store, pair: dict, ratio: float) -> str:
    """Compact JSON evidence for one proposal (the facts and the disputed slot)."""
    import json

    diffs = pair.get("diffs") or []
    texts = {}
    for side in ("keep", "retire"):
        text = pair.get(f"{side}_text")
        if text is None:
            row = store._one("SELECT content FROM facts WHERE fact_id = ?", (int(pair[side]),))
            text = row["content"] if row is not None else ""
        texts[side] = text or ""
    return json.dumps(
        {"kind": pair.get("kind") or "conflict",
         "slot": diffs[0].get("slot") if diffs else None,
         "keep_text": texts["keep"],
         "retire_text": texts["retire"],
         "ratio": ratio},
        sort_keys=True, ensure_ascii=False)


def _record_proposal(store, pair: dict, status: str, batch_id: str | None) -> None:
    from . import maintenance
    maintenance.ensure_tombstones(store)   # keeps schema creation in one place
    _ensure_proposal_columns(store)        # additive: old DBs gain the new columns
    diffs = pair.get("diffs") or []
    slots = ",".join(str(d.get("slot", "")) for d in diffs) or "duplicate"
    delta = " | ".join(f"{d.get('slot')}: {d.get('keep')} vs {d.get('retire')}" for d in diffs) or "same content"
    ratio = float(pair.get("jaccard") or pair.get("ratio") or 0.0)
    evidence = _proposal_evidence(store, pair, ratio)
    with store._tx():
        store._write(
            "INSERT OR IGNORE INTO conflict_proposals"
            "(keep_id, retire_id, jaccard, slots, delta, status, note, batch_id, evidence) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (int(pair["keep"]), int(pair["retire"]), ratio,
             slots, delta, status, f"batch={batch_id}" if batch_id else "",
             batch_id, evidence))


def maintain(store, *, policy=None, dry_run: bool = True, limit: int | None = None,
             smoke_cases: list[dict] | None = None, batch_id: str | None = None,
             apply_fn=None) -> dict:
    """Classify candidates and (optionally) apply the auto class under limits.

    Nothing is written to ``facts`` unless ``dry_run`` is False AND the policy
    allows the auto class. Every retirement is tombstoned first, so a batch can
    be reverted with :func:`factkit.maintenance.restore_fact`.
    """
    import datetime as _dt
    import uuid as _uuid

    from . import maintenance, policy as policy_mod, smoke as smoke_mod

    policy = policy or policy_mod.Policy()
    batch_id = batch_id or _dt.datetime.now().strftime("b%Y%m%d%H%M%S") + "-" + _uuid.uuid4().hex[:6]
    db = store.db_path
    conflicts = [_enrich(store, p) for p in find_conflicts(db)]
    dupes = []
    for p in merge_duplicates(db):
        p = dict(p)
        p["kind"] = "duplicate"
        dupes.append(_enrich(store, p))

    total = store.count_facts()
    candidates = conflicts + dupes
    summary = {"batch": batch_id, "dry_run": dry_run, "scanned": total,
               "candidates": len(candidates), "auto": 0, "proposed": 0,
               "never": 0, "applied": 0, "aborted": None, "reverted": False,
               "smoke_before": None, "smoke_after": None}

    budget = policy.max_apply if limit is None else min(limit, policy.max_apply)
    if total and len(candidates) > policy.max_ratio * total:
        summary["aborted"] = f"{len(candidates)} candidates exceed {policy.max_ratio:.0%} of {total} facts"
        return summary

    before = smoke_mod.run_smoke(store, smoke_cases or []) if smoke_cases else None
    if before is not None:
        summary["smoke_before"] = {"ok": before.get("ok"), "checked": before.get("checked")}

    applied: list[dict] = []
    for pair in candidates:
        verdict = policy_mod.classify(pair, policy=policy)
        if verdict == "never":
            summary["never"] += 1
            continue
        if verdict == "propose" or not policy.auto:
            summary["proposed"] += 1
            if not dry_run:
                _record_proposal(store, pair, "pending", batch_id)
            continue
        summary["auto"] += 1
        if dry_run:
            continue
        if len(applied) >= budget:
            summary["proposed"] += 1
            _record_proposal(store, pair, "pending", batch_id)
            continue
        fn = apply_fn or (apply_merge if pair.get("kind") == "duplicate" else apply_conflict)
        try:
            result = fn(store, pair, dry_run=False, batch_id=batch_id)
        except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the batch
            result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        if result.get("ok"):
            applied.append(result)
            _record_proposal(store, pair, "applied", batch_id)
        else:
            summary["aborted"] = f"apply failed: {result.get('reason')}"
            break

    summary["applied"] = len(applied)

    after = smoke_mod.run_smoke(store, smoke_cases or []) if smoke_cases else None
    if after is not None:
        summary["smoke_after"] = {"ok": after.get("ok"), "checked": after.get("checked")}
        if before is not None and applied:
            cmp = smoke_mod.compare(before, after)
            if not cmp.get("ok"):
                restored = []
                for item in applied:
                    try:
                        restored.append(maintenance.restore_fact(store, item["retire"]))
                    except Exception as exc:  # noqa: BLE001
                        restored.append({"error": str(exc)})
                summary["reverted"] = True
                summary["regressed"] = cmp.get("regressed", [])
                summary["restored"] = restored
    return summary
