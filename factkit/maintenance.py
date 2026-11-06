"""Tombstones for retired facts.

Retiring a fact used to be destructive: ``remove_fact`` drops the row, its FTS
entry, its entity links and its HRR vector, so a wrong conflict resolution or a
merge could not be undone. A tombstone keeps a copy of the retired row before
the caller deletes it, and ``restore_fact`` re-adds the fact through the
store's own ``add_fact_status`` path so FTS, entities and the HRR vector come
back with it.

This module is stdlib-only and imports no Hermes code at module level, so it
can be imported by the CLI and by tests without the plugin installed.  The
store is used duck-typed:

* ``store.db_path`` -- path to the SQLite database (default fallback for writes)
* ``store.read(sql, params)`` -- rows in a mapping shape (``sqlite3.Row``)
* ``store.count_facts()`` -- live fact count
* ``store.add_fact_status(content, category=..., tags=..., trust=...,
  entities=[...])`` -- the holographic ``MemoryStore`` insert path

Writes go through ``store._write`` when it exists.  That is the holographic
plugin's own single-statement writer: it runs on the store's shared connection,
so a tombstone written inside a caller's ``with store._tx():`` block joins that
transaction instead of opening a second connection (which would block on the
write lock).  ``store.write`` and ``store.conn`` are accepted as fallbacks for
other store shapes.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

__all__ = [
    "ensure_tombstones",
    "tombstone_fact",
    "restore_fact",
    "list_tombstones",
    "purge_tombstones",
]

_TABLE = """
CREATE TABLE IF NOT EXISTS fact_tombstones (
    fact_id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT,
    tags TEXT,
    trust_score REAL,
    entities TEXT,
    hrr_vector BLOB,
    retired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retired_by TEXT,
    reason TEXT,
    batch_id TEXT,
    restored_at TIMESTAMP
)
"""

_INDEX = ("CREATE INDEX IF NOT EXISTS idx_fact_tombstones_restored "
          "ON fact_tombstones(restored_at)")

_AUDIT = (
    "CREATE TABLE IF NOT EXISTS audit ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "action TEXT, fact_id INTEGER, actor TEXT, reason TEXT, before TEXT, after TEXT)"
)

_PROPOSALS = (
    "CREATE TABLE IF NOT EXISTS conflict_proposals ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "keep_id INTEGER NOT NULL, retire_id INTEGER NOT NULL, jaccard REAL NOT NULL, "
    "slots TEXT NOT NULL, delta TEXT NOT NULL, status TEXT DEFAULT 'pending', "
    "resolved_at TIMESTAMP, note TEXT DEFAULT '', UNIQUE(keep_id, retire_id, status))"
)

_COLUMNS = ("fact_id, content, category, tags, trust_score, entities, hrr_vector, "
            "retired_at, retired_by, reason, batch_id, restored_at")


# ------------------------------------------------------------------- helpers
def _write(store, sql: str, params=()):
    """Run one write statement on the store's own connection.

    Preference order is ``_write`` (the plugin's shared-connection writer, which
    joins an already open ``_tx``), then ``write``, then the raw connection.
    """
    fn = getattr(store, "_write", None)
    if callable(fn):
        return fn(sql, params)
    fn = getattr(store, "write", None)
    if callable(fn):
        return fn(sql, params)
    conn = getattr(store, "conn", None) or getattr(store, "_conn", None)
    if conn is None:
        raise AttributeError("store exposes no write path (_write, write or conn)")
    cur = conn.execute(sql, params)
    if not getattr(conn, "in_transaction", False):
        conn.commit()
    return cur


@contextlib.contextmanager
def _tx(store):
    """Join the store's transaction when it provides one, else no-op."""
    fn = getattr(store, "_tx", None)
    if callable(fn):
        tx: Any = fn()
        with tx:
            yield
    else:
        yield


def _rows(store, sql: str, params=()) -> list[dict]:
    rows = store.read(sql, params)
    out = []
    for row in rows:
        try:
            out.append(dict(row))
        except (TypeError, ValueError):  # tuple rows from a plain connection
            out.append({str(i): value for i, value in enumerate(row)})
    return out


def _one(store, sql: str, params=()) -> dict | None:
    rows = _rows(store, sql, params)
    return rows[0] if rows else None


def _fact_columns(store) -> set[str]:
    try:
        return {str(row[0]) for row in store.read("PRAGMA table_info(facts)")}
    except Exception:  # noqa: BLE001 - schema probe must never be fatal
        return set()


def _declared_entities(store, fact_id: int) -> list[str]:
    """Names declared for the fact; extracted-only entities are not re-declared.

    Both queries are best-effort: an older schema without ``entity_type``, or a
    store without the entity tables, yields just the names it can read.
    """
    queries = (
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
        "WHERE fe.fact_id = ? AND e.entity_type = 'declared' ORDER BY e.name",
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
        "WHERE fe.fact_id = ? ORDER BY e.name",
    )
    for sql in queries:
        try:
            rows = store.read(sql, (fact_id,))
        except Exception:  # noqa: BLE001 - missing table/column: try the next shape
            continue
        return [str(row[0]) for row in rows if row[0]]
    return []


def _decode_entities(raw) -> list[str]:
    if not raw:
        return []
    try:
        names = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(name) for name in names if isinstance(name, str) and name.strip()]


def _get(store, fact_id: int) -> dict | None:
    return _one(store, f"SELECT {_COLUMNS} FROM fact_tombstones WHERE fact_id = ?", (fact_id,))


# ---------------------------------------------------------------------- API
def ensure_tombstones(store) -> None:
    """Create the tombstone table and its index if they are missing. Idempotent."""
    _write(store, _TABLE)
    _write(store, _INDEX)
    _write(store, _AUDIT)
    _write(store, _PROPOSALS)


def tombstone_fact(store, fact_id, *, reason: str = "", batch_id=None,
                   by: str = "factkit") -> dict:
    """Snapshot a fact into ``fact_tombstones`` and return the stored row.

    Raises ``ValueError`` when the fact does not exist.  The fact itself is left
    untouched: the caller retires it with ``remove_fact`` afterwards (or keeps
    it, for a dry run).
    """
    try:
        fid = int(fact_id)
    except (TypeError, ValueError):
        raise ValueError(f"invalid fact_id: {fact_id!r}") from None

    ensure_tombstones(store)
    columns = _fact_columns(store)
    wanted = ["fact_id", "content", "category", "tags", "trust_score"]
    if "hrr_vector" in columns:
        wanted.append("hrr_vector")
    row = _one(store, f"SELECT {', '.join(wanted)} FROM facts WHERE fact_id = ?", (fid,))
    if row is None:
        raise ValueError(f"fact {fid} not found")

    entities = _declared_entities(store, fid)
    _write(
        store,
        "INSERT OR REPLACE INTO fact_tombstones "
        "(fact_id, content, category, tags, trust_score, entities, hrr_vector, "
        " retired_at, retired_by, reason, batch_id, restored_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, NULL)",
        (fid, row["content"], row.get("category"), row.get("tags"),
         row.get("trust_score"), json.dumps(entities), row.get("hrr_vector"),
         by, reason, batch_id),
    )
    stored = _get(store, fid)
    if stored is None:  # pragma: no cover - the insert above either lands or raises
        raise RuntimeError(f"tombstone for fact {fid} was not stored")
    return stored


def restore_fact(store, fact_id) -> dict:
    """Re-add a retired fact through the store's insert path and mark it restored.

    Returns ``{'fact_id': new_id, 'restored_from': fact_id, 'content': ...}``.
    Raises ``ValueError`` when the tombstone is missing or was already restored.
    """
    try:
        fid = int(fact_id)
    except (TypeError, ValueError):
        raise ValueError(f"invalid fact_id: {fact_id!r}") from None

    ensure_tombstones(store)
    row = _get(store, fid)
    if row is None:
        raise ValueError(f"no tombstone for fact {fid}")
    if row.get("restored_at"):
        raise ValueError(f"fact {fid} was already restored at {row['restored_at']}")

    content = row["content"]
    names = _decode_entities(row.get("entities"))
    with _tx(store):
        # add_fact_status returns (fact_id, created); a duplicate content row is
        # reused rather than failing, which keeps a double retirement harmless.
        new_id, _created = store.add_fact_status(
            content,
            category=row.get("category") or "general",
            tags=row.get("tags") or "",
            trust=row.get("trust_score"),
            entities=names,
        )
        _write(store, "UPDATE fact_tombstones SET restored_at = CURRENT_TIMESTAMP "
                      "WHERE fact_id = ?", (fid,))

    return {"fact_id": int(new_id), "restored_from": fid, "content": content}


def list_tombstones(store, *, include_restored: bool = False) -> list[dict]:
    """Tombstone rows, newest first; restored rows only when asked for."""
    ensure_tombstones(store)
    where = "" if include_restored else " WHERE restored_at IS NULL"
    return _rows(store, f"SELECT {_COLUMNS} FROM fact_tombstones{where} "
                        "ORDER BY retired_at DESC, fact_id DESC")


def purge_tombstones(store, *, days: int = 90) -> int:
    """Delete restored tombstones and unrestored ones older than *days*. Returns the count."""
    ensure_tombstones(store)
    horizon = f"-{max(0, int(days))} days"
    cur = _write(
        store,
        "DELETE FROM fact_tombstones "
        "WHERE restored_at IS NOT NULL OR retired_at < datetime('now', ?)",
        (horizon,),
    )
    count = getattr(cur, "rowcount", 0)
    return int(count) if count and count > 0 else 0
