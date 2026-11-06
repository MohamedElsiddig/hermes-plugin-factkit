"""Maintenance operations over a live holographic store.

All writes go through the holographic plugin's own ``MemoryStore`` API
(``add_fact_status`` / ``update_fact`` / ``replace_fact`` / ``remove_fact``), so
FTS, entity links, HRR vectors, trust scores and bank invalidation stay
consistent — the same path the agent's ``fact_store`` tool uses.

Detection is read-only. Every operation supports ``dry_run`` and records what it
did in the ``audit`` table, written by factkit with actor ``factkit``.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sqlite3
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "holographic"


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

    @classmethod
    def _open_snapshot(cls, path: str) -> sqlite3.Connection:
        import shutil
        import tempfile
        import time

        source = Path(path)
        stamp = (str(source), source.stat().st_mtime_ns, source.stat().st_size)
        cached = cls._snapshots.get(stamp)
        if cached:
            return sqlite3.connect(cached, check_same_thread=False)

        target_dir = Path(tempfile.mkdtemp(prefix="factkit-ro-"))
        target = target_dir / source.name
        shutil.copy2(source, target)
        for suffix in ("-wal", "-shm"):
            side = source.with_name(source.name + suffix)
            if side.exists():
                try:
                    shutil.copy2(side, target.with_name(target.name + suffix))
                except OSError:
                    pass
        conn = sqlite3.connect(str(target), check_same_thread=False)
        conn.execute("SELECT COUNT(*) FROM facts").fetchone()

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

    def read(self, sql: str, params=()) -> list[dict]:
        try:
            return [dict(r) for r in self._conn.execute(sql, params)]
        except sqlite3.OperationalError:
            # opening can succeed and the first read still fail when the WAL
            # side files are off limits; retry once from a private snapshot
            self._conn = self._open_snapshot(self.db_path)
            self._conn.row_factory = sqlite3.Row
            return [dict(r) for r in self._conn.execute(sql, params)]

    def count_facts(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()
        return int(row["n"] if row else 0)


def load_plugin_store(db_path: str | Path | None = None):
    """Import the holographic plugin's MemoryStore without running its __init__."""
    if "holographic" not in sys.modules:
        pkg = types.ModuleType("holographic")
        pkg.__path__ = [str(PLUGIN_DIR)]
        sys.modules["holographic"] = pkg
    try:  # gateway: the real runtime helpers are importable
        import hermes_state_wal  # noqa: F401
    except ImportError:  # CLI: install stand-ins before the provider loads
        try:
            from . import standalone
        except ImportError:
            raise ImportError(
                "holographic provider needs the Hermes runtime; run this from the "
                "gateway or from the factkit CLI"
            )
        standalone.ensure()
    from holographic.store import MemoryStore  # noqa: WPS433 - deliberate late import

    path = str(Path(db_path).expanduser()) if db_path else str(Path.home() / ".hermes" / "memory_store.db")
    return MemoryStore(path)


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
        for r in conn.execute(
                "SELECT fe.fact_id, e.name FROM fact_entities fe "
                "JOIN entities e ON e.entity_id = fe.entity_id "
                "WHERE e.entity_type = 'declared'"):
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
def prefetch(store, message: str, *, limit: int = 5, min_trust: float = 0.3,
             min_score: float = 0.35, timeout: float = 1.5) -> list[dict] | None:
    """Semantic recall for a user message, read from the live store.

    Uses the embeddings in ``fact_embeddings`` and the configured embedding
    endpoint. Returns ``None`` when the endpoint is unavailable (so a caller can
    trip a circuit breaker) and ``[]`` when nothing scores above the threshold.
    """
    from .config import Config
    from .embed import from_config

    cfg = Config.load(str(getattr(store, "db_path", "") or "") or None)
    client = from_config(cfg)
    client.timeout, client.retries = timeout, 0
    try:
        qvec = client.embed_one(message)
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

    rows = [dict(r) for r in store.read(
        "SELECT f.fact_id, f.content, f.trust_score, "
        f"{entity_column} AS entities, e.vec FROM fact_embeddings e "
        "JOIN facts f ON f.fact_id = e.fact_id WHERE e.model = ? "
        "ORDER BY f.fact_id", (cfg.embed_model,))]
    known_entities = _known_entities(store, rows)
    entity_phrases = query_entity_tokens(message, known_entities)
    attributes = _query_attributes(message)

    out = []
    for row in rows:
        blob = row["vec"]
        n = len(blob) // 4
        vec = struct.unpack(f"<{n}f", blob)
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        na = math.sqrt(sum(a * a for a in qvec))
        nb = math.sqrt(sum(b * b for b in vec))
        if not na or not nb:
            continue
        cosine = dot / (na * nb)
        # Admission is unchanged: the raw cosine and trust must clear the gates.
        # The entity/attribute bonus below only reorders facts that are admitted.
        if cosine < min_score or float(row["trust_score"] or 0) < min_trust:
            continue
        boost = _overlap_boost(entity_phrases, attributes, row["content"],
                               row.get("entities"))
        score = cosine + boost
        hit = {"fact_id": int(row["fact_id"]), "content": row["content"],
               "score": round(score, 4), "trust": round(float(row["trust_score"] or 0), 2)}
        if boost:
            hit["boost"] = round(boost, 4)
        out.append(hit)
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


def index_pending(store, *, batch: int = 32, force: bool = False) -> dict:
    """Embed facts that have no vector for the configured model."""
    from .config import Config
    from .embed import from_config
    from .vectors import pack

    cfg = Config.load(str(getattr(store, "db_path", "") or "") or None)
    client = from_config(cfg)
    model = cfg.embed_model
    have = {int(r["fact_id"]) for r in store.read(
        "SELECT fact_id FROM fact_embeddings WHERE model = ?", (model,))}
    rows = [dict(r) for r in store.read("SELECT fact_id, content FROM facts ORDER BY fact_id")]
    todo = [r for r in rows if force or int(r["fact_id"]) not in have]
    if not todo:
        return {"model": model, "indexed": 0, "pending": 0}

    done = 0
    try:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            vecs = client.embed([r["content"] for r in chunk])
            if not vecs:
                break
            conn = sqlite3.connect(store.db_path, timeout=10.0)
            try:
                for row, vec in zip(chunk, vecs):
                    conn.execute(
                        "INSERT OR REPLACE INTO fact_embeddings(fact_id, model, dims, vec) "
                        "VALUES (?,?,?,?)", (row["fact_id"], model, len(vec), pack(vec)))
                conn.commit()
            finally:
                conn.close()
            done += len(chunk)
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

    # Reserve part of the result set for facts the provider missed entirely.
    # Provider misses enter only when they are close to the top of the embedding
    # channel. A fixed cosine floor misbehaves: nomic puts weak matches at ~0.51,
    # barely below strong ones, so rank-relative gating is used instead.
    reserve: list[dict] = []
    cutoff = max(1, min(len(hits), 5))   # top-5 of the embedding channel
    for hit in hits[:cutoff]:
        if hit["fact_id"] in by_id:
            continue
        if len(reserve) >= max(1, limit // 2):
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
    keep = max(0, limit - len(reserve))
    return (ordered[:keep] + reserve)[:limit] if limit else ordered + reserve




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
