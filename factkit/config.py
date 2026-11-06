"""Configuration: every knob is an env var, with sane offline defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The two databases the dashboard can show. ``live`` is the store Hermes' own
# fact tools read and write; ``corpus`` is factkit's standalone store.
LIVE_DB = Path.home() / ".hermes" / "memory_store.db"
CORPUS_DB = Path.home() / ".factkit" / "facts.db"
DB_MODES = ("live", "corpus")

DEFAULTS = {
    "db": "~/.factkit/facts.db",
    "embed_base_url": "http://127.0.0.1:1234/v1",
    "embed_model": "text-embedding-nomic-embed-text-v1.5",
    "embed_api_key": "",
    "embed_timeout": "20",
    "embed_batch": "32",
    "llm_base_url": "http://127.0.0.1:1234/v1",
    "llm_model": "local-model",
    "llm_api_key": "",
    "llm_timeout": "120",
    "ttl_days": "90",
    "slot_ttl_days": "180",
    "dedupe_threshold": "0.92",
    "rrf_k": "60",
    "review_after_default": "0",
}


def _env(key: str) -> str:
    return os.environ.get("FACTKIT_" + key.upper(), DEFAULTS[key])


@dataclass
class Config:
    db_path: Path
    embed_base_url: str
    embed_model: str
    embed_api_key: str
    embed_timeout: float
    embed_batch: int
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout: float
    ttl_days: int
    slot_ttl_days: int
    dedupe_threshold: float
    rrf_k: int
    review_after_default: int
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, db_path: str | None = None) -> "Config":
        def num(key: str, cast):
            raw = os.environ.get("FACTKIT_" + key.upper(), DEFAULTS[key])
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return cast(DEFAULTS[key])

        path = db_path or _env("db")
        return cls(
            db_path=Path(path).expanduser(),
            embed_base_url=_env("embed_base_url").rstrip("/"),
            embed_model=_env("embed_model"),
            embed_api_key=_env("embed_api_key"),
            embed_timeout=num("embed_timeout", float),
            embed_batch=num("embed_batch", int),
            llm_base_url=_env("llm_base_url").rstrip("/"),
            llm_model=_env("llm_model"),
            llm_api_key=_env("llm_api_key"),
            llm_timeout=num("llm_timeout", float),
            ttl_days=num("ttl_days", int),
            slot_ttl_days=num("slot_ttl_days", int),
            dedupe_threshold=num("dedupe_threshold", float),
            rrf_k=num("rrf_k", int),
            review_after_default=num("review_after_default", int),
        )


# --------------------------------------------------------------- dashboard db
@dataclass(frozen=True)
class DashboardDB:
    """Which SQLite file the dashboard serves, and whether it may write it.

    ``readonly`` is enforced in two layers: :meth:`open` returns a
    ``Store(readonly=True)`` (SQLite ``mode=ro`` + ``PRAGMA query_only``), and
    the HTTP handler refuses POSTs. ``live`` is always read-only.
    """

    path: Path
    mode: str
    readonly: bool
    source: str = "default"

    @property
    def label(self) -> str:
        access = "read-only" if self.readonly else "read-write"
        return f"{self.mode} · {self.path} · {access}"

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "mode": self.mode,
            "readonly": self.readonly,
            "writable": not self.readonly,
            "label": self.label,
            "source": self.source,
        }

    def open(self):
        """Open the selected store; never write-capable when ``readonly``."""
        from .store import Store

        return Store(self.path, readonly=self.readonly)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _same_path(a: str | Path, b: str | Path) -> bool:
    try:
        return Path(a).expanduser().resolve(strict=False) == Path(b).expanduser().resolve(strict=False)
    except OSError:
        return False


def resolve_dashboard_db(db: str | None = None, *, mode: str | None = None,
                         readonly: bool | None = None,
                         environ: dict | None = None) -> DashboardDB:
    """Resolve the dashboard's database and access mode.

    Precedence for the path: ``db`` argument (``--db``) > ``FACTKIT_DB`` >
    the mode's default (live: ``~/.hermes/memory_store.db``, corpus:
    ``~/.factkit/facts.db``). ``mode`` argument > ``FACTKIT_DB_MODE`` > live.

    Safety: a live database is *always* read-only. Pointing ``FACTKIT_DB`` at
    ``~/.hermes/memory_store.db`` with ``FACTKIT_DB_MODE=corpus`` still yields
    ``readonly=True``. ``FACTKIT_READONLY=1`` (or ``readonly=True``) makes a
    corpus dashboard read-only too; there is no flag that opens live read-write.
    """
    env = os.environ if environ is None else environ
    raw_mode = (mode or env.get("FACTKIT_DB_MODE") or "").strip().lower()
    if raw_mode and raw_mode not in DB_MODES:
        raise ValueError(f"unknown database mode {raw_mode!r}; expected one of {DB_MODES}")

    explicit = db or env.get("FACTKIT_DB") or None
    if explicit:
        path = Path(str(explicit)).expanduser()
        source = "--db" if db else "FACTKIT_DB"
        resolved_mode = raw_mode or ("live" if _same_path(path, LIVE_DB) else "corpus")
    else:
        resolved_mode = raw_mode or "live"
        path = LIVE_DB if resolved_mode == "live" else CORPUS_DB
        source = "default"

    forced_ro = _truthy(env.get("FACTKIT_READONLY")) or readonly is True
    is_live = resolved_mode == "live" or _same_path(path, LIVE_DB)
    return DashboardDB(path=path, mode=resolved_mode,
                       readonly=is_live or forced_ro, source=source)
