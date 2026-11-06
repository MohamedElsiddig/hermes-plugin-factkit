"""Configuration: every knob is an env var, with sane offline defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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
