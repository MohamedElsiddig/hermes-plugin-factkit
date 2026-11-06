"""OpenAI-compatible embedding client: Ollama, vLLM, llama.cpp, hosted APIs.

The client is deliberately dependency-free (urllib) so the kit runs on a Pi with
nothing but the standard library. Failures are *soft*: callers get ``None`` and
retrieval degrades to BM25 instead of raising.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Sequence


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingClient:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 20.0
    batch_size: int = 32
    retries: int = 2
    backoff: float = 1.5

    # ---------------------------------------------------------------- request
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:  # 4xx is not retryable
                detail = exc.read().decode(errors="replace")[:400]
                if 400 <= exc.code < 500 and exc.code != 429:
                    raise EmbeddingError(f"HTTP {exc.code}: {detail}") from exc
                last = EmbeddingError(f"HTTP {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(self.backoff ** attempt)
        raise EmbeddingError(f"embedding request failed: {last}")

    # ------------------------------------------------------------------- api
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            data = self._post("/embeddings", {"model": self.model, "input": chunk})
            rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            if len(rows) != len(chunk):
                raise EmbeddingError(
                    f"expected {len(chunk)} embeddings, got {len(rows)}")
            out.extend([row["embedding"] for row in rows])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def try_embed(self, texts: Iterable[str]) -> list[list[float]] | None:
        """Never raises: returns None when the service is unreachable."""
        try:
            return self.embed(list(texts))
        except EmbeddingError:
            return None

    def health(self) -> dict:
        started = time.time()
        try:
            models = self._post  # noqa: F841  (documented use below)
            req = urllib.request.Request(f"{self.base_url}/models")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            ids = [m.get("id") for m in data.get("data", [])]
            return {"ok": True, "models": ids, "has_model": self.model in ids,
                    "latency_ms": round((time.time() - started) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return {"ok": False, "error": str(exc)[:200],
                    "latency_ms": round((time.time() - started) * 1000, 1)}


def from_config(cfg) -> EmbeddingClient:
    return EmbeddingClient(
        base_url=cfg.embed_base_url,
        model=cfg.embed_model,
        api_key=cfg.embed_api_key,
        timeout=cfg.embed_timeout,
        batch_size=cfg.embed_batch,
    )
