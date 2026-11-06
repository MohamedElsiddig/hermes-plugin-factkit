"""Float32 vector packing and cosine similarity (numpy-accelerated when present)."""
from __future__ import annotations

import array
import math

try:  # optional
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def unpack(blob: bytes) -> array.array:
    out = array.array("f")
    out.frombytes(blob)
    return out


def dims(blob: bytes) -> int:
    return len(blob) // 4


def cosine(a: array.array, b: array.array) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    if _np is not None:
        x = _np.frombuffer(a.tobytes(), dtype=_np.float32)
        y = _np.frombuffer(b.tobytes(), dtype=_np.float32)
        nx = float(_np.linalg.norm(x))
        ny = float(_np.linalg.norm(y))
        if nx == 0.0 or ny == 0.0:
            return 0.0
        return float(_np.dot(x, y) / (nx * ny))
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
