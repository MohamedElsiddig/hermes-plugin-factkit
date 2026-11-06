"""Exact canonicalisation of attribute values, for conflict detection.

A global similarity threshold cannot separate a contradiction from a
rephrasing (``16GB`` vs ``16384MB`` vs ``16 gb``), so unit normalisation here
is deliberately *exact*: parse one recognised value shape, return a tuple that
compares equal only when the underlying value is equal. Anything unfamiliar
falls back to a lower-cased string, which degrades to plain string equality.

Every function is total: junk, empty strings and ``None`` never raise. A value
that cannot be parsed is returned unchanged (lower-cased and stripped), never
guessed at. Pure stdlib, no imports from the rest of the package.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# Attribute names whose values are byte capacities.
CAPACITY_ATTRS = frozenset({"unit", "ram", "memory", "storage", "disk", "ssd"})

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?b)\s*$", re.IGNORECASE)
_SIZE_FACTORS = {
    "b": 1,
    "kb": 1024,
    "mb": 1024 ** 2,
    "gb": 1024 ** 3,
    "tb": 1024 ** 4,
}

_CLOCK_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ghz|mhz|hz)\s*$", re.IGNORECASE)
_CLOCK_FACTORS = {"hz": Decimal("0.001"), "mhz": Decimal(1), "ghz": Decimal(1000)}

_CORES_RE = re.compile(
    r"^\s*(\d+)\s*(?:cores?|cpus?|vcpus?|threads?)?\s*$", re.IGNORECASE
)

# "8080", "port 8080", "port=8080", "8000-8100" (hyphen or en/em dash).
_PORT_RE = re.compile(
    r"^\s*(?:port\s*[=:]?\s*)?(\d{1,5})(?:\s*[-\u2013\u2014]\s*(\d{1,5}))?\s*$",
    re.IGNORECASE,
)

_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)\s*$", re.IGNORECASE)

# Negation cues. All alternatives are word-bounded, so "notification" and
# "notable" do not read as "not".
_NEGATION_RE = re.compile(
    r"\bno\s+longer\b"
    r"|\bnot\b"
    r"|\b(?:does|do|is|are|was|were|has|have|can|could|would|should)n['\u2019]?t\b"
    r"|\binstead\s+of\b"
    r"|\bstopped\b",
    re.IGNORECASE,
)


def _text(value: object) -> str:
    """Coerce anything to text without raising."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    try:
        return str(value)
    except Exception:  # pathological __str__
        return ""


def _name(attr: object) -> str:
    return _text(attr).strip().lower()


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return None


def _capacity(text: str) -> tuple | None:
    match = _SIZE_RE.match(text)
    if not match:
        return None
    amount = _decimal(match.group(1))
    if amount is None or not amount.is_finite():
        return None
    total = amount * _SIZE_FACTORS[match.group(2).lower()]
    return (int(total.to_integral_value()),)


def _clock(text: str) -> tuple | None:
    match = _CLOCK_RE.match(text)
    if not match:
        return None
    amount = _decimal(match.group(1))
    if amount is None or not amount.is_finite():
        return None
    mhz = amount * _CLOCK_FACTORS[match.group(2).lower()]
    return (float(mhz),)


def _cores(text: str) -> tuple | None:
    match = _CORES_RE.match(text)
    if not match:
        return None
    return (int(match.group(1)),)


def _port(value: object) -> tuple | None:
    """Return ('single', n) or ('range', lo, hi); None when not a port."""
    match = _PORT_RE.match(_text(value).strip())
    if not match:
        return None
    lo = int(match.group(1))
    hi = int(match.group(2)) if match.group(2) is not None else None
    if lo > 65535 or (hi is not None and hi > 65535):
        return None
    if hi is None or hi == lo:
        return ("single", lo)
    if hi < lo:
        lo, hi = hi, lo
    return ("range", lo, hi)


def _version(text: str) -> tuple | None:
    match = _VERSION_RE.match(text)
    if not match:
        return None
    parts = [int(p) for p in match.group(1).split(".")]
    # 1.2 == 1.2.0; trailing zero components carry no information.
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def canonical(attr: str, value: str) -> tuple:
    """Normalise one value to a comparison tuple for its attribute.

    Recognised shapes: capacities (bytes), clock (MHz), cores (int), ports
    (single/range) and versions (int tuple). Everything else, including
    unparseable input, becomes ``(lowercased_stripped_string,)``.
    """
    text = _text(value).strip()
    name = _name(attr)

    parsed = None
    if name in CAPACITY_ATTRS:
        parsed = _capacity(text)
    elif name == "clock":
        parsed = _clock(text)
    elif name == "cores":
        parsed = _cores(text)
    elif name == "port":
        parsed = _port(text)
    elif name == "version":
        parsed = _version(text)

    if parsed is not None:
        return parsed
    return (text.lower(),)


def _bounds(port: tuple) -> tuple[int, int]:
    if port[0] == "single":
        return (port[1], port[1])
    return (port[1], port[2])


def ports_overlap(a: str, b: str) -> bool:
    """True when two port specs share at least one port number."""
    pa, pb = _port(a), _port(b)
    if pa is None or pb is None:
        return False
    a_lo, a_hi = _bounds(pa)
    b_lo, b_hi = _bounds(pb)
    return max(a_lo, b_lo) <= min(a_hi, b_hi)


def negated(text: str) -> bool:
    """True when the sentence carries an explicit negation cue."""
    return bool(_NEGATION_RE.search(_text(text)))


def conflicts(attr: str, a: str, b: str) -> bool:
    """True only when two values genuinely disagree for this attribute.

    Equal canonical values never conflict (``16GB`` vs ``16384MB``). For
    ports, only overlapping-but-different specs conflict; disjoint ports can
    both be true. Unknown attributes compare as case-insensitive strings.
    """
    if canonical(attr, a) == canonical(attr, b):
        return False
    if _name(attr) == "port":
        return ports_overlap(a, b)
    return True
