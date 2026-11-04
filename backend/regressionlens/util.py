"""Small helpers shared across the package: canonical hashing, seeds, ids."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import UTC, date, datetime
from typing import Any


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace. Equal content -> equal string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Content address of any JSON-able object, e.g. ``sha256:3f2a...``."""
    return "sha256:" + sha256_hex(canonical_json(obj))


def derive_seed(*parts: Any) -> int:
    """A stable 31-bit seed derived from arbitrary parts (run seed, case key, repeat index...)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def new_run_seed() -> int:
    return random.SystemRandom().randrange(1, 2**31 - 1)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted paths (lists are kept as leaf values)."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict) and value:
                out.update(flatten(value, path))
            else:
                out[path] = value
    else:
        out[prefix] = obj
    return out
