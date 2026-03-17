"""I/O helpers: JSON/JSONL read/write, run-ID generation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def generate_run_id() -> str:
    return uuid.uuid4().hex[:12]


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_default_serializer) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _default_serializer(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
