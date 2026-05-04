from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def config_hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(_canonical_repr(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _canonical_repr(obj: Any) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonical_repr(asdict(obj))
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return "{" + ",".join(f"{k!r}:{_canonical_repr(v)}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonical_repr(x) for x in obj) + "]"
    if isinstance(obj, Path):
        return f"Path({str(obj)!r})"
    return repr(obj)


def cache_key_path(root: Path, modality: str, cfg_hash: str, stem: str) -> Path:
    return root / modality / cfg_hash / f"{stem}.npy"


def get_or_compute(
    cache_path: Path,
    compute: Callable[[], np.ndarray],
    enabled: bool = True,
    source_path: Path | None = None,
) -> np.ndarray:
    if enabled and cache_path.is_file():
        if source_path is None or source_path.stat().st_mtime <= cache_path.stat().st_mtime:
            return np.load(cache_path)
    arr = compute()
    if enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # numpy.save always appends .npy; strip then let it re-add so cache_path
        # ends up at exactly its declared filename.
        np.save(cache_path.with_suffix(""), arr)
    return arr
