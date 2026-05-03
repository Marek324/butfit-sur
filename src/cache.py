"""Per-sample feature cache.

Goal: amortize MFCC/HOG extraction across runs that vary only the classifier.
The cache key is (modality, feature-config hash, sample stem). Cache misses
fall through to the supplied compute function.

Cache entries are also keyed against the source file's mtime: if the source
file is newer than the cached entry (e.g. after re-running augment_dataset.py
with a different envelope), the entry is invalidated and recomputed.

Layout: <root>/<modality>/<config_hash>/<stem>.npy
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def config_hash(*parts: Any) -> str:
    """Stable 12-char hash of one or more dataclass/primitive structures.

    Uses a deterministic repr; tuples and lists hash the same way regardless
    of which one was used in the schema.
    """
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
    """Return cached array if present and fresh, otherwise compute and persist.

    Freshness: if `source_path` is given and its mtime is newer than the cache
    entry's mtime, the cached value is treated as stale. Lets the offline
    augmentation script re-write `<stem>_aug{kk}.{wav,png}` without leaving
    callers reading old features.
    """
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
