from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupKFold


class _HasSession(Protocol):
    name: str
    session_id: str | None


@dataclass(frozen=True)
class CvFold:
    fold_idx: int
    train_idx: np.ndarray  # int64 indices into the input list
    dev_idx: np.ndarray


def session_groups(samples: list[_HasSession]) -> list[str]:
    return [(s.session_id or s.name) for s in samples]


def session_grouped_folds(samples: list[_HasSession], n_folds: int) -> list[CvFold]:
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    groups = session_groups(samples)
    n_groups = len(set(groups))
    k = min(n_folds, n_groups)
    if k < 2:
        raise ValueError(
            f"need at least 2 distinct sessions for grouped CV, got {n_groups}"
        )
    splitter = GroupKFold(n_splits=k)
    out: list[CvFold] = []
    for i, (tr, dv) in enumerate(splitter.split(np.zeros(len(samples)), groups=groups)):
        out.append(
            CvFold(
                fold_idx=i,
                train_idx=np.asarray(tr, dtype=np.int64),
                dev_idx=np.asarray(dv, dtype=np.int64),
            )
        )
    return out


def iter_session_grouped(samples: list[_HasSession], n_folds: int) -> Iterator[CvFold]:
    yield from session_grouped_folds(samples, n_folds)
