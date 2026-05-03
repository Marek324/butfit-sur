"""Session-grouped K-fold cross-validation.

Instructor explicitly recommends a CV scheme that respects recording sessions
(`<speaker>_<session>` from the filename) — same-session samples are highly
correlated, so letting them straddle a fold boundary inflates dev metrics.

We use sklearn.GroupKFold with `session_id` as the grouping key. For samples
with no parsable session_id we fall back to using the filename stem (each
sample its own group), which keeps them disjoint by accident but does not
affect the leakage guarantee.

Note on stratification: GroupKFold does not stratify by class. With only ~20
target sessions for m431 across train+dev (≈10 sessions × 2 takes), targets
distribute roughly evenly into 3-5 folds; non-targets dominate and have many
distinct sessions so their distribution is automatically uniform. If a fold
ends up with 0 targets in either side, downstream evaluation guards against
NaN AUC/EER.
"""

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
    """Return per-sample group labels — session_id, or filename stem if missing."""
    return [(s.session_id or s.name) for s in samples]


def session_grouped_folds(samples: list[_HasSession], n_folds: int) -> list[CvFold]:
    """Materialize all (train, dev) index pairs for session-grouped K-fold.

    Returned indices are positions into `samples`. K is clipped to the number
    of distinct groups (sklearn would otherwise raise).
    """
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
    """Generator-flavored helper for code that prefers iteration."""
    yield from session_grouped_folds(samples, n_folds)
