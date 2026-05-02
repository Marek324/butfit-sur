"""Logistic-regression classifier with persist/load.

class_weight='balanced' centers the bias term so σ(wx+b) reflects equal
priors regardless of training class counts; threshold at logit=0 is then
the Bayes-optimal decision under the assignment's stated 50/50 prior.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import ClassifierCfg


@dataclass
class Model:
    estimator: LogisticRegression
    feature_dim: int
    feature_names: list[str]


def fit(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cfg: ClassifierCfg
) -> Model:
    if cfg.type != "logreg":
        raise ValueError(f"unsupported classifier type: {cfg.type}")
    cw: str | None = cfg.logreg.class_weight or None
    est = LogisticRegression(
        C=cfg.logreg.C,
        class_weight=cw,
        penalty=cfg.logreg.penalty,
        max_iter=cfg.logreg.max_iter,
    )
    est.fit(X, y)
    return Model(estimator=est, feature_dim=X.shape[1], feature_names=feature_names)


def score(model: Model, X: np.ndarray) -> np.ndarray:
    """Logit score (linear w·x + b), monotonic with P(target=1|x).

    sklearn's decision_function returns the score for the positive class,
    which is class label 1 (target) here, so higher = more m431.
    """
    return np.asarray(model.estimator.decision_function(X), dtype=np.float64)


def save(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load(path: Path) -> Model:
    with path.open("rb") as f:
        return pickle.load(f)
