"""Generic classifier with persist/load — handles logreg and SVM.

Both `LogisticRegression` and `SVC` expose `decision_function`, which returns
a real-valued score monotonic with P(target=1|x) and aligned so that
score >= 0 ⇔ predicted target. That makes the assignment-mandated 50/50-prior
decision threshold equal to 0 regardless of classifier choice.

Note on score semantics:
  - logreg: decision_function = log-odds (logit). Calibrated under standard
    LR assumptions; threshold=0 ⇔ P(target)=0.5.
  - svm:    decision_function = signed distance to the hyperplane. NOT a
    probability; threshold=0 is still the trained boundary, but absolute
    score magnitudes are not comparable across SVMs with different C.

For score-level fusion the fusion logreg absorbs whatever scale each upstream
classifier produces, so this asymmetry is benign.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import ClassifierCfg


@dataclass
class Model:
    estimator: LogisticRegression | SVC
    feature_dim: int
    feature_names: list[str]
    type: str


def fit(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cfg: ClassifierCfg
) -> Model:
    if cfg.type == "logreg":
        cw: str | None = cfg.logreg.class_weight or None
        # sklearn 1.8 deprecated `penalty=`; translate to `l1_ratio` to silence
        # the warning and stay forward-compatible with future versions.
        if cfg.logreg.penalty == "l2":
            l1_ratio: float = 0.0
        elif cfg.logreg.penalty == "l1":
            l1_ratio = 1.0
        elif cfg.logreg.penalty == "elasticnet":
            l1_ratio = 0.5
        else:
            raise ValueError(f"unsupported logreg penalty: {cfg.logreg.penalty}")
        est: LogisticRegression | SVC = LogisticRegression(
            C=cfg.logreg.C,
            class_weight=cw,
            l1_ratio=l1_ratio,
            max_iter=cfg.logreg.max_iter,
        )
    elif cfg.type in ("svm_rbf", "svm_linear"):
        cw = cfg.svm.class_weight or None
        kernel = "rbf" if cfg.type == "svm_rbf" else "linear"
        gamma_raw = cfg.svm.gamma
        gamma: str | float
        if gamma_raw in ("scale", "auto"):
            gamma = gamma_raw
        else:
            gamma = float(gamma_raw)
        est = SVC(C=cfg.svm.C, kernel=kernel, gamma=gamma, class_weight=cw)
    else:
        raise ValueError(f"unsupported classifier type: {cfg.type}")

    est.fit(X, y)
    return Model(
        estimator=est,
        feature_dim=X.shape[1],
        feature_names=feature_names,
        type=cfg.type,
    )


def score(model: Model, X: np.ndarray) -> np.ndarray:
    """Real-valued score; higher ⇒ more target. Aligned with class label 1."""
    return np.asarray(model.estimator.decision_function(X), dtype=np.float64)


def save(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load(path: Path) -> Model:
    with path.open("rb") as f:
        return pickle.load(f)
