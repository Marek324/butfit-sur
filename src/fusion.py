"""Score-level multimodal fusion.

Strategy: train two unimodal classifiers on `target_train`/`non_target_train`,
then fit a 2-input logistic regression on stacked dev scores → labels. At
test time, score each eval sample with both classifiers and apply the fusion
LR. Logreg-on-top is the canonical score-level fusion choice — it learns the
optimal weighting between modalities and absorbs whatever scale each upstream
classifier produces.

Caveat: when `train_on="dev"` (the default), fusion is trained on the dev
split, so dev metrics REPORTED for the fusion system overestimate test
performance. The honest path is `train_on="cv_oof"` (out-of-fold CV scores)
but it requires retraining the per-modality models per fold; that's wired
through the experiments module rather than here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import ClassifierCfg, FusionCfg


@dataclass
class FusionModel:
    estimator: LogisticRegression
    score_normalize: str
    audio_mean: float
    audio_std: float
    image_mean: float
    image_std: float
    audio_artifact: str
    image_artifact: str


def fit_fusion(
    audio_scores: np.ndarray,
    image_scores: np.ndarray,
    y: np.ndarray,
    fusion_cfg: FusionCfg,
    classifier_cfg: ClassifierCfg | None = None,
) -> FusionModel:
    """Fit a 2-input logistic-regression fusion model on stacked scores."""
    cls = classifier_cfg or fusion_cfg.classifier
    if cls.type != "logreg":
        raise ValueError(
            f"fusion classifier must be logreg (got {cls.type!r}); SVM on top of "
            "two scalars is overkill"
        )

    a_mean, a_std = _norm_stats(audio_scores, fusion_cfg.score_normalize)
    i_mean, i_std = _norm_stats(image_scores, fusion_cfg.score_normalize)
    a_n = _apply_norm(audio_scores, a_mean, a_std)
    i_n = _apply_norm(image_scores, i_mean, i_std)
    X = np.column_stack([a_n, i_n])

    cw: str | None = cls.logreg.class_weight or None
    if cls.logreg.penalty == "l2":
        l1_ratio = 0.0
    elif cls.logreg.penalty == "l1":
        l1_ratio = 1.0
    else:
        raise ValueError(f"unsupported fusion penalty: {cls.logreg.penalty}")
    est = LogisticRegression(
        C=cls.logreg.C,
        class_weight=cw,
        l1_ratio=l1_ratio,
        max_iter=cls.logreg.max_iter,
    )
    est.fit(X, y)
    return FusionModel(
        estimator=est,
        score_normalize=fusion_cfg.score_normalize,
        audio_mean=a_mean,
        audio_std=a_std,
        image_mean=i_mean,
        image_std=i_std,
        audio_artifact=fusion_cfg.audio_artifact,
        image_artifact=fusion_cfg.image_artifact,
    )


def score_fusion(
    model: FusionModel, audio_scores: np.ndarray, image_scores: np.ndarray
) -> np.ndarray:
    """Real-valued fusion score, monotonic with P(target=1|audio,image)."""
    a_n = _apply_norm(audio_scores, model.audio_mean, model.audio_std)
    i_n = _apply_norm(image_scores, model.image_mean, model.image_std)
    X = np.column_stack([a_n, i_n])
    return np.asarray(model.estimator.decision_function(X), dtype=np.float64)


def save(model: FusionModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load(path: Path) -> FusionModel:
    with path.open("rb") as f:
        return pickle.load(f)


def _norm_stats(s: np.ndarray, mode: str) -> tuple[float, float]:
    if mode == "none":
        return 0.0, 1.0
    if mode == "zscore":
        mean = float(np.mean(s))
        std = float(np.std(s))
        return mean, (std if std > 1e-9 else 1.0)
    raise ValueError(f"unknown score_normalize mode: {mode}")


def _apply_norm(s: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(s, dtype=np.float64) - mean) / std
