from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class EvalMetrics:
    n: int
    n_target: int
    n_nontarget: int
    auc: float
    eer: float
    eer_threshold: float
    confusion: tuple[int, int, int, int]  # (tp, fp, fn, tn) at the chosen threshold
    threshold: float


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.0
) -> EvalMetrics:
    n = len(scores)
    n_target = int((labels == 1).sum())
    n_nontarget = int((labels == 0).sum())

    if n_target == 0 or n_nontarget == 0:
        return EvalMetrics(
            n=n,
            n_target=n_target,
            n_nontarget=n_nontarget,
            auc=float("nan"),
            eer=float("nan"),
            eer_threshold=float("nan"),
            confusion=(0, 0, 0, 0),
            threshold=threshold,
        )

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_thr = float(thr[idx]) if idx < len(thr) else 0.0

    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    return EvalMetrics(
        n=n,
        n_target=n_target,
        n_nontarget=n_nontarget,
        auc=auc,
        eer=eer,
        eer_threshold=eer_thr,
        confusion=(tp, fp, fn, tn),
        threshold=threshold,
    )


def format_metrics(m: EvalMetrics) -> str:
    tp, fp, fn, tn = m.confusion
    pr = tp / (tp + fp) if (tp + fp) else float("nan")
    rc = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else float("nan")
    return (
        f"n={m.n} (target={m.n_target}, non-target={m.n_nontarget})\n"
        f"  AUC = {m.auc:.4f}\n"
        f"  EER = {m.eer * 100:.2f}%   (at threshold = {m.eer_threshold:+.3f})\n"
        f"  Confusion @ logit={m.threshold:+.2f}:  TP={tp} FP={fp} FN={fn} TN={tn}\n"
        f"  Precision = {pr:.3f}   Recall = {rc:.3f}   F1 = {f1:.3f}"
    )
