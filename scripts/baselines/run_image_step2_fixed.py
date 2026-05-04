#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.feature import hog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ikrlib import (  # noqa: E402
    png2fea,
    train_generative_linear_classifier,
    train_linear_logistic_regression,
)


def _hog_set(d: dict) -> np.ndarray:
    rows = []
    for v in d.values():
        if v.ndim == 3:
            v = v.sum(axis=2)
        v = v / max(float(v.max()), 1.0)
        rows.append(hog(
            v, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            transform_sqrt=False, feature_vector=True,
        ))
    return np.asarray(rows, dtype=np.float64)


def _eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> int:
    np.random.seed(0)
    train_t = _hog_set(png2fea("dataset/target_train"))
    train_n = _hog_set(png2fea("dataset/non_target_train"))
    test_t  = _hog_set(png2fea("dataset/target_dev"))
    test_n  = _hog_set(png2fea("dataset/non_target_dev"))

    X = np.r_[train_t, train_n]
    T = np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]

    # PCA(D=21) on HOG, mirroring the baseline's pixel-PCA scaffold.
    mean_vec = X.mean(axis=0)
    Xc = X - mean_vec
    _V, _S, U = np.linalg.svd(Xc, full_matrices=False)
    D = 21
    Xp = Xc.dot(U.T)[:, :D]
    test_t_p = (test_t - mean_vec).dot(U.T)[:, :D]
    test_n_p = (test_n - mean_vec).dot(U.T)[:, :D]
    print(f"HOG raw dim: {X.shape[1]}  ->  PCA D={D}  Xp={Xp.shape}")

    w, w0, _ = train_generative_linear_classifier(Xp, T)
    s = np.r_[test_t_p.dot(w) + w0, test_n_p.dot(w) + w0]
    y = np.r_[np.ones(len(test_t_p)), np.zeros(len(test_n_p))]
    auc_g = float(roc_auc_score(y, s)); eer_g = _eer(y, s)
    print(f"after generative LDA only:  AUC={auc_g:.4f}  EER={eer_g*100:.2f}%")

    for _ in range(2):
        w, w0 = train_linear_logistic_regression(Xp, T, w, w0)
    s = np.r_[test_t_p.dot(w) + w0, test_n_p.dot(w) + w0]
    auc_r = float(roc_auc_score(y, s)); eer_r = _eer(y, s)
    pred = (s >= 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    print(f"after 2 IRLS refinements:   AUC={auc_r:.4f}  EER={eer_r*100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}")

    out = Path("artifacts/baseline_image_step2_fixed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "HOG (2916-d) -> PCA(D=21) -> Gaussian LDA + 2 IRLS",
        "after_LDA": {"auc": auc_g, "eer": eer_g},
        "after_IRLS": {
            "auc": auc_r, "eer": eer_r,
            "confusion": [tp, fp, fn, tn],
        },
    }, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
