
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.feature import hog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ikrlib import png2fea  


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
    print(f"X={X.shape} positives={int(T.sum())}")

    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    clf.fit(X, T)

    s_t = clf.decision_function(test_t)
    s_n = clf.decision_function(test_n)
    y = np.r_[np.ones(len(s_t)), np.zeros(len(s_n))]
    s = np.r_[s_t, s_n]
    auc = float(roc_auc_score(y, s))
    eer = _eer(y, s)
    pred = (s >= 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    print(f"AUC={auc:.4f}  EER={eer*100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}")

    out = Path("artifacts/baseline_image_step3.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "HOG + balanced LogReg C=0.1 (no augmentation)",
        "auc": auc, "eer": eer, "confusion": [tp, fp, fn, tn],
    }, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
