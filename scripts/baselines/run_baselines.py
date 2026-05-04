#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import randint
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ikrlib import (  # noqa: E402
    png2fea,
    wav16khz2mfcc,
    train_gmm,
    logpdf_gmm,
    train_generative_linear_classifier,
    train_linear_logistic_regression,
)


def _eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def _confusion(y: np.ndarray, s: np.ndarray, thr: float = 0.0):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _print_metrics(name: str, y: np.ndarray, s: np.ndarray, thr: float = 0.0) -> dict:
    auc = float(roc_auc_score(y, s))
    eer = _eer(y, s)
    tp, fp, fn, tn = _confusion(y, s, thr)
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else float("nan")
    print(f"\n{name}")
    print(f"  AUC = {auc:.4f}")
    print(f"  EER = {eer*100:.2f}%")
    print(f"  Confusion @ score>={thr}:  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision = {p:.3f}   Recall = {r:.3f}   F1 = {f1:.3f}")
    return {"auc": auc, "eer": eer, "confusion": [tp, fp, fn, tn],
            "precision": p, "recall": r, "f1": (f1 if f1 == f1 else None)}


# ---------------------------- IMAGE ----------------------------

def _flatten_pngs(d: dict) -> np.ndarray:
    rows = []
    for v in d.values():
        if v.ndim == 3:
            v = v.sum(axis=2)
        rows.append(v.reshape(-1))
    return np.asarray(rows, dtype=np.float64)


def run_image_baseline(seed: int = 0) -> dict:
    np.random.seed(seed)
    print("=== Image baseline (PCA D=21 + Gaussian LDA + IRLS refinement) ===")
    train_t = _flatten_pngs(png2fea("dataset/target_train"))
    train_n = _flatten_pngs(png2fea("dataset/non_target_train"))
    test_t  = _flatten_pngs(png2fea("dataset/target_dev"))
    test_n  = _flatten_pngs(png2fea("dataset/non_target_dev"))
    print(f"  shapes: train_t={train_t.shape} train_n={train_n.shape} "
          f"test_t={test_t.shape} test_n={test_n.shape}")

    X = np.r_[train_t, train_n]
    T = np.r_[np.ones(len(train_t)), np.zeros(len(train_n))]

    mean_face = np.mean(X, axis=0)
    Xc = X - mean_face
    V, S, U = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc.dot(U.T)
    test_t_p = (test_t - mean_face).dot(U.T)
    test_n_p = (test_n - mean_face).dot(U.T)

    D = 21
    w, w0, _ = train_generative_linear_classifier(Xp[:, :D], T)

    out = {}

    s_t = test_t_p[:, :D].dot(w) + w0
    s_n = test_n_p[:, :D].dot(w) + w0
    y = np.r_[np.ones(len(s_t)), np.zeros(len(s_n))]
    s = np.r_[s_t, s_n]
    out["generative_only"] = _print_metrics("after generative LDA only", y, s)

    for it in range(2):
        w, w0 = train_linear_logistic_regression(Xp[:, :D], T, w, w0)
    s_t = test_t_p[:, :D].dot(w) + w0
    s_n = test_n_p[:, :D].dot(w) + w0
    s = np.r_[s_t, s_n]
    out["after_2_logreg_refinements"] = _print_metrics("after 2 IRLS refinements", y, s)
    return out


# ---------------------------- AUDIO ----------------------------

def run_audio_baseline(seed: int = 0, n_iter: int = 40, M: int = 30) -> dict:
    np.random.seed(seed)
    print(f"\n=== Audio baseline (13-d MFCC + two GMMs M={M}, {n_iter} EM iters) ===")
    train_t_dict = wav16khz2mfcc("dataset/target_train")
    train_n_dict = wav16khz2mfcc("dataset/non_target_train")
    test_t_dict  = wav16khz2mfcc("dataset/target_dev")
    test_n_dict  = wav16khz2mfcc("dataset/non_target_dev")

    train_t = np.vstack(list(train_t_dict.values()))
    train_n = np.vstack(list(train_n_dict.values()))
    print(f"  train target frames: {train_t.shape}")
    print(f"  train non-target frames: {train_n.shape}")

    M_t = M
    MUs_t = train_t[randint(1, len(train_t), M_t)]
    COVs_t = [np.var(train_t, axis=0)] * M_t
    Ws_t = np.ones(M_t) / M_t

    M_n = M
    MUs_n = train_n[randint(1, len(train_n), M_n)]
    COVs_n = [np.var(train_n, axis=0)] * M_n
    Ws_n = np.ones(M_n) / M_n

    t0 = time.time()
    for jj in range(n_iter):
        Ws_t, MUs_t, COVs_t, _ = train_gmm(train_t, Ws_t, MUs_t, COVs_t)
        Ws_n, MUs_n, COVs_n, _ = train_gmm(train_n, Ws_n, MUs_n, COVs_n)
    print(f"  {n_iter} EM iterations took {time.time() - t0:.1f}s")

    P_t = 0.5
    P_n = 1.0 - P_t

    def _score_dict(d):
        out = []
        for tst in d.values():
            ll_t = logpdf_gmm(tst, Ws_t, MUs_t, COVs_t)
            ll_n = logpdf_gmm(tst, Ws_n, MUs_n, COVs_n)
            out.append((sum(ll_t) + np.log(P_t)) - (sum(ll_n) + np.log(P_n)))
        return np.asarray(out, dtype=np.float64)

    s_t = _score_dict(test_t_dict)
    s_n = _score_dict(test_n_dict)
    y = np.r_[np.ones(len(s_t)), np.zeros(len(s_n))]
    s = np.r_[s_t, s_n]
    return _print_metrics("audio GMM baseline", y, s)


def main() -> int:
    out = {}
    out["image"] = run_image_baseline()
    out["audio"] = run_audio_baseline()
    out_path = Path("artifacts/baselines_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
