#!/usr/bin/env python3
"""
Audio progression step 2: course's MFCC + simple energy VAD + per-utterance
CMVN, then the same 2-GMM (M=30, 40 EM iters) classifier as the teacher
baseline. Isolates the preprocessing change at fixed model architecture.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import randint
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ikrlib import (  # noqa: E402
    mfcc as course_mfcc,
    train_gmm,
    logpdf_gmm,
    framing,
)


WINDOW = 400
NOVERLAP = 240
NFFT = 512
FS = 16000
NBANKS = 23
NCEPS = 13
HOP = WINDOW - NOVERLAP


def _frame_log_energy(s: np.ndarray) -> np.ndarray:
    """Per-frame log-energy aligned to course mfcc framing."""
    if s.shape[0] < WINDOW:
        return np.zeros(0)
    fr = framing(s.astype(np.float64), WINDOW, HOP)
    e = (fr ** 2).mean(axis=1)
    e = np.maximum(e, 1e-12)
    return 10.0 * np.log10(e)


def _vad_mask(s: np.ndarray, margin_db: float = 8.0, noise_pct: float = 10.0) -> np.ndarray:
    """Energy-relative VAD on course-aligned frames."""
    e = _frame_log_energy(s)
    if e.size == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(e, noise_pct))
    return e > (floor + margin_db)


def _extract(dir_name: str, apply_vad: bool, apply_cmvn: bool) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    from glob import glob
    for f in sorted(glob(f"{dir_name}/*.wav")):
        rate, s = wavfile.read(f)
        assert rate == FS
        feats = course_mfcc(s, WINDOW, NOVERLAP, NFFT, FS, NBANKS, NCEPS)
        if apply_vad:
            mask = _vad_mask(s)
            n = min(len(mask), len(feats))
            feats = feats[:n][mask[:n]]
        if apply_cmvn and len(feats) > 0:
            feats = feats - feats.mean(axis=0, keepdims=True)
        out[f] = feats
    return out


def main() -> int:
    np.random.seed(0)
    print("Extracting course MFCCs with VAD + CMVN ...")
    t0 = time.time()
    train_t = _extract("dataset/target_train", apply_vad=True, apply_cmvn=True)
    train_n = _extract("dataset/non_target_train", apply_vad=True, apply_cmvn=True)
    test_t  = _extract("dataset/target_dev", apply_vad=True, apply_cmvn=True)
    test_n  = _extract("dataset/non_target_dev", apply_vad=True, apply_cmvn=True)
    print(f"  extraction took {time.time() - t0:.1f}s")

    train_t_stack = np.vstack([v for v in train_t.values() if v.shape[0] > 0])
    train_n_stack = np.vstack([v for v in train_n.values() if v.shape[0] > 0])
    print(f"  train target frames: {train_t_stack.shape}")
    print(f"  train non-target frames: {train_n_stack.shape}")

    M = 30
    MUs_t  = train_t_stack[randint(1, len(train_t_stack), M)]
    COVs_t = [np.var(train_t_stack, axis=0)] * M
    Ws_t   = np.ones(M) / M
    MUs_n  = train_n_stack[randint(1, len(train_n_stack), M)]
    COVs_n = [np.var(train_n_stack, axis=0)] * M
    Ws_n   = np.ones(M) / M

    print(f"Running 40 EM iterations (M={M}) ...")
    t0 = time.time()
    for _ in range(40):
        Ws_t, MUs_t, COVs_t, _ttl = train_gmm(train_t_stack, Ws_t, MUs_t, COVs_t)
        Ws_n, MUs_n, COVs_n, _ttl = train_gmm(train_n_stack, Ws_n, MUs_n, COVs_n)
    print(f"  EM took {time.time() - t0:.1f}s")

    P_t = 0.5
    P_n = 1.0 - P_t
    SENTINEL = -10.0  # for VAD-empty utterances

    def _score_dict(d):
        out = []
        for v in d.values():
            if v.shape[0] == 0:
                out.append(SENTINEL)
                continue
            ll_t = logpdf_gmm(v, Ws_t, MUs_t, COVs_t)
            ll_n = logpdf_gmm(v, Ws_n, MUs_n, COVs_n)
            out.append(float((sum(ll_t) + np.log(P_t)) - (sum(ll_n) + np.log(P_n))))
        return np.asarray(out, dtype=np.float64)

    s_t = _score_dict(test_t)
    s_n = _score_dict(test_n)
    y = np.r_[np.ones(len(s_t)), np.zeros(len(s_n))]
    s = np.r_[s_t, s_n]
    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    pred = (s >= 0).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    print(f"\nAUC={auc:.4f}  EER={eer*100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}")

    out = Path("artifacts/baseline_audio_step2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "course MFCC + VAD + CMVN, 2 GMMs M=30",
        "auc": auc, "eer": eer, "confusion": [tp, fp, fn, tn],
    }, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
