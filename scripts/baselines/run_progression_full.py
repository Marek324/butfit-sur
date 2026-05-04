#!/usr/bin/env python3
import json
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
from numpy.random import randint
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.feature import hog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ikrlib import (  # noqa: E402
    png2fea,
    train_gmm,
    logpdf_gmm,
    train_generative_linear_classifier,
    train_linear_logistic_regression,
    mfcc as course_mfcc,
    framing,
)


# ---------------- shared metric helpers ----------------

def _eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def _metrics(y, s, thr=0.0):
    auc = float(roc_auc_score(y, s))
    eer = _eer(y, s)
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"auc": auc, "eer": eer, "confusion": [tp, fp, fn, tn]}


def _eval_count(s, thr=0.0) -> int:
    return int((s >= thr).sum())


# ---------------- IMAGE ----------------

def _to_gray80(v):
    from skimage.transform import resize
    if v.ndim == 3:
        v = v.sum(axis=2)
    if v.shape != (80, 80):
        v = resize(v, (80, 80), anti_aliasing=True)
    return v


def _load_pixels(d):
    return np.asarray(
        [_to_gray80(v).reshape(-1) for v in d.values()], dtype=np.float64
    )


def _load_hog(d):
    rows = []
    for v in d.values():
        v = _to_gray80(v)
        v = v / max(float(v.max()), 1.0)
        rows.append(hog(
            v, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            transform_sqrt=False, feature_vector=True,
        ))
    return np.asarray(rows, dtype=np.float64)


def _pca_fit_apply(X_train, *Xs, D=21):
    mean = X_train.mean(axis=0)
    Xc = X_train - mean
    _V, _S, U = np.linalg.svd(Xc, full_matrices=False)
    proj = lambda M: (M - mean).dot(U.T)[:, :D]
    return proj(X_train), [proj(M) for M in Xs]


def _train_baseline_classifier(Xp, T):
    w, w0, _ = train_generative_linear_classifier(Xp, T)
    for _ in range(2):
        w, w0 = train_linear_logistic_regression(Xp, T, w, w0)
    return w, w0


def run_image_steps():
    print("\n========== IMAGE ==========\n")
    np.random.seed(0)

    # Enumerate (preserve filename ordering for eval count traceability)
    splits = {
        "train_t": "dataset/target_train",
        "train_n": "dataset/non_target_train",
        "dev_t":   "dataset/target_dev",
        "dev_n":   "dataset/non_target_dev",
        "stress_t":"dataset/target_dev_stress",
        "stress_n":"dataset/non_target_dev_stress",
        "eval":    "dataset/eval",
    }

    print("Loading PNGs ...")
    t0 = time.time()
    raw = {k: png2fea(v) for k, v in splits.items()}
    print(f"  loaded {sum(len(v) for v in raw.values())} images in {time.time()-t0:.1f}s")

    pix = {k: _load_pixels(v) for k, v in raw.items()}
    h   = {k: _load_hog(v) for k, v in raw.items()}

    out = {}

    # Step 1: pixel-PCA(21) + LDA + IRLS
    X_tr  = np.r_[pix["train_t"], pix["train_n"]]
    T_tr  = np.r_[np.ones(len(pix["train_t"])), np.zeros(len(pix["train_n"]))]
    X_dv_t, X_dv_n = pix["dev_t"], pix["dev_n"]
    X_st_t, X_st_n = pix["stress_t"], pix["stress_n"]
    X_ev = pix["eval"]

    Xp_tr, [Xp_dv_t, Xp_dv_n, Xp_st_t, Xp_st_n, Xp_ev] = _pca_fit_apply(
        X_tr, X_dv_t, X_dv_n, X_st_t, X_st_n, X_ev, D=21,
    )
    w, w0 = _train_baseline_classifier(Xp_tr, T_tr)

    s_dv = np.r_[Xp_dv_t.dot(w) + w0, Xp_dv_n.dot(w) + w0]
    y_dv = np.r_[np.ones(len(Xp_dv_t)), np.zeros(len(Xp_dv_n))]
    s_st = np.r_[Xp_st_t.dot(w) + w0, Xp_st_n.dot(w) + w0]
    y_st = np.r_[np.ones(len(Xp_st_t)), np.zeros(len(Xp_st_n))]
    s_ev = Xp_ev.dot(w) + w0
    out["step1_pixel_PCA21_LDA_IRLS"] = {
        "clean": _metrics(y_dv, s_dv),
        "stress": _metrics(y_st, s_st),
        "eval_pred_target": _eval_count(s_ev),
        "eval_n": int(len(s_ev)),
    }

    # Step 2: HOG + PCA(21) + LDA + IRLS
    X_tr2 = np.r_[h["train_t"], h["train_n"]]
    Xp_tr2, [Xp_dv_t2, Xp_dv_n2, Xp_st_t2, Xp_st_n2, Xp_ev2] = _pca_fit_apply(
        X_tr2, h["dev_t"], h["dev_n"], h["stress_t"], h["stress_n"], h["eval"], D=21,
    )
    w, w0 = _train_baseline_classifier(Xp_tr2, T_tr)
    s_dv2 = np.r_[Xp_dv_t2.dot(w) + w0, Xp_dv_n2.dot(w) + w0]
    s_st2 = np.r_[Xp_st_t2.dot(w) + w0, Xp_st_n2.dot(w) + w0]
    s_ev2 = Xp_ev2.dot(w) + w0
    out["step2_HOG_PCA21_LDA_IRLS"] = {
        "clean": _metrics(y_dv, s_dv2),
        "stress": _metrics(y_st, s_st2),
        "eval_pred_target": _eval_count(s_ev2),
        "eval_n": int(len(s_ev2)),
    }

    # Step 3: HOG (full 2916-d) + balanced LogReg, no aug
    Xfull_tr = np.r_[h["train_t"], h["train_n"]]
    clf3 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    clf3.fit(Xfull_tr, T_tr)
    s_dv3 = np.r_[clf3.decision_function(h["dev_t"]),  clf3.decision_function(h["dev_n"])]
    s_st3 = np.r_[clf3.decision_function(h["stress_t"]), clf3.decision_function(h["stress_n"])]
    s_ev3 = clf3.decision_function(h["eval"])
    out["step3_HOG_LogReg_balanced_noaug"] = {
        "clean": _metrics(y_dv, s_dv3),
        "stress": _metrics(y_st, s_st3),
        "eval_pred_target": _eval_count(s_ev3),
        "eval_n": int(len(s_ev3)),
    }

    # Step 4: production - load already-trained model, score eval/dev/stress
    # We can read existing eval result file for the eval count
    sys.path.insert(0, str(HERE.parent.parent / "src"))
    try:
        from classifier import load as load_classifier  # noqa: E402
        prod = load_classifier(Path("artifacts/image_baseline_logreg/model.pkl"))
        s_dv4 = np.r_[prod.estimator.decision_function(h["dev_t"]), prod.estimator.decision_function(h["dev_n"])]
        s_st4 = np.r_[prod.estimator.decision_function(h["stress_t"]), prod.estimator.decision_function(h["stress_n"])]
        s_ev4 = prod.estimator.decision_function(h["eval"])
        out["step4_production_HOG_LogReg_balanced_aug"] = {
            "clean": _metrics(y_dv, s_dv4),
            "stress": _metrics(y_st, s_st4),
            "eval_pred_target": _eval_count(s_ev4),
            "eval_n": int(len(s_ev4)),
        }
    except Exception as e:
        print(f"  (could not load production image model: {e})")
        out["step4_production_HOG_LogReg_balanced_aug"] = {"error": str(e)}

    return out


# ---------------- AUDIO ----------------

WINDOW = 400
NOVERLAP = 240
NFFT = 512
FS = 16000
NBANKS = 23
NCEPS = 13
HOP = WINDOW - NOVERLAP


def _frame_log_energy(s):
    if s.shape[0] < WINDOW:
        return np.zeros(0)
    fr = framing(s.astype(np.float64), WINDOW, HOP)
    e = (fr ** 2).mean(axis=1)
    e = np.maximum(e, 1e-12)
    return 10.0 * np.log10(e)


def _vad_mask(s, margin_db=8.0, noise_pct=10.0):
    e = _frame_log_energy(s)
    if e.size == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(e, noise_pct))
    return e > (floor + margin_db)


def _extract_audio(dir_name, apply_vad, apply_cmvn):
    out = {}
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


def _train_two_gmm(train_t, train_n, M=30, n_iter=40):
    np.random.seed(0)
    train_t = np.vstack([v for v in train_t.values() if v.shape[0] > 0])
    train_n = np.vstack([v for v in train_n.values() if v.shape[0] > 0])
    MUs_t  = train_t[randint(1, len(train_t), M)]
    COVs_t = [np.var(train_t, axis=0)] * M
    Ws_t   = np.ones(M) / M
    MUs_n  = train_n[randint(1, len(train_n), M)]
    COVs_n = [np.var(train_n, axis=0)] * M
    Ws_n   = np.ones(M) / M
    for _ in range(n_iter):
        Ws_t, MUs_t, COVs_t, _ = train_gmm(train_t, Ws_t, MUs_t, COVs_t)
        Ws_n, MUs_n, COVs_n, _ = train_gmm(train_n, Ws_n, MUs_n, COVs_n)
    return (Ws_t, MUs_t, COVs_t), (Ws_n, MUs_n, COVs_n)


def _score_two_gmm(d, gmm_t, gmm_n, sentinel=-10.0):
    Ws_t, MUs_t, COVs_t = gmm_t
    Ws_n, MUs_n, COVs_n = gmm_n
    out = []
    for v in d.values():
        if v.shape[0] == 0:
            out.append(sentinel)
            continue
        ll_t = logpdf_gmm(v, Ws_t, MUs_t, COVs_t)
        ll_n = logpdf_gmm(v, Ws_n, MUs_n, COVs_n)
        out.append(float(sum(ll_t) + np.log(0.5) - sum(ll_n) - np.log(0.5)))
    return np.asarray(out, dtype=np.float64)


def run_audio_steps():
    print("\n========== AUDIO ==========\n")
    out = {}

    # Step 1: course MFCC + 2 GMMs M=30 (no VAD, no CMVN)
    print("Step 1: course MFCC, no VAD, no CMVN ...")
    t0 = time.time()
    tr_t = _extract_audio("dataset/target_train", apply_vad=False, apply_cmvn=False)
    tr_n = _extract_audio("dataset/non_target_train", apply_vad=False, apply_cmvn=False)
    dv_t = _extract_audio("dataset/target_dev", apply_vad=False, apply_cmvn=False)
    dv_n = _extract_audio("dataset/non_target_dev", apply_vad=False, apply_cmvn=False)
    st_t = _extract_audio("dataset/target_dev_stress", apply_vad=False, apply_cmvn=False)
    st_n = _extract_audio("dataset/non_target_dev_stress", apply_vad=False, apply_cmvn=False)
    ev   = _extract_audio("dataset/eval", apply_vad=False, apply_cmvn=False)
    print(f"  features in {time.time()-t0:.1f}s")
    print("  training GMMs ...")
    t0 = time.time()
    g_t1, g_n1 = _train_two_gmm(tr_t, tr_n)
    print(f"  GMMs in {time.time()-t0:.1f}s")
    s_dv1 = np.r_[_score_two_gmm(dv_t, g_t1, g_n1), _score_two_gmm(dv_n, g_t1, g_n1)]
    y_dv  = np.r_[np.ones(len(dv_t)), np.zeros(len(dv_n))]
    s_st1 = np.r_[_score_two_gmm(st_t, g_t1, g_n1), _score_two_gmm(st_n, g_t1, g_n1)]
    y_st  = np.r_[np.ones(len(st_t)), np.zeros(len(st_n))]
    s_ev1 = _score_two_gmm(ev, g_t1, g_n1)
    out["step1_courseMFCC_2GMM"] = {
        "clean": _metrics(y_dv, s_dv1),
        "stress": _metrics(y_st, s_st1),
        "eval_pred_target": _eval_count(s_ev1),
        "eval_n": int(len(s_ev1)),
    }

    # Step 2: course MFCC + VAD + CMVN
    print("Step 2: course MFCC + VAD + CMVN ...")
    t0 = time.time()
    tr_t2 = _extract_audio("dataset/target_train", apply_vad=True, apply_cmvn=True)
    tr_n2 = _extract_audio("dataset/non_target_train", apply_vad=True, apply_cmvn=True)
    dv_t2 = _extract_audio("dataset/target_dev", apply_vad=True, apply_cmvn=True)
    dv_n2 = _extract_audio("dataset/non_target_dev", apply_vad=True, apply_cmvn=True)
    st_t2 = _extract_audio("dataset/target_dev_stress", apply_vad=True, apply_cmvn=True)
    st_n2 = _extract_audio("dataset/non_target_dev_stress", apply_vad=True, apply_cmvn=True)
    ev2   = _extract_audio("dataset/eval", apply_vad=True, apply_cmvn=True)
    print(f"  features in {time.time()-t0:.1f}s")
    print("  training GMMs ...")
    t0 = time.time()
    g_t2, g_n2 = _train_two_gmm(tr_t2, tr_n2)
    print(f"  GMMs in {time.time()-t0:.1f}s")
    s_dv2 = np.r_[_score_two_gmm(dv_t2, g_t2, g_n2), _score_two_gmm(dv_n2, g_t2, g_n2)]
    s_st2 = np.r_[_score_two_gmm(st_t2, g_t2, g_n2), _score_two_gmm(st_n2, g_t2, g_n2)]
    s_ev2 = _score_two_gmm(ev2, g_t2, g_n2)
    out["step2_courseMFCC_VAD_CMVN_2GMM"] = {
        "clean": _metrics(y_dv, s_dv2),
        "stress": _metrics(y_st, s_st2),
        "eval_pred_target": _eval_count(s_ev2),
        "eval_n": int(len(s_ev2)),
    }

    # Step 3: production GMM-UBM (load already-computed metrics + eval count)
    prod_metrics = json.loads(Path("artifacts/audio_gmm_ubm_map_free/metrics.json").read_text())
    out["step3_production_GMM_UBM_map_free"] = {
        "clean": prod_metrics["dev"],
        "stress": prod_metrics.get("stress_dev"),
        "eval_pred_target": prod_metrics["eval"]["predicted_target"],
        "eval_n": prod_metrics["eval"]["n"],
    }
    return out


def _print_table(results, modality):
    print(f"\n========== {modality.upper()} TABLE ==========")
    print(f"{'Step':<48s}  {'Clean AUC':>9s}  {'Clean EER':>9s}  {'Stress AUC':>10s}  {'Stress EER':>10s}  {'Eval target':>11s}")
    for name, r in results.items():
        cl = r.get("clean", {}) if isinstance(r, dict) else {}
        st = r.get("stress", {}) if isinstance(r, dict) else {}
        ev = r.get("eval_pred_target", "—")
        ev_n = r.get("eval_n", "—")
        ev_str = f"{ev}/{ev_n}" if ev_n != "—" else str(ev)
        print(
            f"{name:<48s}  "
            f"{cl.get('auc', float('nan')):>9.4f}  "
            f"{(cl.get('eer', float('nan')) or 0)*100:>8.2f}%  "
            f"{(st.get('auc', float('nan')) if st else float('nan')):>10.4f}  "
            f"{((st.get('eer', float('nan')) or 0) if st else 0)*100:>9.2f}%  "
            f"{ev_str:>11s}"
        )


def main() -> int:
    img = run_image_steps()
    aud = run_audio_steps()
    out = {"image": img, "audio": aud}
    out_path = Path("artifacts/baseline_progression_full.json")
    out_path.write_text(json.dumps(out, indent=2))
    _print_table(img, "image")
    _print_table(aud, "audio")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
