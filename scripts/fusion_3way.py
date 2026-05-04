
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from audio_features import load_pre, pre_config_hash
from audio_gmm import GmmModel, load as load_gmm, score_utterance
from classifier import load as load_classifier
from config import load_config
from data import enumerate_split
from experiments.extraction import extract_set


def _eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def _hybrid_audio_scores(samples, cache_root: Path, pre_hash: str,
                          prod_model: GmmModel, no_vad_model: GmmModel,
                          sentinel: float = -5.0):
    s = np.empty(len(samples), dtype=np.float64)
    for i, sm in enumerate(samples):
        p = cache_root / "audio_pre" / pre_hash / f"{sm.name}.npz"
        pre = load_pre(p)
        if pre.sp_mask.any():
            fr = pre.mfcc[:, pre.sp_mask].T.astype(np.float64, copy=False)
            s[i] = score_utterance(fr, prod_model)
        else:
            fr = pre.mfcc.T.astype(np.float64, copy=False)
            s[i] = score_utterance(fr, no_vad_model) if fr.shape[0] > 0 else sentinel
    return s


def _confusion(s, y, thr=0.0):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-config", type=Path, default=Path("configs/audio_baseline.toml"))
    ap.add_argument("--image-config", type=Path, default=Path("configs/image_baseline.toml"))
    ap.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    ap.add_argument("--audio-logreg", type=Path,
                    default=Path("artifacts/audio_baseline_logreg/model.pkl"))
    ap.add_argument("--prod-gmm", type=Path,
                    default=Path("artifacts/audio_gmm_ubm_map_free/model.pkl"))
    ap.add_argument("--no-vad-gmm", type=Path,
                    default=Path("artifacts/audio_gmm_hybrid/model_no_vad.pkl"))
    ap.add_argument("--image-model", type=Path,
                    default=Path("artifacts/image_baseline_logreg/model.pkl"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_3way"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    audio_cfg = load_config(args.audio_config)
    image_cfg = load_config(args.image_config)
    pre_hash = pre_config_hash(audio_cfg.audio.vad, audio_cfg.audio.features)

    print("Loading models ...")
    audio_logreg = load_classifier(args.audio_logreg)
    prod_gmm = load_gmm(args.prod_gmm)
    no_vad_gmm = load_gmm(args.no_vad_gmm)
    image_model = load_classifier(args.image_model)

    audio_dev = enumerate_split(audio_cfg.data.dev_dirs, ".wav")
    image_dev = enumerate_split(image_cfg.data.dev_dirs, ".png")
    assert [s.name for s in audio_dev] == [s.name for s in image_dev]
    y_dev = np.asarray([s.label for s in audio_dev])

    print("Scoring dev with audio_logreg ...")
    Xa_dev, _, _, _ = extract_set(audio_dev, audio_cfg, "audio")
    a1_dev = np.asarray(audio_logreg.estimator.decision_function(Xa_dev), dtype=np.float64)
    zero_mask = np.all(Xa_dev == 0, axis=1)
    a1_dev[zero_mask] = -5.0

    print("Scoring dev with audio hybrid ...")
    a2_dev = _hybrid_audio_scores(audio_dev, args.cache_root, pre_hash, prod_gmm, no_vad_gmm)

    print("Scoring dev with image ...")
    Xi_dev, _, _, _ = extract_set(image_dev, image_cfg, "image")
    i_dev = np.asarray(image_model.estimator.decision_function(Xi_dev), dtype=np.float64)

    norms = {}
    for name, s in [("a1", a1_dev), ("a2", a2_dev), ("i", i_dev)]:
        m, sd = float(s.mean()), float(s.std()) or 1.0
        norms[f"{name}_mean"] = m
        norms[f"{name}_std"] = sd
    a1z = (a1_dev - norms["a1_mean"]) / norms["a1_std"]
    a2z = (a2_dev - norms["a2_mean"]) / norms["a2_std"]
    iz  = (i_dev  - norms["i_mean"])  / norms["i_std"]
    X_dev = np.column_stack([a1z, a2z, iz])

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_dev, y_dev)
    print(f"Fusion coefs (audio_logreg, audio_hybrid, image): {clf.coef_.ravel()}")
    print(f"Intercept: {float(clf.intercept_[0]):+.3f}")

    fused_dev = clf.decision_function(X_dev)
    auc_d = float(roc_auc_score(y_dev, fused_dev))
    eer_d = _eer(y_dev, fused_dev)
    cd = _confusion(fused_dev, y_dev)
    print(f"\nClean dev: AUC={auc_d:.4f}  EER={eer_d*100:.2f}%  TP/FP/FN/TN={cd}")

    stress_cfg = load_config(Path("configs/audio_baseline_stress.toml"))
    audio_st = enumerate_split(stress_cfg.data.dev_dirs, ".wav")
    image_st = enumerate_split(stress_cfg.data.dev_dirs, ".png")
    assert [s.name for s in audio_st] == [s.name for s in image_st]
    y_st = np.asarray([s.label for s in audio_st])

    Xa_st, _, _, _ = extract_set(audio_st, audio_cfg, "audio")
    a1_st = np.asarray(audio_logreg.estimator.decision_function(Xa_st), dtype=np.float64)
    a1_st[np.all(Xa_st == 0, axis=1)] = -5.0
    a2_st = _hybrid_audio_scores(audio_st, args.cache_root, pre_hash, prod_gmm, no_vad_gmm)
    Xi_st, _, _, _ = extract_set(image_st, image_cfg, "image")
    i_st = np.asarray(image_model.estimator.decision_function(Xi_st), dtype=np.float64)
    X_st = np.column_stack([
        (a1_st - norms["a1_mean"]) / norms["a1_std"],
        (a2_st - norms["a2_mean"]) / norms["a2_std"],
        (i_st  - norms["i_mean"])  / norms["i_std"],
    ])
    fused_st = clf.decision_function(X_st)
    auc_s = float(roc_auc_score(y_st, fused_st))
    eer_s = _eer(y_st, fused_st)
    cs = _confusion(fused_st, y_st)
    print(f"Stress dev: AUC={auc_s:.4f}  EER={eer_s*100:.2f}%  TP/FP/FN/TN={cs}")

    eval_dirs = audio_cfg.data.eval_dirs or [Path("dataset/eval")]
    audio_ev = enumerate_split(eval_dirs, ".wav", label_required=False)
    image_ev = enumerate_split(eval_dirs, ".png", label_required=False)
    assert [s.name for s in audio_ev] == [s.name for s in image_ev]

    Xa_ev, _, _, _ = extract_set(audio_ev, audio_cfg, "audio")
    a1_ev = np.asarray(audio_logreg.estimator.decision_function(Xa_ev), dtype=np.float64)
    a1_ev[np.all(Xa_ev == 0, axis=1)] = -5.0
    a2_ev = _hybrid_audio_scores(audio_ev, args.cache_root, pre_hash, prod_gmm, no_vad_gmm)
    Xi_ev, _, _, _ = extract_set(image_ev, image_cfg, "image")
    i_ev = np.asarray(image_model.estimator.decision_function(Xi_ev), dtype=np.float64)
    X_ev = np.column_stack([
        (a1_ev - norms["a1_mean"]) / norms["a1_std"],
        (a2_ev - norms["a2_mean"]) / norms["a2_std"],
        (i_ev  - norms["i_mean"])  / norms["i_std"],
    ])
    fused_ev = clf.decision_function(X_ev)
    n_target = int((fused_ev >= 0).sum())
    print(f"\nEval predicted target: {n_target}/{len(audio_ev)} ({100*n_target/len(audio_ev):.1f}%)")
    print(f"  score range [{fused_ev.min():.3f}, {fused_ev.max():.3f}], median {np.median(fused_ev):.3f}")

    out_path = args.results_dir / "fusion_3way.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s, score_val in zip(audio_ev, fused_ev):
            hard = 1 if score_val >= 0.0 else 0
            f.write(f"{s.name} {score_val:.6f} {hard}\n")
    print(f"Wrote {out_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "fusion_3way",
        "modality": "fusion",
        "inputs": ["audio_logreg", "audio_gmm_hybrid", "image_logreg"],
        "z_norm": norms,
        "fusion_coefs": clf.coef_.ravel().tolist(),
        "fusion_intercept": float(clf.intercept_[0]),
        "dev": {"auc": auc_d, "eer": eer_d, "confusion": list(cd)},
        "stress_dev": {"auc": auc_s, "eer": eer_s, "confusion": list(cs)},
        "eval": {"n": len(audio_ev), "predicted_target": n_target,
                  "score_min": float(fused_ev.min()),
                  "score_med": float(np.median(fused_ev)),
                  "score_max": float(fused_ev.max())},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
