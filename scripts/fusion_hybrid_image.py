
import argparse
import json
import sys
import time
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
from data import enumerate_paired, enumerate_split
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-config", type=Path, default=Path("configs/audio.toml"))
    ap.add_argument("--image-config", type=Path, default=Path("configs/image.toml"))
    ap.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    ap.add_argument("--prod-gmm", type=Path,
                    default=Path("artifacts/audio_gmm_ubm_map_free/model.pkl"))
    ap.add_argument("--no-vad-gmm", type=Path,
                    default=Path("artifacts/audio_gmm_hybrid/model_no_vad.pkl"))
    ap.add_argument("--image-model", type=Path,
                    default=Path("artifacts/image_baseline_logreg/model.pkl"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_hybrid_image"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    audio_cfg = load_config(args.audio_config)
    image_cfg = load_config(args.image_config)
    pre_hash = pre_config_hash(audio_cfg.audio.vad, audio_cfg.audio.features)

    print("Loading production GMM-UBM (VAD-on) ...")
    prod_gmm = load_gmm(args.prod_gmm)
    print("Loading no-VAD GMM-UBM ...")
    no_vad_gmm = load_gmm(args.no_vad_gmm)
    print("Loading image logreg ...")
    image_model = load_classifier(args.image_model)

    paired = enumerate_paired(audio_cfg.data.dev_dirs)
    print(f"Dev paired samples: {len(paired)}")
    audio_dev_samples = [s for s in enumerate_split(audio_cfg.data.dev_dirs, ".wav")]
    image_dev_samples = [s for s in enumerate_split(image_cfg.data.dev_dirs, ".png")]
    name_a = [s.name for s in audio_dev_samples]
    name_i = [s.name for s in image_dev_samples]
    assert name_a == name_i, "audio/image dev sample names mismatch"

    print("Scoring dev with hybrid audio ...")
    a_dev = _hybrid_audio_scores(audio_dev_samples, args.cache_root, pre_hash,
                                  prod_gmm, no_vad_gmm)
    print("Scoring dev with image ...")
    Xi, _, _, _ = extract_set(image_dev_samples, image_cfg, "image")
    i_dev = np.asarray(image_model.estimator.decision_function(Xi), dtype=np.float64)
    y_dev = np.asarray([s.label for s in audio_dev_samples])
    print(f"  audio dev: range [{a_dev.min():.3f}, {a_dev.max():.3f}], mean {a_dev.mean():.3f}")
    print(f"  image dev: range [{i_dev.min():.3f}, {i_dev.max():.3f}], mean {i_dev.mean():.3f}")

    a_mean, a_std = float(a_dev.mean()), float(a_dev.std()) or 1.0
    i_mean, i_std = float(i_dev.mean()), float(i_dev.std()) or 1.0
    a_z = (a_dev - a_mean) / a_std
    i_z = (i_dev - i_mean) / i_std
    X = np.column_stack([a_z, i_z])

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X, y_dev)
    print(f"Fusion logreg coefs: {clf.coef_.ravel()}, intercept: {float(clf.intercept_[0]):+.3f}")

    fused_dev = clf.decision_function(X)
    auc = float(roc_auc_score(y_dev, fused_dev))
    eer = _eer(y_dev, fused_dev)
    pred = (fused_dev >= 0).astype(int)
    tp = int(((pred == 1) & (y_dev == 1)).sum())
    fp = int(((pred == 1) & (y_dev == 0)).sum())
    fn = int(((pred == 0) & (y_dev == 1)).sum())
    tn = int(((pred == 0) & (y_dev == 0)).sum())
    print(f"\nClean dev: AUC={auc:.4f}  EER={eer*100:.2f}%  TP={tp} FP={fp} FN={fn} TN={tn}")

    stress_cfg_path = Path("configs/audio_stress.toml")
    stress_metrics = None
    if stress_cfg_path.is_file():
        stress_cfg = load_config(stress_cfg_path)
        audio_st = enumerate_split(stress_cfg.data.dev_dirs, ".wav")
        image_st = enumerate_split(stress_cfg.data.dev_dirs, ".png")
        a_st = _hybrid_audio_scores(audio_st, args.cache_root, pre_hash, prod_gmm, no_vad_gmm)
        Xi_st, _, _, _ = extract_set(image_st, image_cfg, "image")
        i_st = np.asarray(image_model.estimator.decision_function(Xi_st), dtype=np.float64)
        y_st = np.asarray([s.label for s in audio_st])
        a_z_st = (a_st - a_mean) / a_std
        i_z_st = (i_st - i_mean) / i_std
        X_st = np.column_stack([a_z_st, i_z_st])
        fused_st = clf.decision_function(X_st)
        auc_s = float(roc_auc_score(y_st, fused_st))
        eer_s = _eer(y_st, fused_st)
        pred_s = (fused_st >= 0).astype(int)
        tp_s = int(((pred_s == 1) & (y_st == 1)).sum())
        fp_s = int(((pred_s == 1) & (y_st == 0)).sum())
        fn_s = int(((pred_s == 0) & (y_st == 1)).sum())
        tn_s = int(((pred_s == 0) & (y_st == 0)).sum())
        print(f"Stress dev: AUC={auc_s:.4f}  EER={eer_s*100:.2f}%  TP={tp_s} FP={fp_s} FN={fn_s} TN={tn_s}")
        stress_metrics = {
            "auc": auc_s, "eer": eer_s,
            "confusion": [tp_s, fp_s, fn_s, tn_s],
        }

    eval_dirs = audio_cfg.data.eval_dirs or [Path("dataset/eval")]
    audio_ev = enumerate_split(eval_dirs, ".wav", label_required=False)
    image_ev = enumerate_split(eval_dirs, ".png", label_required=False)
    print(f"\nScoring eval ({len(audio_ev)} samples) ...")
    a_ev = _hybrid_audio_scores(audio_ev, args.cache_root, pre_hash, prod_gmm, no_vad_gmm)
    Xi_ev, _, _, _ = extract_set(image_ev, image_cfg, "image")
    i_ev = np.asarray(image_model.estimator.decision_function(Xi_ev), dtype=np.float64)
    a_z_ev = (a_ev - a_mean) / a_std
    i_z_ev = (i_ev - i_mean) / i_std
    fused_ev = clf.decision_function(np.column_stack([a_z_ev, i_z_ev]))
    n_target = int((fused_ev >= 0).sum())
    print(f"  predicted target: {n_target} ({100*n_target/len(audio_ev):.1f}%)")
    print(f"  score range: [{fused_ev.min():.3f}, {fused_ev.max():.3f}]  median {np.median(fused_ev):.3f}")

    out_path = args.results_dir / "fusion_hybrid_image.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s, score_val in zip(audio_ev, fused_ev):
            hard = 1 if score_val >= 0.0 else 0
            f.write(f"{s.name} {score_val:.6f} {hard}\n")
    print(f"Wrote {out_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "fusion_hybrid_image",
        "modality": "fusion",
        "audio_path": "audio_gmm_hybrid (production GMM-UBM with no-VAD fallback)",
        "image_path": "image_logreg (HOG + balanced LogReg + augmentation)",
        "z_norm": {"a_mean": a_mean, "a_std": a_std, "i_mean": i_mean, "i_std": i_std},
        "fusion_coefs": clf.coef_.ravel().tolist(),
        "fusion_intercept": float(clf.intercept_[0]),
        "dev": {
            "auc": auc, "eer": eer, "confusion": [tp, fp, fn, tn],
        },
        "stress_dev": stress_metrics,
        "eval": {
            "n": len(audio_ev),
            "predicted_target": n_target,
            "score_min": float(fused_ev.min()),
            "score_med": float(np.median(fused_ev)),
            "score_max": float(fused_ev.max()),
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
