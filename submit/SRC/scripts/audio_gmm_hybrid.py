#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np

from audio_features import load_pre, pre_config_hash
from audio_gmm import (
    GmmModel,
    fit_gmm,
    save as save_gmm,
    load as load_gmm,
    score_utterance,
)
from config import load_config
from data import enumerate_split
from eval import evaluate, format_metrics


def _load_frames_for_samples(samples, cache_root: Path, hash_id: str, voiced_only: bool):
    out = []
    for s in samples:
        p = cache_root / "audio_pre" / hash_id / f"{s.name}.npz"
        pre = load_pre(p)
        has_voiced = bool(pre.sp_mask.any())
        if voiced_only:
            if has_voiced:
                fr = pre.mfcc[:, pre.sp_mask].T.astype(np.float64, copy=False)
            else:
                fr = np.zeros((0, pre.mfcc.shape[0]), dtype=np.float64)
        else:
            fr = pre.mfcc.T.astype(np.float64, copy=False)
        out.append((fr, has_voiced))
    return out


def _score_set(per_sample, model: GmmModel, sentinel: float | None = None):
    s = np.empty(len(per_sample), dtype=np.float64)
    for i, (fr, has_voiced) in enumerate(per_sample):
        if fr.shape[0] == 0:
            s[i] = sentinel if sentinel is not None else float("nan")
        else:
            s[i] = score_utterance(fr, model)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-config", type=Path, default=Path("configs/audio_baseline.toml"))
    ap.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    ap.add_argument(
        "--production-model",
        type=Path,
        default=Path("artifacts/audio_gmm_ubm_map_free/model.pkl"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/audio_gmm_hybrid"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-mix-ubm", type=int, default=16)
    ap.add_argument("--n-mix-target", type=int, default=4)
    ap.add_argument("--reg-covar", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = load_config(args.audio_config)
    pre_hash = pre_config_hash(cfg.audio.vad, cfg.audio.features)
    print(f"pre-cache hash: {pre_hash}")

    print("Loading production GMM-UBM (VAD-on) ...")
    prod = load_gmm(args.production_model)

    train_samples = enumerate_split(cfg.data.train_dirs, extension=".wav")
    dev_samples = enumerate_split(cfg.data.dev_dirs, extension=".wav")
    print(f"train: {len(train_samples)}  dev: {len(dev_samples)}")

    print("Loading full-frame (no-VAD) MFCCs from pre-cache ...")
    t0 = time.time()
    train_full = _load_frames_for_samples(train_samples, args.cache_root, pre_hash, voiced_only=False)
    print(f"  train frames loaded in {time.time() - t0:.1f}s")

    y_tr = np.array([s.label for s in train_samples], dtype=int)
    target_concat = np.concatenate(
        [train_full[i][0] for i in range(len(train_full)) if y_tr[i] == 1 and train_full[i][0].shape[0] > 0],
        axis=0,
    )
    ubm_concat = np.concatenate(
        [train_full[i][0] for i in range(len(train_full)) if y_tr[i] == 0 and train_full[i][0].shape[0] > 0],
        axis=0,
    )
    print(f"  target frames (no VAD): {target_concat.shape}")
    print(f"  UBM frames    (no VAD): {ubm_concat.shape}")

    print(f"Fitting no-VAD UBM (n_mix={args.n_mix_ubm}) ...")
    t0 = time.time()
    ubm = fit_gmm(ubm_concat, n_components=args.n_mix_ubm, reg_covar=args.reg_covar, random_state=0)
    print(f"  UBM fit in {time.time() - t0:.1f}s")

    print(f"Fitting no-VAD target GMM (n_mix={args.n_mix_target}) ...")
    t0 = time.time()
    target = fit_gmm(target_concat, n_components=args.n_mix_target, reg_covar=args.reg_covar, random_state=0)
    print(f"  target fit in {time.time() - t0:.1f}s")

    no_vad = GmmModel(
        target=target, ubm=ubm,
        n_mfcc=int(target_concat.shape[1]),
        n_target_frames=int(target_concat.shape[0]),
        n_ubm_frames=int(ubm_concat.shape[0]),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_gmm(no_vad, args.out_dir / "model_no_vad.pkl")

    print()
    print("Evaluating clean dev (hybrid scoring) ...")
    dev_voiced = _load_frames_for_samples(dev_samples, args.cache_root, pre_hash, voiced_only=True)
    dev_full   = _load_frames_for_samples(dev_samples, args.cache_root, pre_hash, voiced_only=False)
    y_dv = np.array([s.label for s in dev_samples], dtype=int)
    s_dv = np.empty(len(dev_samples), dtype=np.float64)
    n_fallback_dv = 0
    for i in range(len(dev_samples)):
        fr_v, has_voiced = dev_voiced[i]
        if has_voiced:
            s_dv[i] = score_utterance(fr_v, prod)
        else:
            s_dv[i] = score_utterance(dev_full[i][0], no_vad)
            n_fallback_dv += 1
    print(f"  fallback used on {n_fallback_dv}/{len(dev_samples)} clean dev samples")
    metrics_dv = evaluate(s_dv, y_dv, threshold=0.0)
    print(format_metrics(metrics_dv))

    stress_cfg_path = Path("configs/audio_baseline_stress.toml")
    metrics_st = None
    if stress_cfg_path.is_file():
        print()
        print("Evaluating stress dev (hybrid scoring) ...")
        stress_cfg = load_config(stress_cfg_path)
        stress_samples = enumerate_split(stress_cfg.data.dev_dirs, extension=".wav")
        st_voiced = _load_frames_for_samples(stress_samples, args.cache_root, pre_hash, voiced_only=True)
        st_full   = _load_frames_for_samples(stress_samples, args.cache_root, pre_hash, voiced_only=False)
        y_st = np.array([s.label for s in stress_samples], dtype=int)
        s_st = np.empty(len(stress_samples), dtype=np.float64)
        n_fallback_st = 0
        for i in range(len(stress_samples)):
            fr_v, has_voiced = st_voiced[i]
            if has_voiced:
                s_st[i] = score_utterance(fr_v, prod)
            else:
                s_st[i] = score_utterance(st_full[i][0], no_vad)
                n_fallback_st += 1
        print(f"  fallback used on {n_fallback_st}/{len(stress_samples)} stress dev samples")
        metrics_st = evaluate(s_st, y_st, threshold=0.0)
        print(format_metrics(metrics_st))

    eval_dirs = cfg.data.eval_dirs or [Path("dataset/eval")]
    eval_samples = enumerate_split(eval_dirs, extension=".wav", label_required=False)
    print()
    print(f"Scoring eval set ({len(eval_samples)} samples, hybrid) ...")
    ev_voiced = _load_frames_for_samples(eval_samples, args.cache_root, pre_hash, voiced_only=True)
    ev_full   = _load_frames_for_samples(eval_samples, args.cache_root, pre_hash, voiced_only=False)
    s_ev = np.empty(len(eval_samples), dtype=np.float64)
    used_fallback = np.zeros(len(eval_samples), dtype=bool)
    for i in range(len(eval_samples)):
        fr_v, has_voiced = ev_voiced[i]
        if has_voiced:
            s_ev[i] = score_utterance(fr_v, prod)
        else:
            s_ev[i] = score_utterance(ev_full[i][0], no_vad)
            used_fallback[i] = True
    n_fb = int(used_fallback.sum())
    n_target = int((s_ev >= 0.0).sum())
    n_target_via_fb = int(((used_fallback) & (s_ev >= 0.0)).sum())
    print(f"  fallback used on {n_fb}/{len(eval_samples)} eval samples")
    print(f"  predicted target: {n_target} ({100*n_target/len(eval_samples):.1f}%), of which {n_target_via_fb} via fallback")
    print(f"  score range: [{s_ev.min():.3f}, {s_ev.max():.3f}], median {np.median(s_ev):.3f}")

    out_path = args.results_dir / "audio_gmm_hybrid.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s, score_val in zip(eval_samples, s_ev):
            hard = 1 if score_val >= 0.0 else 0
            f.write(f"{s.name} {score_val:.6f} {hard}\n")
    print(f"Wrote {out_path}")

    payload = {
        "name": "audio_gmm_hybrid",
        "modality": "audio",
        "production_model": str(args.production_model),
        "no_vad_model": str(args.out_dir / "model_no_vad.pkl"),
        "n_mix_ubm": args.n_mix_ubm,
        "n_mix_target": args.n_mix_target,
        "n_train_target_frames_no_vad": no_vad.n_target_frames,
        "n_train_ubm_frames_no_vad": no_vad.n_ubm_frames,
        "dev": {
            "auc": float(metrics_dv.auc) if np.isfinite(metrics_dv.auc) else None,
            "eer": float(metrics_dv.eer) if np.isfinite(metrics_dv.eer) else None,
            "confusion": list(metrics_dv.confusion),
            "fallback_used": n_fallback_dv,
        },
        "stress_dev": (
            {
                "auc": float(metrics_st.auc) if np.isfinite(metrics_st.auc) else None,
                "eer": float(metrics_st.eer) if np.isfinite(metrics_st.eer) else None,
                "confusion": list(metrics_st.confusion),
                "fallback_used": n_fallback_st,
            }
            if metrics_st is not None
            else None
        ),
        "eval": {
            "n": len(eval_samples),
            "predicted_target": n_target,
            "fallback_used": n_fb,
            "predicted_target_via_fallback": n_target_via_fb,
            "score_min": float(s_ev.min()),
            "score_med": float(np.median(s_ev)),
            "score_max": float(s_ev.max()),
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
