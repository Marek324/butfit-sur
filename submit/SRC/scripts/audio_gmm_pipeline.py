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
    map_adapt_means,
    save as save_gmm,
    score_utterance,
)
from config import load_config
from data import enumerate_split
from eval import evaluate, format_metrics


SENTINEL_SCORE = -5.0


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def _f1(conf: tuple[int, int, int, int]) -> float:
    tp, fp, fn, _tn = conf
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    if not (p == p and r == r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def _metrics_dict(m, conf: tuple[int, int, int, int]) -> dict:
    return {
        "auc": float(m.auc) if np.isfinite(m.auc) else None,
        "eer": float(m.eer) if np.isfinite(m.eer) else None,
        "confusion": list(conf),
        "precision": _safe_div(conf[0], conf[0] + conf[1]),
        "recall": _safe_div(conf[0], conf[0] + conf[2]),
        "f1": _f1(conf),
    }


def _pre_path_for(stem: str, cache_root: Path, hash_id: str) -> Path:
    return cache_root / "audio_pre" / hash_id / f"{stem}.npz"


def _load_frames_for_samples(
    samples, cache_root: Path, hash_id: str
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for s in samples:
        p = _pre_path_for(s.name, cache_root, hash_id)
        if not p.is_file():
            raise FileNotFoundError(
                f"missing pre cache for {s.name} at {p}; run scripts/train.py "
                "with the audio_baseline config first to populate the pre cache"
            )
        pre = load_pre(p)
        if pre.sp_mask.any():
            voiced = pre.mfcc[:, pre.sp_mask].T.astype(np.float64, copy=False)
        else:
            voiced = np.zeros((0, pre.mfcc.shape[0]), dtype=np.float64)
        out.append(voiced)
    return out


def _score_set(frames_list: list[np.ndarray], model: GmmModel) -> np.ndarray:
    s = np.empty(len(frames_list), dtype=np.float64)
    for i, fr in enumerate(frames_list):
        if fr.shape[0] == 0:
            s[i] = SENTINEL_SCORE
        else:
            s[i] = score_utterance(fr, model)
    return s


def _apply_sentinel(scores: np.ndarray, frames_list: list[np.ndarray]) -> np.ndarray:
    out = scores.copy()
    for i, fr in enumerate(frames_list):
        if fr.shape[0] == 0:
            out[i] = SENTINEL_SCORE
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-config", type=Path, default=Path("configs/audio_baseline.toml"))
    ap.add_argument("--cache-root", type=Path, default=Path("artifacts/cache"))
    ap.add_argument(
        "--mode",
        choices=["map_free", "map"],
        default="map_free",
        help="map_free = independent target+UBM GMMs; map = UBM + MAP-adapted target",
    )
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default depends on --mode")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--result-name", type=Path, default=None,
                    help="default depends on --mode")
    ap.add_argument("--n-mix-ubm", type=int, default=16)
    ap.add_argument("--n-mix-target", type=int, default=4,
                    help="ignored in --mode map (target inherits UBM components)")
    ap.add_argument("--relevance-factor", type=float, default=16.0,
                    help="MAP adaptation strength (only used in --mode map)")
    ap.add_argument("--reg-covar", type=float, default=1e-3)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true", help="skip eval-set scoring")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"artifacts/audio_gmm_ubm_{args.mode}")
    if args.result_name is None:
        args.result_name = f"audio_gmm_ubm_{args.mode}.txt"

    cfg = load_config(args.audio_config)
    pre_hash = pre_config_hash(cfg.audio.vad, cfg.audio.features)
    print(f"Production audio pre-cache hash: {pre_hash}")

    train_samples = enumerate_split(cfg.data.train_dirs, extension=".wav")
    dev_samples = enumerate_split(cfg.data.dev_dirs, extension=".wav")
    print(f"train: {len(train_samples)}  dev: {len(dev_samples)}")

    print("Loading frame data ...")
    t0 = time.time()
    train_frames = _load_frames_for_samples(train_samples, args.cache_root, pre_hash)
    dev_frames = _load_frames_for_samples(dev_samples, args.cache_root, pre_hash)
    print(f"  loaded train+dev frames in {time.time() - t0:.1f}s")

    y_tr = np.array([s.label for s in train_samples], dtype=int)
    target_concat = np.concatenate(
        [train_frames[i] for i in range(len(train_frames)) if y_tr[i] == 1 and train_frames[i].shape[0] > 0],
        axis=0,
    )
    ubm_concat = np.concatenate(
        [train_frames[i] for i in range(len(train_frames)) if y_tr[i] == 0 and train_frames[i].shape[0] > 0],
        axis=0,
    )
    print(f"target frames: {target_concat.shape}")
    print(f"UBM frames:    {ubm_concat.shape}")

    print(f"Fitting UBM (n_mix={args.n_mix_ubm}) ...")
    t0 = time.time()
    ubm = fit_gmm(
        ubm_concat,
        n_components=args.n_mix_ubm,
        reg_covar=args.reg_covar,
        random_state=args.random_state,
    )
    print(f"  UBM fit in {time.time() - t0:.1f}s, log-likelihood per frame ~ {ubm.score(ubm_concat):+.3f}")

    if args.mode == "map_free":
        print(f"Fitting target GMM (n_mix={args.n_mix_target}) ...")
        t0 = time.time()
        target = fit_gmm(
            target_concat,
            n_components=args.n_mix_target,
            reg_covar=args.reg_covar,
            random_state=args.random_state,
        )
        print(f"  target fit in {time.time() - t0:.1f}s, log-likelihood per frame ~ {target.score(target_concat):+.3f}")
    else:
        print(f"MAP-adapting UBM means with relevance_factor={args.relevance_factor} ...")
        t0 = time.time()
        target = map_adapt_means(ubm, target_concat, relevance_factor=args.relevance_factor)
        deltas = np.linalg.norm(target.means_ - ubm.means_, axis=1)
        print(f"  MAP adapted in {time.time() - t0:.2f}s")
        print(f"  per-component mean shift L2: min={deltas.min():.3f}  med={np.median(deltas):.3f}  max={deltas.max():.3f}")
        print(f"  target log-likelihood per frame after MAP ~ {target.score(target_concat):+.3f}")

    model = GmmModel(
        target=target,
        ubm=ubm,
        n_mfcc=int(target_concat.shape[1]),
        n_target_frames=int(target_concat.shape[0]),
        n_ubm_frames=int(ubm_concat.shape[0]),
    )
    save_gmm(model, args.out_dir / "model.pkl")
    print(f"Saved model to {args.out_dir / 'model.pkl'}")

    y_dv = np.array([s.label for s in dev_samples], dtype=int)
    s_dv = _score_set(dev_frames, model)
    metrics = evaluate(s_dv, y_dv, threshold=0.0)
    print()
    print("Clean dev:")
    print(format_metrics(metrics))

    payload = {
        "name": f"audio_gmm_ubm_{args.mode}",
        "modality": "audio",
        "mode": args.mode,
        "n_mix_ubm": args.n_mix_ubm,
        "n_mix_target": args.n_mix_target if args.mode == "map_free" else args.n_mix_ubm,
        "relevance_factor": args.relevance_factor if args.mode == "map" else None,
        "n_train_target_frames": model.n_target_frames,
        "n_train_ubm_frames": model.n_ubm_frames,
        "dev": _metrics_dict(metrics, metrics.confusion),
    }

    stress_cfg_path = Path("configs/audio_baseline_stress.toml")
    if stress_cfg_path.is_file():
        stress_cfg = load_config(stress_cfg_path)
        stress_samples = enumerate_split(stress_cfg.data.dev_dirs, extension=".wav")
        stress_frames = _load_frames_for_samples(stress_samples, args.cache_root, pre_hash)
        y_st = np.array([s.label for s in stress_samples], dtype=int)
        s_st = _score_set(stress_frames, model)
        metrics_st = evaluate(s_st, y_st, threshold=0.0)
        print()
        print("Stress dev:")
        print(format_metrics(metrics_st))
        payload["stress_dev"] = _metrics_dict(metrics_st, metrics_st.confusion)
        n_empty_stress = sum(1 for fr in stress_frames if fr.shape[0] == 0)
        payload["stress_dev"]["vad_empty_count"] = n_empty_stress
        print(f"  (VAD-empty stress samples: {n_empty_stress} of {len(stress_samples)})")

    if not args.no_eval:
        eval_dirs = cfg.data.eval_dirs or [Path("dataset/eval")]
        eval_samples = enumerate_split(eval_dirs, extension=".wav", label_required=False)
        print()
        print(f"Scoring eval set ({len(eval_samples)} samples) ...")
        eval_frames = _load_frames_for_samples(eval_samples, args.cache_root, pre_hash)
        s_ev = _score_set(eval_frames, model)
        n_empty = sum(1 for fr in eval_frames if fr.shape[0] == 0)
        n_target = int((s_ev >= 0.0).sum())
        print(f"  VAD-empty (sentinel): {n_empty}")
        print(f"  predicted target:    {n_target} ({100*n_target/len(s_ev):.1f}%)")
        print(f"  score range: [{s_ev.min():.3f}, {s_ev.max():.3f}], median {np.median(s_ev):.3f}")

        out_path = args.results_dir / args.result_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for s, score_val in zip(eval_samples, s_ev):
                hard = 1 if score_val >= 0.0 else 0
                f.write(f"{s.name} {score_val:.6f} {hard}\n")
        print(f"Wrote {out_path}")
        payload["eval"] = {
            "n": len(s_ev),
            "predicted_target": n_target,
            "vad_empty": n_empty,
            "score_min": float(s_ev.min()),
            "score_med": float(np.median(s_ev)),
            "score_max": float(s_ev.max()),
        }

    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
