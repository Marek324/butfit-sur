import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  
import numpy as np  
from scipy.special import expit  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import StratifiedKFold  

from classifier import load as load_classifier  
from config import load_config  
from data import enumerate_paired  
from experiments.runners import _score_paired  


def _calibration_metrics(p: np.ndarray, y: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred_mean = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            pred_mean[b] = float(p[mask].mean())
            obs_freq[b] = float(y[mask].mean())
            counts[b] = int(mask.sum())
    brier = float(np.mean((p - y) ** 2))
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        ece += (counts[b] / n) * abs(obs_freq[b] - pred_mean[b])
    return brier, float(ece), centers, pred_mean, obs_freq, counts


def _oof_fusion_scores(
    a_scores: np.ndarray, i_scores: np.ndarray, y: np.ndarray, n_folds: int, seed: int = 42
) -> np.ndarray:
    n = len(y)
    n_targets = int((y == 1).sum())
    n_folds = min(n_folds, n_targets)
    oof = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, dv in skf.split(np.zeros(n), y):
        a_mean = float(a_scores[tr].mean())
        a_std = float(a_scores[tr].std()) or 1.0
        i_mean = float(i_scores[tr].mean())
        i_std = float(i_scores[tr].std()) or 1.0
        a_n = (a_scores - a_mean) / a_std
        i_n = (i_scores - i_mean) / i_std
        X = np.column_stack([a_n, i_n])
        if (y[tr] == 1).sum() == 0 or (y[tr] == 0).sum() == 0:
            continue
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", l1_ratio=0.0, max_iter=1000
        )
        clf.fit(X[tr], y[tr])
        oof[dv] = clf.decision_function(X[dv])
    return oof


def _plot_reliability(
    ax, name: str, p: np.ndarray, y: np.ndarray, n_bins: int, color: str
) -> tuple[float, float]:
    brier, ece, _centers, pred_mean, obs_freq, counts = _calibration_metrics(p, y, n_bins)
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1, label="ideal")
    valid = counts > 0
    ax.plot(
        pred_mean[valid],
        obs_freq[valid],
        marker="o",
        color=color,
        linewidth=2,
        markersize=7,
        label=name,
    )
    for x, yv, c in zip(pred_mean[valid], obs_freq[valid], counts[valid], strict=True):
        ax.annotate(
            f"n={int(c)}",
            (x, yv),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color="0.4",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("predicted P(target=1)")
    ax.set_ylabel("observed frequency of target")
    ax.set_title(f"{name}\nBrier={brier:.4f}  ECE={ece:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    return brier, ece


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--fusion-config", type=Path, default=Path("configs/fusion_baseline.toml")
    )
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/calibration_dev.png")
    )
    args = ap.parse_args()

    cfg = load_config(args.fusion_config)
    audio_dir = args.artifacts / cfg.fusion.audio_artifact
    image_dir = args.artifacts / cfg.fusion.image_artifact
    audio_model = load_classifier(audio_dir / "model.pkl")
    image_model = load_classifier(image_dir / "model.pkl")
    audio_cfg = load_config(audio_dir / "config.toml")
    image_cfg = load_config(image_dir / "config.toml")

    paired = enumerate_paired(cfg.data.dev_dirs)
    a_scores, _ = _score_paired(paired, audio_cfg, "audio", audio_model)
    i_scores, _ = _score_paired(paired, image_cfg, "image", image_model)
    y = np.array([s.label for s in paired], dtype=int)
    print(
        f"Dev paired samples: {len(paired)} "
        f"(target={int((y == 1).sum())}, non-target={int((y == 0).sum())})"
    )

    f_scores = _oof_fusion_scores(a_scores, i_scores, y, n_folds=args.n_folds)
    valid = ~np.isnan(f_scores)
    if not valid.all():
        print(f"  warning: {int((~valid).sum())} fusion OOF scores missing")

    a_p = expit(a_scores)
    i_p = expit(i_scores)
    f_p = expit(f_scores[valid])
    f_y = y[valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    a_brier, a_ece = _plot_reliability(axes[0], "Audio", a_p, y, args.n_bins, "tab:blue")
    i_brier, i_ece = _plot_reliability(axes[1], "Image", i_p, y, args.n_bins, "tab:green")
    f_brier, f_ece = _plot_reliability(
        axes[2], "Fusion (CV-OOF)", f_p, f_y, args.n_bins, "tab:orange"
    )
    fig.suptitle(
        f"Score calibration on dev (n={len(paired)}, {args.n_bins} bins) - "
        "predicted P(target) via sigmoid(logit)",
        fontsize=12,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 60)
    print("Calibration metrics (dev):")
    print(f"  {'modality':<16} {'Brier':>8} {'ECE':>8}")
    print(f"  {'audio':<16} {a_brier:>8.4f} {a_ece:>8.4f}")
    print(f"  {'image':<16} {i_brier:>8.4f} {i_ece:>8.4f}")
    print(f"  {'fusion (OOF)':<16} {f_brier:>8.4f} {f_ece:>8.4f}")
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
