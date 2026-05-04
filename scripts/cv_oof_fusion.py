
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import roc_auc_score, roc_curve  
from sklearn.model_selection import StratifiedKFold  

from classifier import load as load_classifier  
from config import load_config  
from data import enumerate_paired  
from experiments.runners import _score_paired  


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fusion-config", type=Path, default=Path("configs/fusion.toml")
    )
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cv_oof_fusion.png"),
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
    print(f"Dev paired samples: {len(paired)}")
    a_scores, _ = _score_paired(paired, audio_cfg, "audio", audio_model)
    i_scores, _ = _score_paired(paired, image_cfg, "image", image_model)
    y = np.array([s.label for s in paired], dtype=int)
    print(f"Per-modality dev scores: audio range [{a_scores.min():.2f}, {a_scores.max():.2f}], "
          f"image range [{i_scores.min():.2f}, {i_scores.max():.2f}]")
    print(f"Targets: {(y == 1).sum()}, non-targets: {(y == 0).sum()}")

    # K-fold OOF.
    n = len(paired)
    n_targets = int((y == 1).sum())
    n_folds = min(args.n_folds, n_targets)
    if n_folds < args.n_folds:
        print(f"  reducing K to {n_folds} (only {n_targets} targets)")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_scores = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)
    for fold_idx, (tr, dv) in enumerate(skf.split(np.zeros(n), y)):
        # Per-fold zscore stats on train portion only.
        a_mean = float(a_scores[tr].mean())
        a_std = float(a_scores[tr].std()) or 1.0
        i_mean = float(i_scores[tr].mean())
        i_std = float(i_scores[tr].std()) or 1.0
        a_n = (a_scores - a_mean) / a_std
        i_n = (i_scores - i_mean) / i_std
        X = np.column_stack([a_n, i_n])

        if (y[tr] == 1).sum() == 0 or (y[tr] == 0).sum() == 0:
            print(f"  fold {fold_idx}: only one class in train, skipping")
            continue
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", l1_ratio=0.0, max_iter=1000
        )
        clf.fit(X[tr], y[tr])
        oof_scores[dv] = clf.decision_function(X[dv])
        valid_mask[dv] = True
        print(
            f"  fold {fold_idx}: train n={len(tr)} ({(y[tr] == 1).sum()} tgt), "
            f"holdout n={len(dv)} ({(y[dv] == 1).sum()} tgt), "
            f"weights={np.asarray(clf.coef_).ravel()}, intercept={float(np.asarray(clf.intercept_).ravel()[0]):+.3f}"
        )

    if not valid_mask.all():
        print(f"  warning: {(~valid_mask).sum()} samples weren't held out")

    # Metrics on OOF predictions.
    s = oof_scores[valid_mask]
    yy = y[valid_mask]
    auc = float(roc_auc_score(yy, s))
    fpr, tpr, _thresholds = roc_curve(yy, s)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    hard = (s >= 0).astype(int)
    tp = int(((hard == 1) & (yy == 1)).sum())
    fp = int(((hard == 1) & (yy == 0)).sum())
    fn = int(((hard == 0) & (yy == 1)).sum())
    tn = int(((hard == 0) & (yy == 0)).sum())

    # Compare with the dev-trained fusion (biased).
    fusion_dev = (args.artifacts / cfg.name / "metrics.json")
    biased = None
    if fusion_dev.is_file():
        import json

        biased = json.loads(fusion_dev.read_text()).get("dev", {})

    print()
    print("=" * 60)
    print(f"Unbiased K-fold OOF fusion ({n_folds} folds, stratified):")
    print(f"  AUC = {auc:.4f}")
    print(f"  EER = {eer * 100:.2f}%")
    print(f"  Confusion @ logit=0:  TP={tp} FP={fp} FN={fn} TN={tn}")
    if biased is not None:
        b_auc = biased.get("auc", float("nan"))
        b_eer = biased.get("eer", float("nan"))
        b_conf = biased.get("confusion", [None, None, None, None])
        print()
        print("Compared to production fusion (trained on full dev - biased):")
        print(f"  AUC = {b_auc:.4f}  (delta {auc - b_auc:+.4f})")
        print(f"  EER = {b_eer * 100:.2f}%  (delta {(eer - b_eer) * 100:+.2f}pp)")
        if all(c is not None for c in b_conf):
            print(f"  Confusion @ logit=0: TP={b_conf[0]} FP={b_conf[1]} FN={b_conf[2]} TN={b_conf[3]}")

    # Plot ROC: OOF vs biased dev (if both available).
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot(fpr, tpr, color="tab:orange", label=f"CV-OOF (AUC={auc:.3f})")
    ax.scatter([fpr[idx]], [tpr[idx]], color="tab:orange", s=60, zorder=5)
    if biased is not None and biased.get("auc") is not None:
        # Re-compute the biased ROC from saved scores file if possible.
        # Simpler: just label the comparison point in the title.
        ax.set_title(
            f"Fusion ROC: CV-OOF (unbiased) vs dev-trained\n"
            f"OOF AUC={auc:.3f}, biased AUC={biased['auc']:.3f}"
        )
    else:
        ax.set_title(f"Fusion ROC - CV-OOF (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
