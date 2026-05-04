#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from classifier import load as load_classifier  # noqa: E402
from config import load_config  # noqa: E402
from data import enumerate_paired  # noqa: E402
from experiments.runners import _score_paired  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--fusion-config", type=Path, default=Path("configs/fusion_baseline.toml")
    )
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/dev_error_correlation.png")
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

    a_pred = (a_scores >= 0).astype(int)
    i_pred = (i_scores >= 0).astype(int)
    a_correct = a_pred == y
    i_correct = i_pred == y

    n = len(paired)
    n_target = int((y == 1).sum())
    n_nontgt = int((y == 0).sum())
    print(f"Dev set: n={n} (target={n_target}, non-target={n_nontgt})")
    print("Per-modality accuracy at logit=0:")
    print(
        f"  audio: {int(a_correct.sum())}/{n} = {a_correct.mean() * 100:.1f}%  "
        f"(TP={int(((a_pred == 1) & (y == 1)).sum())} "
        f"FP={int(((a_pred == 1) & (y == 0)).sum())} "
        f"FN={int(((a_pred == 0) & (y == 1)).sum())} "
        f"TN={int(((a_pred == 0) & (y == 0)).sum())})"
    )
    print(
        f"  image: {int(i_correct.sum())}/{n} = {i_correct.mean() * 100:.1f}%  "
        f"(TP={int(((i_pred == 1) & (y == 1)).sum())} "
        f"FP={int(((i_pred == 1) & (y == 0)).sum())} "
        f"FN={int(((i_pred == 0) & (y == 1)).sum())} "
        f"TN={int(((i_pred == 0) & (y == 0)).sum())})"
    )

    n11 = int((a_correct & i_correct).sum())
    n10 = int((a_correct & ~i_correct).sum())
    n01 = int((~a_correct & i_correct).sum())
    n00 = int((~a_correct & ~i_correct).sum())

    print()
    print("Error correlation (2x2):")
    print(f"  {'':<22} {'image ✓':>10} {'image ✗':>10}")
    print(f"  {'audio ✓':<22} {n11:>10} {n10:>10}")
    print(f"  {'audio ✗':<22} {n01:>10} {n00:>10}")
    print()
    print(f"  Both modalities correct:    {n11}  ({n11 / n * 100:.1f}%)")
    print(f"  Audio rescues image error:  {n10}  ({n10 / n * 100:.1f}%)")
    print(f"  Image rescues audio error:  {n01}  ({n01 / n * 100:.1f}%)")
    print(f"  Both modalities wrong:      {n00}  ({n00 / n * 100:.1f}%)")
    print()
    print(f"  Fusion-rescuable errors: {n10 + n01} (one modality has the answer)")
    print(f"  Unrescuable errors:      {n00} (both wrong; fusion can't help)")

    n1x = n11 + n10
    n0x = n01 + n00
    nx1 = n11 + n01
    nx0 = n10 + n00
    denom = float(np.sqrt(n1x * n0x * nx1 * nx0))
    phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else float("nan")
    print()
    print(f"Phi correlation of correctness: {phi:+.3f}")
    print("  φ=+1 → modalities make identical errors (fusion can't help)")
    print("  φ= 0 → independent errors (typical fusion gain)")
    print("  φ<0 → complementary errors (fusion has lots to gain)")

    # Plot a 2x2 heatmap of the contingency table.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    table = np.array([[n11, n10], [n01, n00]])
    im = ax.imshow(table, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["image ✓", "image ✗"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["audio ✓", "audio ✗"])
    for i in range(2):
        for j in range(2):
            v = int(table[i, j])
            color = "white" if v > table.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{v}\n({v / n * 100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    ax.set_title(
        f"Audio × Image correctness on dev (n={n})\n"
        f"φ = {phi:+.3f} - fusion-rescuable: {n10 + n01}, unrescuable: {n00}"
    )
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
