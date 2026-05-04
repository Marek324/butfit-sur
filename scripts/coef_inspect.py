#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_pkl(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/coefficients.png")
    )
    ap.add_argument("--top-k", type=int, default=15, help="HOG dims to show each side")
    args = ap.parse_args()

    audio = _load_pkl(args.artifacts / "audio_baseline_logreg" / "model.pkl")
    image = _load_pkl(args.artifacts / "image_baseline_logreg" / "model.pkl")
    fusion = _load_pkl(args.artifacts / "fusion_baseline_logreg" / "model.pkl")

    a_coef = np.asarray(audio.estimator.coef_).ravel()
    i_coef = np.asarray(image.estimator.coef_).ravel()
    f_coef = np.asarray(fusion.estimator.coef_).ravel()
    a_int = float(np.asarray(audio.estimator.intercept_).ravel()[0])
    i_int = float(np.asarray(image.estimator.intercept_).ravel()[0])
    f_int = float(np.asarray(fusion.estimator.intercept_).ravel()[0])

    print(f"Audio coefficients (n={len(a_coef)}, intercept={a_int:+.4f}):")
    print(f"  {'feature':<28} {'coef':>10}")
    pairs = list(zip(audio.feature_names, a_coef))
    pairs.sort(key=lambda kv: -kv[1])
    for name, c in pairs:
        print(f"  {name:<28} {c:+10.4f}")

    print()
    print(
        f"Image HOG coefficients (n={len(i_coef)}, intercept={i_int:+.4f}, "
        f"|w|: min={np.abs(i_coef).min():.4f} med={np.median(np.abs(i_coef)):.4f} "
        f"max={np.abs(i_coef).max():.4f}):"
    )
    top_pos = np.argsort(i_coef)[-args.top_k:][::-1]
    top_neg = np.argsort(i_coef)[: args.top_k]
    print(f"  Top {args.top_k} positive (target-favoring) HOG dims:")
    for i in top_pos:
        print(f"    HOG[{i:>5d}]  {i_coef[i]:+.4f}")
    print(f"  Top {args.top_k} negative (impostor-favoring) HOG dims:")
    for i in top_neg:
        print(f"    HOG[{i:>5d}]  {i_coef[i]:+.4f}")

    print()
    print(f"Fusion coefficients (intercept={f_int:+.4f}):")
    print(f"  audio_score (zscored)  {f_coef[0]:+.4f}")
    print(f"  image_score (zscored)  {f_coef[1]:+.4f}")
    if abs(f_coef[1]) > abs(f_coef[0]):
        print(
            f"  → fusion leans on image  (|w_image|/|w_audio| = "
            f"{abs(f_coef[1]) / abs(f_coef[0]):.2f}×)"
        )
    else:
        print(
            f"  → fusion leans on audio  (|w_audio|/|w_image| = "
            f"{abs(f_coef[0]) / abs(f_coef[1]):.2f}×)"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, 0.3 * len(audio.feature_names) + 2)))

    # Audio bar chart, sorted by coef value.
    ax = axes[0]
    sort_idx = np.argsort(a_coef)
    names_sorted = [audio.feature_names[i] for i in sort_idx]
    coef_sorted = a_coef[sort_idx]
    y_pos = np.arange(len(coef_sorted))
    ax.barh(
        y_pos,
        coef_sorted,
        color=["steelblue" if c >= 0 else "indianred" for c in coef_sorted],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("coefficient (>0 → favors target)")
    ax.set_title(f"Audio (n={len(a_coef)}, intercept={a_int:+.3f})")
    ax.grid(True, axis="x", alpha=0.3)

    # Image - top-K each + histogram inset.
    ax = axes[1]
    shown = np.concatenate([top_pos, top_neg])
    shown_coef = i_coef[shown]
    order = np.argsort(shown_coef)
    shown = shown[order]
    shown_coef = shown_coef[order]
    y_pos = np.arange(len(shown))
    ax.barh(
        y_pos,
        shown_coef,
        color=["steelblue" if c >= 0 else "indianred" for c in shown_coef],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"HOG[{i}]" for i in shown], fontsize=8)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("coefficient")
    ax.set_title(
        f"Image: top-{args.top_k} per direction (of {len(i_coef)}), intercept={i_int:+.3f}"
    )
    ax.grid(True, axis="x", alpha=0.3)
    # Inset histogram of all weights.
    ax_inset = ax.inset_axes((0.62, 0.05, 0.36, 0.32))
    ax_inset.hist(i_coef, bins=80, color="0.6", edgecolor="white")
    ax_inset.axvline(0, color="0.3", linewidth=0.8)
    ax_inset.set_title("all HOG weights", fontsize=8)
    ax_inset.tick_params(labelsize=7)

    # Fusion.
    ax = axes[2]
    bars = ax.bar(["audio", "image"], f_coef, color=["tab:blue", "tab:orange"])
    for b, v in zip(bars, f_coef):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (0.05 if v >= 0 else -0.15),
            f"{v:+.3f}",
            ha="center",
            fontsize=10,
        )
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_ylabel("coefficient (>0 → favors target)")
    ax.set_title(f"Fusion (zscored inputs), intercept={f_int:+.3f}")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
