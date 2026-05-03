#!/usr/bin/env python3
"""Score histogram for eval-set result files.

  uv run scripts/plot_eval_scores.py results/*.txt --output artifacts/eval_scores.png

Reads one or more result files in `<segment> <score> <decision>` format,
and plots a per-system histogram of scores with the 50/50-prior decision
threshold (score = 0) marked. Useful sanity check after `evaluate.py --split eval`:
- if the histogram is bimodal with a clean separation, the model has high confidence
- if it's a single fat blob, the model is uncertain on most of the eval set
- the fraction above 0 tells you the predicted-target rate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_scores(p: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    decisions: list[int] = []
    with p.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            scores.append(float(parts[1]))
            decisions.append(int(parts[2]))
    return np.asarray(scores), np.asarray(decisions)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/eval_scores.png")
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="decision threshold to mark on each histogram (default 0.0 — 50/50 prior)",
    )
    args = ap.parse_args()

    files: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in args.results:
        if not p.is_file():
            print(f"  skip {p}: not a file", file=sys.stderr)
            continue
        scores, decisions = _load_scores(p)
        files.append((p.stem, scores, decisions))
        n_pos = int((decisions == 1).sum())
        print(
            f"{p.name:<35} n={len(scores)}  predicted_target={n_pos} ({n_pos / len(scores) * 100:.1f}%)  "
            f"med={np.median(scores):+.3f}"
        )

    if not files:
        print("no result files to plot", file=sys.stderr)
        return 1

    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    # Use a common range across systems for easier visual comparison.
    all_scores = np.concatenate([s for _, s, _ in files])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    bins = np.linspace(lo, hi, 60)

    for i, (label, scores, decisions) in enumerate(files):
        ax = axes[i]
        n_pos = int((decisions == 1).sum())
        ax.hist(scores, bins=bins, color=cmap(i % 10), alpha=0.85, edgecolor="white")
        ax.axvline(args.threshold, color="0.2", linestyle="--", linewidth=1.2)
        ax.set_ylabel("count")
        ax.set_title(
            f"{label}  —  n={len(scores)}, predicted target={n_pos} ({n_pos / len(scores) * 100:.1f}%)"
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("score (decision_function output)")
    fig.suptitle("Eval-set score distributions (threshold dashed at score = 0)", y=1.0)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
