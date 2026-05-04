import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _load(p: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    scores: list[float] = []
    decisions: list[int] = []
    with p.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            names.append(parts[0])
            scores.append(float(parts[1]))
            decisions.append(int(parts[2]))
    return names, np.asarray(scores), np.asarray(decisions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/score_correlation.png")
    )
    args = ap.parse_args()

    systems: list[tuple[str, list[str], np.ndarray, np.ndarray]] = []
    for p in args.results:
        names, scores, decisions = _load(p)
        systems.append((p.stem, names, scores, decisions))

    if len(systems) < 2:
        print("need at least 2 result files for pairwise correlation", file=sys.stderr)
        return 1

    print()
    print("Pairwise correlation:")
    print("=" * 95)
    print(
        f"{'pair':<55} {'Pearson':>9} {'Spearman':>10} {'agree%':>9} {'both+':>7} {'both-':>7}"
    )
    print("-" * 95)
    for (a_label, a_names, a_s, a_d), (b_label, b_names, b_s, b_d) in combinations(
        systems, 2
    ):
        # Align by segment name (intersection).
        a_idx = {n: i for i, n in enumerate(a_names)}
        common = [n for n in b_names if n in a_idx]
        if len(common) < 2:
            print(f"{a_label}↔{b_label}: <2 shared segments, skipping")
            continue
        ai = np.array([a_idx[n] for n in common])
        bi = np.array([i for i, n in enumerate(b_names) if n in a_idx])
        a_s_c = a_s[ai]
        b_s_c = b_s[bi]
        a_d_c = a_d[ai]
        b_d_c = b_d[bi]

        pr = float(pearsonr(a_s_c, b_s_c).statistic)
        sr = float(spearmanr(a_s_c, b_s_c).statistic)
        agree = float((a_d_c == b_d_c).mean()) * 100
        both_pos = int(((a_d_c == 1) & (b_d_c == 1)).sum())
        both_neg = int(((a_d_c == 0) & (b_d_c == 0)).sum())

        pair = f"{a_label} ↔ {b_label}"
        print(f"{pair:<55} {pr:>9.3f} {sr:>10.3f} {agree:>8.1f}% {both_pos:>7d} {both_neg:>7d}")

    n_pairs = len(systems) * (len(systems) - 1) // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 5), squeeze=False)
    axes = axes[0]
    for ax, ((a_label, a_names, a_s, a_d), (b_label, b_names, b_s, b_d)) in zip(
        axes, combinations(systems, 2), strict=False
    ):
        a_idx = {n: i for i, n in enumerate(a_names)}
        common = [n for n in b_names if n in a_idx]
        if len(common) < 2:
            continue
        ai = np.array([a_idx[n] for n in common])
        bi = np.array([i for i, n in enumerate(b_names) if n in a_idx])
        a_s_c = a_s[ai]
        b_s_c = b_s[bi]
        a_d_c = a_d[ai]
        b_d_c = b_d[bi]

        # Color by agreement type.
        both_pos = (a_d_c == 1) & (b_d_c == 1)
        both_neg = (a_d_c == 0) & (b_d_c == 0)
        disagree = ~(both_pos | both_neg)

        ax.scatter(
            a_s_c[both_neg], b_s_c[both_neg], s=8, alpha=0.5, color="tab:blue", label="both ⇒ non-target"
        )
        ax.scatter(
            a_s_c[both_pos], b_s_c[both_pos], s=12, alpha=0.7, color="tab:green", label="both ⇒ target"
        )
        ax.scatter(
            a_s_c[disagree], b_s_c[disagree], s=12, alpha=0.7, color="tab:red", label="disagree"
        )
        ax.axvline(0, color="0.3", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="0.3", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f"{a_label} score")
        ax.set_ylabel(f"{b_label} score")
        pr = float(pearsonr(a_s_c, b_s_c).statistic)
        sr = float(spearmanr(a_s_c, b_s_c).statistic)
        ax.set_title(f"{a_label} vs {b_label}\nPearson={pr:.3f}  Spearman={sr:.3f}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Eval-set score correlation across modalities (threshold dashed at 0)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
