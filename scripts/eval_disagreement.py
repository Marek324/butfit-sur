#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    ap.add_argument("audio", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("fusion", type=Path)
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/eval_disagreement.png")
    )
    args = ap.parse_args()

    a_names, a_s, a_d = _load(args.audio)
    i_names, i_s, i_d = _load(args.image)
    f_names, f_s, f_d = _load(args.fusion)

    if not (a_names == i_names == f_names):
        print("error: result files don't share segment ordering", file=sys.stderr)
        return 1

    n = len(a_d)
    counts: dict[tuple[int, int, int], int] = {}
    for a, i, f in zip(a_d, i_d, f_d, strict=True):
        counts[(int(a), int(i), int(f))] = counts.get((int(a), int(i), int(f)), 0) + 1

    print(f"Hard-decision combinations on eval (n={n}):")
    print(f"  {'audio':>5} {'image':>5} {'fusion':>6}  {'count':>6}  {'%':>6}  label")
    print("  " + "-" * 45)
    keys_in_order = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    label_map = {0: "N", 1: "T"}
    for k in keys_in_order:
        c = counts.get(k, 0)
        label = label_map[k[0]] + label_map[k[1]] + label_map[k[2]]
        print(
            f"  {k[0]:>5} {k[1]:>5} {k[2]:>6}  {c:>6}  {c / n * 100:>5.1f}%  {label}"
        )

    agree_NN = counts.get((0, 0, 0), 0) + counts.get((0, 0, 1), 0)
    agree_TT = counts.get((1, 1, 0), 0) + counts.get((1, 1, 1), 0)
    fusion_overrides_NN = counts.get((0, 0, 1), 0)
    fusion_overrides_TT = counts.get((1, 1, 0), 0)
    disagree = n - agree_NN - agree_TT
    fusion_with_audio = counts.get((0, 1, 0), 0) + counts.get((1, 0, 1), 0)
    fusion_with_image = counts.get((0, 1, 1), 0) + counts.get((1, 0, 0), 0)

    print()
    print("Audio + image agreement:")
    print(
        f"  Both → non-target: {agree_NN}  ({fusion_overrides_NN} fusion-overridden to target)"
    )
    print(
        f"  Both → target:     {agree_TT}  ({fusion_overrides_TT} fusion-overridden to non-target)"
    )
    print(f"  Disagree:          {disagree}")
    print(f"    Fusion sides with audio: {fusion_with_audio}")
    print(f"    Fusion sides with image: {fusion_with_image}")

    # Plot - stacked bars by (audio, image) consensus showing fusion outcome.
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    consensus_keys = [
        ("both → non-target", (0, 0)),
        ("audio T, image N", (1, 0)),
        ("audio N, image T", (0, 1)),
        ("both → target", (1, 1)),
    ]
    fusion_n = np.array([counts.get((a, i, 0), 0) for _, (a, i) in consensus_keys])
    fusion_t = np.array([counts.get((a, i, 1), 0) for _, (a, i) in consensus_keys])
    labels = [k[0] for k in consensus_keys]
    ax.bar(labels, fusion_n, color="tab:blue", label="fusion → non-target")
    ax.bar(labels, fusion_t, bottom=fusion_n, color="tab:orange", label="fusion → target")
    for i, (n_n, n_t) in enumerate(zip(fusion_n, fusion_t, strict=True)):
        if n_n > 0:
            ax.text(i, n_n / 2, str(int(n_n)), ha="center", va="center", color="white", fontsize=10)
        if n_t > 0:
            ax.text(
                i,
                n_n + n_t / 2,
                str(int(n_t)),
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
    ax.set_ylabel("eval segments")
    ax.set_title(
        f"Audio×Image consensus → Fusion outcome (n={n} eval segments)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
