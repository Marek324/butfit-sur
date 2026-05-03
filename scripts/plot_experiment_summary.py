#!/usr/bin/env python3
"""Bar-chart summary of experiment results from metrics.json sidecars.

  uv run scripts/plot_experiment_summary.py \\
      configs/experiments/image_*.toml configs/image_baseline.toml \\
      --output artifacts/experiments/image_summary.png

Reads `artifacts/<name>/metrics.json` for each given config (its `name` field
keys the artifact dir) and emits a 2-panel figure: AUC bar chart on the left,
EER% bar chart on the right. Compact birds-eye view that complements
plot_curves.py (which is per-system curves, not aggregates).

If `--dev-dirs` is given, *re-scores* dev on those dirs and uses those
metrics — useful for stress-dev summaries since metrics.json sidecars only
hold clean-dev numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from config import load_config  # noqa: E402

# We re-use the plot_curves.py scoring helpers when --dev-dirs is given.
from plot_curves import score_dev  # noqa: E402


def _eer(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan"), float("nan")
    fpr, tpr, _ = roc_curve(y, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    auc = float(roc_auc_score(y, scores))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return auc, eer


def _from_sidecar(artifacts_root: Path, name: str) -> tuple[float, float] | None:
    p = artifacts_root / name / "metrics.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text())
    dev = payload.get("dev", {})
    auc = dev.get("auc")
    eer = dev.get("eer")
    if auc is None or eer is None:
        return None
    return float(auc), float(eer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--output", type=Path, default=Path("artifacts/experiment_summary.png")
    )
    ap.add_argument(
        "--dev-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="re-score dev on these dirs instead of using sidecar (use for stress-dev summary)",
    )
    ap.add_argument("--title", type=str, default="Experiment summary")
    args = ap.parse_args()

    names: list[str] = []
    aucs: list[float] = []
    eers: list[float] = []
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        if args.dev_dirs:
            scores, y = score_dev(cfg, args.artifacts, dev_dirs=list(args.dev_dirs))
            auc, eer = _eer(scores, y)
        else:
            res = _from_sidecar(args.artifacts, cfg.name)
            if res is None:
                print(f"  no metrics for {cfg.name}, skipping", file=sys.stderr)
                continue
            auc, eer = res
        names.append(cfg.name)
        aucs.append(auc)
        eers.append(eer * 100.0)
        print(f"{cfg.name:<40} AUC={auc:.4f}  EER={eer * 100:.2f}%")

    if not names:
        print("no metrics to plot", file=sys.stderr)
        return 1

    fig, (ax_auc, ax_eer) = plt.subplots(1, 2, figsize=(14, max(4, 0.4 * len(names) + 2)))
    y_pos = np.arange(len(names))[::-1]  # top-down ordering

    ax_auc.barh(y_pos, aucs, color="steelblue")
    ax_auc.set_yticks(y_pos)
    ax_auc.set_yticklabels(names, fontsize=9)
    ax_auc.set_xlim(0.5, 1.0)
    ax_auc.axvline(0.5, color="0.5", linestyle="--", linewidth=1, alpha=0.6)
    ax_auc.set_xlabel("AUC")
    ax_auc.set_title("AUC")
    ax_auc.grid(True, axis="x", alpha=0.3)
    for yv, av in zip(y_pos, aucs, strict=True):
        ax_auc.text(av + 0.005, yv, f"{av:.3f}", va="center", fontsize=8)

    ax_eer.barh(y_pos, eers, color="indianred")
    ax_eer.set_yticks(y_pos)
    ax_eer.set_yticklabels([])
    ax_eer.set_xlabel("EER (%)")
    ax_eer.set_title("EER %")
    ax_eer.grid(True, axis="x", alpha=0.3)
    for yv, ev in zip(y_pos, eers, strict=True):
        ax_eer.text(ev + 0.5, yv, f"{ev:.2f}%", va="center", fontsize=8)

    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
