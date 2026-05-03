#!/usr/bin/env python3
"""Train a classifier from a TOML config (audio | image | fusion).

  uv run scripts/train.py --config configs/audio_baseline.toml
  uv run scripts/train.py --config configs/image_baseline.toml
  uv run scripts/train.py --config configs/fusion_baseline.toml

Persists artifacts/<name>/{model.pkl, train_features.npz, metrics.json, config.toml}.
The metrics.json sidecar makes side-by-side comparisons cheap (see
scripts/run_experiments.py).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
from experiments import run_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "config.toml")

    run_experiment(cfg, out_dir)
    print(f"\nArtifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
