#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
from experiments import compare_runs, run_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run even if metrics.json already exists",
    )
    args = ap.parse_args()

    metric_files: list[Path] = []
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        out_dir = args.out_root / cfg.name
        sidecar = out_dir / "metrics.json"
        if sidecar.is_file() and not args.force:
            print(f"[skip] {cfg.name}  (metrics.json present)")
            metric_files.append(sidecar)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg_path, out_dir / "config.toml")
        t0 = time.time()
        run_experiment(cfg, out_dir)
        print(f"[done] {cfg.name}  ({time.time() - t0:.1f}s)")
        metric_files.append(sidecar)

    print()
    print("=" * 90)
    print("Side-by-side dev metrics:")
    print("=" * 90)
    print(compare_runs(metric_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
