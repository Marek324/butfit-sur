import argparse
import shutil
import sys
from pathlib import Path
import random
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import load_config  
from experiments import run_experiment  

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("artifacts"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out_dir = args.out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "config.toml")

    run_experiment(cfg, out_dir)
    print(f"\nArtifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
