#!/usr/bin/env python3
"""Evaluate a trained audio classifier.

  # Dev metrics
  uv run scripts/evaluate.py --config configs/audio_baseline.toml --split dev

  # Score an unlabeled directory (eval set on May 3rd) and emit result file
  uv run scripts/evaluate.py --config configs/audio_baseline.toml --input dataset/eval
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from audio_features import extract  # noqa: E402
from classifier import load, score  # noqa: E402
from config import Config, load_config  # noqa: E402
from data import enumerate_split, load_wav_mono16k  # noqa: E402
from eval import evaluate, format_metrics  # noqa: E402
from score import write_result_file  # noqa: E402


def extract_set(
    cfg: Config, dirs: list[Path], require_label: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    samples = enumerate_split(dirs, extension=".wav", label_required=require_label)
    print(f"  enumerated {len(samples)} samples")
    X: list[np.ndarray] = []
    y: list[int] = []
    names: list[str] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        x, sr = load_wav_mono16k(s.path)
        v = extract(x, sr, cfg.audio.vad, cfg.audio.features)
        X.append(v)
        y.append(s.label if s.label is not None else -1)
        names.append(s.name)
        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            print(f"  extract: {i + 1}/{len(samples)}  ({time.time() - t0:.1f}s)")
    return np.stack(X), np.asarray(y), names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument(
        "--split",
        choices=["dev"],
        default=None,
        help="Evaluate metrics on the dev split listed in the config",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Unlabeled directory of WAVs (test/eval mode)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Where to write the result file in --input mode",
    )
    args = ap.parse_args()

    if (args.split is None) == (args.input is None):
        ap.error("specify exactly one of --split dev | --input DIR")

    cfg = load_config(args.config)
    if cfg.modality != "audio":
        raise ValueError(f"this script handles modality='audio', got {cfg.modality!r}")

    model_path = args.artifacts / cfg.name / "model.pkl"
    print(f"Loading model: {model_path}")
    model = load(model_path)

    if args.split == "dev":
        print(f"Extracting dev features from {[str(d) for d in cfg.data.dev_dirs]} ...")
        X, y, _names = extract_set(cfg, cfg.data.dev_dirs, require_label=True)
        scores = score(model, X)
        m = evaluate(scores, y, threshold=cfg.output.hard_threshold_logit)
        print()
        print(format_metrics(m))
        print("\nScore distribution:")
        for lab, lab_name in [(1, "target  "), (0, "non-tgt ")]:
            s_lab = scores[y == lab]
            if len(s_lab) > 0:
                print(
                    f"  {lab_name} n={len(s_lab):3d}  "
                    f"min={s_lab.min():+7.3f}  med={float(np.median(s_lab)):+7.3f}  "
                    f"max={s_lab.max():+7.3f}"
                )
    else:
        assert args.input is not None
        print(f"Extracting eval features from {args.input} ...")
        X, _y, names = extract_set(cfg, [args.input], require_label=False)
        scores = score(model, X)
        out = args.results_dir / cfg.output.result_filename
        write_result_file(names, scores, cfg.output.hard_threshold_logit, out)
        n_pos = int((scores >= cfg.output.hard_threshold_logit).sum())
        print(f"Wrote {out}: {len(names)} segments  ({n_pos} predicted target)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
