#!/usr/bin/env python3
"""Train an audio classifier from a TOML config.

  uv run scripts/train.py --config configs/audio_baseline.toml

Persists model.pkl + train_features.npz + the config under artifacts/<name>/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from audio_augment import (  # noqa: E402
    augment_features,
    augment_waveform,
    estimate_within_speaker_sigmas,
)
from audio_features import extract, feature_names  # noqa: E402
from classifier import fit, save  # noqa: E402
from config import Config, load_config  # noqa: E402
from data import enumerate_split, load_wav_mono16k  # noqa: E402


def build_train_features(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str | None]]:
    samples = enumerate_split(cfg.data.train_dirs, extension=".wav")
    n_target = sum(1 for s in samples if s.label == 1)
    print(f"Enumerated {len(samples)} train samples (target={n_target}, non-target={len(samples) - n_target})")

    base_X: list[np.ndarray] = []
    base_y: list[int] = []
    base_names: list[str] = []
    base_spkrs: list[str | None] = []
    raws: list[tuple[np.ndarray, int]] = []

    t0 = time.time()
    for i, s in enumerate(samples):
        x, sr = load_wav_mono16k(s.path)
        v = extract(x, sr, cfg.audio.vad, cfg.audio.features)
        base_X.append(v)
        assert s.label is not None
        base_y.append(s.label)
        base_names.append(s.name)
        base_spkrs.append(s.speaker_id)
        raws.append((x, sr))
        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            print(f"  base extract: {i + 1}/{len(samples)}  ({time.time() - t0:.1f}s)")

    X = list(base_X)
    y = list(base_y)
    names = list(base_names)
    spkrs = list(base_spkrs)

    aug = cfg.audio.augment_train
    if aug.enabled:
        rng = np.random.default_rng(aug.seed)
        n_t = aug.n_per_target
        n_nt = aug.n_per_nontarget
        total = n_target * n_t + (len(samples) - n_target) * n_nt
        if total > 0:
            print(f"Generating {total} waveform-augmented variants ...")
            t0 = time.time()
            done = 0
            for v_base, lab, nm, spk, (x, sr) in zip(
                base_X, base_y, base_names, base_spkrs, raws, strict=True
            ):
                n_aug = n_t if lab == 1 else n_nt
                for k in range(n_aug):
                    xa = augment_waveform(x, sr, aug.waveform, rng)
                    va = extract(xa, sr, cfg.audio.vad, cfg.audio.features)
                    X.append(va)
                    y.append(lab)
                    names.append(f"{nm}_aug{k}")
                    spkrs.append(spk)
                    done += 1
                    if done % 50 == 0:
                        print(f"  aug: {done}/{total}  ({time.time() - t0:.1f}s)")

    X_arr = np.stack(X)
    y_arr = np.asarray(y)

    fs = aug.feature_space
    if fs.enabled:
        print("Applying feature-space augmentation ...")
        sigmas = estimate_within_speaker_sigmas(np.stack(base_X), base_spkrs)
        rng_fs = np.random.default_rng(aug.seed + 1)
        extra = np.stack([augment_features(row, sigmas, fs, rng_fs) for row in X_arr])
        X_arr = np.concatenate([X_arr, extra], axis=0)
        y_arr = np.concatenate([y_arr, y_arr])
        names = names + [n + "_fs" for n in names]
        spkrs = spkrs + spkrs

    return X_arr, y_arr, names, spkrs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.modality != "audio":
        raise ValueError(f"this script handles modality='audio', got {cfg.modality!r}")

    X, y, names, _ = build_train_features(cfg)
    print(f"Final training set: X={X.shape}, target={(y == 1).sum()}, non-target={(y == 0).sum()}")

    print("Fitting classifier ...")
    fnames = feature_names(cfg.audio.features)
    model = fit(X, y, fnames, cfg.classifier)

    out_dir = args.out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save(model, out_dir / "model.pkl")
    np.savez(
        out_dir / "train_features.npz",
        X=X,
        y=y,
        names=np.array(names),
    )
    shutil.copyfile(args.config, out_dir / "config.toml")
    print(f"Saved model + features + config to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
