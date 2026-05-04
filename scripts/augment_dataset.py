#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from audio_augment import augment_waveform  # noqa: E402
from config import (  # noqa: E402
    AudioAugmentConfig,
    ImageAugmentConfig,
    load_augment_config,
)
from data import EXPECTED_SR, load_png, load_wav_mono16k  # noqa: E402
from image_augment import augment as augment_image_fn  # noqa: E402


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_clipped = np.clip(x, -1.0, 1.0)
    x_int16 = (x_clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(x_int16.tobytes())


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def _audio_jobs(cfg: AudioAugmentConfig) -> list[tuple[Path, Path, int]]:
    return [
        (cfg.source_target_dir, cfg.output_target_dir, cfg.n_per_target),
        (cfg.source_nontarget_dir, cfg.output_nontarget_dir, cfg.n_per_nontarget),
    ]


def _image_jobs(cfg: ImageAugmentConfig) -> list[tuple[Path, Path, int]]:
    return [
        (cfg.source_target_dir, cfg.output_target_dir, cfg.n_per_target),
        (cfg.source_nontarget_dir, cfg.output_nontarget_dir, cfg.n_per_nontarget),
    ]


def _augment_audio(cfg: AudioAugmentConfig) -> int:
    rng = np.random.default_rng(cfg.seed)
    total_in = total_out = 0
    t0 = time.time()
    for src_dir, out_dir, n_aug in _audio_jobs(cfg):
        if n_aug <= 0:
            print(f"  skip {src_dir} (n_per_*=0)")
            continue
        if not src_dir.is_dir():
            raise FileNotFoundError(f"source dir not found: {src_dir}")
        wavs = sorted(src_dir.glob("*.wav"))
        print(f"  {src_dir}: {len(wavs)} sources × {n_aug} variants → {out_dir}")
        for p in wavs:
            x, sr = load_wav_mono16k(p)
            assert sr == EXPECTED_SR
            total_in += 1
            for k in range(n_aug):
                xa = augment_waveform(x, sr, cfg.waveform, rng)
                out_path = out_dir / f"{p.stem}_aug{k:02d}.wav"
                _write_wav(out_path, xa, sr)
                total_out += 1
    print(f"audio aug: {total_in} sources → {total_out} variants  ({time.time() - t0:.1f}s)")
    return total_out


def _augment_image(cfg: ImageAugmentConfig) -> int:
    rng = np.random.default_rng(cfg.seed)
    total_in = total_out = 0
    t0 = time.time()
    for src_dir, out_dir, n_aug in _image_jobs(cfg):
        if n_aug <= 0:
            print(f"  skip {src_dir} (n_per_*=0)")
            continue
        if not src_dir.is_dir():
            raise FileNotFoundError(f"source dir not found: {src_dir}")
        pngs = sorted(src_dir.glob("*.png"))
        print(f"  {src_dir}: {len(pngs)} sources × {n_aug} variants → {out_dir}")
        for p in pngs:
            img = load_png(p, grayscale=False)
            total_in += 1
            for k in range(n_aug):
                ia = augment_image_fn(img, cfg.envelope, rng)
                out_path = out_dir / f"{p.stem}_aug{k:02d}.png"
                _write_png(out_path, ia)
                total_out += 1
    print(f"image aug: {total_in} sources → {total_out} variants  ({time.time() - t0:.1f}s)")
    return total_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_augment_config(args.config)
    if isinstance(cfg, AudioAugmentConfig):
        _augment_audio(cfg)
    elif isinstance(cfg, ImageAugmentConfig):
        _augment_image(cfg)
    else:
        raise ValueError(f"unknown augment config type: {type(cfg).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
