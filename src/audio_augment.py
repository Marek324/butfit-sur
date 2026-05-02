"""Audio augmentations chosen to be invariant under speaker identity.

Safe transforms only: time stretch (pitch-preserving), gain, polarity, random
crop, peaking EQ. Each leaves the (log F0, jitter, shimmer, voicing ratio,
CMVN-MFCC means) feature vector approximately unchanged in expectation.

Pitch shift, reverb, low-SNR noise, and SpecAugment-style masking are
DELIBERATELY NOT IMPLEMENTED — they would change speaker-defining features.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import lfilter

from config import FeatureSpaceAugCfg, WaveformAugCfg


def augment_waveform(
    x: np.ndarray, sr: int, cfg: WaveformAugCfg, rng: np.random.Generator
) -> np.ndarray:
    """Apply one random sample from the waveform-augmentation envelope."""
    x = x.astype(np.float32)

    if rng.random() < cfg.polarity_flip_p:
        x = -x

    rate = float(rng.uniform(*cfg.time_stretch_range))
    if abs(rate - 1.0) > 1e-3:
        x = librosa.effects.time_stretch(x, rate=rate).astype(np.float32)

    min_n = int(sr * cfg.crop_min_s)
    if len(x) > min_n:
        max_start = len(x) - min_n
        start = int(rng.integers(0, max_start + 1))
        max_len = len(x) - start
        length = int(rng.integers(min_n, max_len + 1))
        x = x[start : start + length]

    if cfg.eq_enabled:
        f0 = float(rng.uniform(*cfg.eq_freq_hz_range))
        g_db = float(rng.uniform(*cfg.eq_gain_db_range))
        x = _peaking_eq(x, sr, f0, g_db, cfg.eq_q)

    g_db = float(rng.uniform(*cfg.gain_db_range))
    x = x * float(10 ** (g_db / 20))
    return x.astype(np.float32)


def _peaking_eq(x: np.ndarray, sr: int, f0_hz: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking biquad."""
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * f0_hz / sr
    cos_w = np.cos(omega)
    alpha = np.sin(omega) / (2 * q)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, x).astype(np.float32)


def augment_features(
    v: np.ndarray, sigmas: np.ndarray, cfg: FeatureSpaceAugCfg, rng: np.random.Generator
) -> np.ndarray:
    """Per-dim Gaussian noise. sigmas: (D,) within-speaker std estimate."""
    if not cfg.enabled:
        return v
    noise = rng.standard_normal(v.shape).astype(v.dtype)
    return v + (cfg.sigma_multiplier * sigmas).astype(v.dtype) * noise


def estimate_within_speaker_sigmas(
    features: np.ndarray, speaker_ids: list[str | None]
) -> np.ndarray:
    """Mean over speakers of per-speaker std (per dim).

    features: (N, D). speaker_ids: length N; speakers with <2 utterances are skipped.
    Falls back to global std if no speaker has ≥2 utterances.
    """
    by_speaker: dict[str, list[np.ndarray]] = {}
    for f, spk in zip(features, speaker_ids, strict=True):
        if spk is None:
            continue
        by_speaker.setdefault(spk, []).append(f)
    stds: list[np.ndarray] = []
    for vs in by_speaker.values():
        if len(vs) >= 2:
            stds.append(np.stack(vs).std(axis=0))
    if not stds:
        return features.std(axis=0)
    return np.mean(np.stack(stds), axis=0)
