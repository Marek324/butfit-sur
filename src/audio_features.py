"""Per-utterance audio feature extraction.

Default 10-dim vector for single-target verification:
  log F0 mean, log F0 std, jitter (local), shimmer (local), voicing ratio,
  + first 5 CMVN-MFCC means over voiced-speech frames.

All features are utterance-level aggregates over VAD-detected speech and
pyin-voiced frames. Pure DSP — librosa.pyin, librosa.feature.mfcc, and
hand-rolled jitter/shimmer. No pretrained anything.
"""

from __future__ import annotations

import librosa
import numpy as np

from config import AudioFeaturesCfg, VadCfg
from vad import speech_mask


def extract(x: np.ndarray, sr: int, vad_cfg: VadCfg, feat_cfg: AudioFeaturesCfg) -> np.ndarray:
    """Compute the per-utterance feature vector.

    Output order: feat_cfg.include items in given order, then mfcc_n_means
    MFCC mean coefficients (low-order first).
    """
    hop = int(sr * vad_cfg.hop_ms / 1000)
    win = int(sr * vad_cfg.frame_ms / 1000)

    sp_mask = speech_mask(x, sr, vad_cfg)

    if feat_cfg.f0_method != "pyin":
        raise ValueError(f"unsupported f0_method: {feat_cfg.f0_method}")
    f0, voiced_flag, _ = librosa.pyin(
        x.astype(np.float32),
        fmin=feat_cfg.f0_min_hz,
        fmax=feat_cfg.f0_max_hz,
        sr=sr,
        frame_length=2048,
        hop_length=hop,
    )
    voiced_flag = voiced_flag.astype(bool) if voiced_flag is not None else np.zeros_like(f0, dtype=bool)

    n = min(len(sp_mask), len(f0))
    sp_mask = sp_mask[:n]
    f0 = f0[:n]
    voiced_flag = voiced_flag[:n]
    voiced_in_speech = sp_mask & voiced_flag & ~np.isnan(f0)

    rms = librosa.feature.rms(y=x.astype(np.float32), frame_length=win, hop_length=hop)[0]
    rms = rms[:n] if len(rms) >= n else np.pad(rms, (0, n - len(rms)))

    feats: list[float] = []
    for name in feat_cfg.include:
        feats.append(_compute_named(name, f0, voiced_in_speech, sp_mask, rms))

    if feat_cfg.mfcc_n_means > 0:
        feats.extend(_mfcc_means(x, sr, hop, win, sp_mask, feat_cfg).tolist())

    return np.asarray(feats, dtype=np.float32)


def feature_names(feat_cfg: AudioFeaturesCfg) -> list[str]:
    """Order-aligned with extract() output, useful for debugging / reporting."""
    names = list(feat_cfg.include)
    names.extend([f"mfcc_mean_{i}" for i in range(feat_cfg.mfcc_n_means)])
    return names


def _compute_named(
    name: str,
    f0: np.ndarray,
    voiced_in_speech: np.ndarray,
    sp_mask: np.ndarray,
    rms: np.ndarray,
) -> float:
    if name == "log_f0_mean":
        v = f0[voiced_in_speech]
        return float(np.mean(np.log(v))) if len(v) > 0 else 0.0
    if name == "log_f0_std":
        v = f0[voiced_in_speech]
        return float(np.std(np.log(v))) if len(v) >= 2 else 0.0
    if name == "jitter_local":
        v = f0[voiced_in_speech]
        if len(v) < 2:
            return 0.0
        periods = 1.0 / v
        denom = float(np.mean(periods))
        if denom <= 0:
            return 0.0
        return float(np.mean(np.abs(np.diff(periods))) / denom)
    if name == "shimmer_local":
        v = rms[voiced_in_speech]
        if len(v) < 2:
            return 0.0
        denom = float(np.mean(v))
        if denom <= 1e-10:
            return 0.0
        return float(np.mean(np.abs(np.diff(v))) / denom)
    if name == "voicing_ratio":
        sp_total = int(sp_mask.sum())
        if sp_total == 0:
            return 0.0
        return float(int(voiced_in_speech.sum()) / sp_total)
    raise ValueError(f"unknown feature: {name}")


def _mfcc_means(
    x: np.ndarray,
    sr: int,
    hop: int,
    win: int,
    sp_mask: np.ndarray,
    cfg: AudioFeaturesCfg,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=x.astype(np.float32),
        sr=sr,
        n_mfcc=cfg.mfcc_n,
        n_mels=cfg.mel_filters,
        hop_length=hop,
        n_fft=2048,
        win_length=win,
    )
    m = min(mfcc.shape[1], len(sp_mask))
    mfcc = mfcc[:, :m]
    in_speech = sp_mask[:m]
    if cfg.cmvn == "per_utterance":
        mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)
    elif cfg.cmvn != "none":
        raise ValueError(f"unknown cmvn mode: {cfg.cmvn}")
    if in_speech.any():
        mean = mfcc[:, in_speech].mean(axis=1)
    else:
        mean = np.zeros(cfg.mfcc_n, dtype=mfcc.dtype)
    return mean[: cfg.mfcc_n_means]
