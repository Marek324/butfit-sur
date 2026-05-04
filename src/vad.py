from __future__ import annotations

import numpy as np

from config import VadCfg


def _frame(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return x[idx]


def _features(x: np.ndarray, sr: int, cfg: VadCfg) -> tuple[np.ndarray, np.ndarray]:
    pre = np.empty_like(x)
    pre[0] = x[0]
    pre[1:] = x[1:] - cfg.preemphasis * x[:-1]
    flen = int(sr * cfg.frame_ms / 1000)
    hlen = int(sr * cfg.hop_ms / 1000)
    frames = _frame(pre, flen, hlen)
    eps = 1e-10
    energy_db = 10.0 * np.log10(np.sum(frames * frames, axis=1) + eps)
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1) / (flen - 1)
    return energy_db, zcr


def _smooth_runs(mask: np.ndarray, min_speech: int, min_silence: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_speech:
                out[i:j] = False
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) < min_silence:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def speech_mask(x: np.ndarray, sr: int, cfg: VadCfg) -> np.ndarray:
    if cfg.algorithm == "off":
        # Same number of frames the energy_zcr branch would produce, all True.
        flen = int(sr * cfg.frame_ms / 1000)
        hlen = int(sr * cfg.hop_ms / 1000)
        n_frames = max(0, 1 + (max(len(x), flen) - flen) // hlen)
        return np.ones(n_frames, dtype=bool)
    if cfg.algorithm != "energy_zcr":
        raise ValueError(f"unsupported VAD algorithm: {cfg.algorithm}")
    energy_db, zcr = _features(x, sr, cfg)
    if len(energy_db) == 0:
        return np.zeros(0, dtype=bool)
    noise_db = float(np.percentile(energy_db, cfg.noise_percentile))
    e_strong = energy_db > noise_db + cfg.energy_margin_db
    e_loose = energy_db > noise_db + cfg.zcr_aid_margin_db
    fricative = e_loose & (zcr >= cfg.zcr_low) & (zcr <= cfg.zcr_high)
    raw = e_strong | fricative
    min_sp = max(1, cfg.min_speech_ms // cfg.hop_ms)
    min_sl = max(1, cfg.min_silence_ms // cfg.hop_ms)
    return _smooth_runs(raw, min_sp, min_sl)
