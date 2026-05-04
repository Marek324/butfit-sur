import librosa
import numpy as np
from scipy.signal import lfilter

from config import WaveformAugCfg


def augment_waveform(
    x: np.ndarray, sr: int, cfg: WaveformAugCfg, rng: np.random.Generator
) -> np.ndarray:
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

    snr_lo, snr_hi = cfg.noise_snr_db_range
    snr_db = float(rng.uniform(snr_lo, snr_hi))
    signal_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if signal_rms > 1e-10:
        noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        x = x + (noise_rms * rng.standard_normal(len(x))).astype(np.float32)
    return x.astype(np.float32)


def _peaking_eq(
    x: np.ndarray, sr: int, f0_hz: float, gain_db: float, q: float
) -> np.ndarray:
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
