from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from config import AudioFeaturesCfg, VadCfg
from vad import speech_mask


@dataclass(frozen=True)
class AudioPre:
    sp_mask: np.ndarray         # (T,) bool   - VAD speech mask
    f0: np.ndarray              # (T,) float  - pyin F0 in Hz, NaN where unvoiced
    voiced_flag: np.ndarray     # (T,) bool   - pyin voiced indicator
    voiced_in_speech: np.ndarray  # (T,) bool - sp_mask & voiced_flag & finite(f0)
    rms: np.ndarray             # (T,) float  - frame RMS
    mfcc: np.ndarray            # (n_mfcc, T) - CMVN'd if cmvn_per_utterance else raw
    formants_per_frame: np.ndarray   # (T, n_formants) - LPC pole frequencies, NaN-padded
    centroid_per_frame: np.ndarray   # (T,) - spectral centroid in Hz


def extract(
    x: np.ndarray, sr: int, vad_cfg: VadCfg, feat_cfg: AudioFeaturesCfg
) -> np.ndarray:
    pre = extract_pre(x, sr, vad_cfg, feat_cfg)
    return aggregate(pre, feat_cfg)


def extract_pre(
    x: np.ndarray, sr: int, vad_cfg: VadCfg, feat_cfg: AudioFeaturesCfg
) -> AudioPre:
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
    voiced_flag = (
        voiced_flag.astype(bool)
        if voiced_flag is not None
        else np.zeros_like(f0, dtype=bool)
    )

    n = min(len(sp_mask), len(f0))
    sp_mask = sp_mask[:n]
    f0 = f0[:n]
    voiced_flag = voiced_flag[:n]
    voiced_in_speech = sp_mask & voiced_flag & ~np.isnan(f0)

    rms = librosa.feature.rms(y=x.astype(np.float32), frame_length=win, hop_length=hop)[
        0
    ]
    rms = rms[:n] if len(rms) >= n else np.pad(rms, (0, n - len(rms)))

    mfcc = librosa.feature.mfcc(
        y=x.astype(np.float32),
        sr=sr,
        n_mfcc=feat_cfg.mfcc_n,
        n_mels=feat_cfg.mel_filters,
        hop_length=hop,
        n_fft=2048,
        win_length=win,
    )
    if feat_cfg.cmvn == "per_utterance":
        mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)
    elif feat_cfg.cmvn != "none":
        raise ValueError(f"unknown cmvn mode: {feat_cfg.cmvn}")
    mfcc = mfcc[:, :n] if mfcc.shape[1] >= n else np.pad(mfcc, ((0, 0), (0, n - mfcc.shape[1])))

    formants_per_frame = _formants_per_frame(
        x, sr, hop, win, lpc_order=12, n_formants=4, voiced_mask=voiced_in_speech
    )
    centroid_per_frame = _spectral_centroid_per_frame(x, sr, hop, win)

    # Trim per-frame arrays to the common length.
    formants_per_frame = formants_per_frame[:n]
    centroid_per_frame = centroid_per_frame[:n]

    return AudioPre(
        sp_mask=sp_mask,
        f0=f0,
        voiced_flag=voiced_flag,
        voiced_in_speech=voiced_in_speech,
        rms=rms,
        mfcc=mfcc.astype(np.float32, copy=False),
        formants_per_frame=formants_per_frame,
        centroid_per_frame=centroid_per_frame.astype(np.float32, copy=False),
    )


def aggregate(pre: AudioPre, feat_cfg: AudioFeaturesCfg) -> np.ndarray:
    feats: list[float] = []
    for name in feat_cfg.include:
        feats.append(
            _compute_named(
                name,
                pre.f0,
                pre.voiced_in_speech,
                pre.sp_mask,
                pre.rms,
                formants_per_frame=pre.formants_per_frame,
                centroid_per_frame=pre.centroid_per_frame,
            )
        )

    if feat_cfg.mfcc_n_means > 0:
        in_speech = pre.sp_mask
        if in_speech.any():
            mean = pre.mfcc[:, in_speech].mean(axis=1)
        else:
            mean = np.zeros(pre.mfcc.shape[0], dtype=pre.mfcc.dtype)
        feats.extend(mean[: feat_cfg.mfcc_n_means].tolist())

    return np.asarray(feats, dtype=np.float32)


def pre_config_hash(vad_cfg: VadCfg, feat_cfg: AudioFeaturesCfg) -> str:
    from cache import config_hash

    return config_hash(
        vad_cfg,
        feat_cfg.f0_method,
        feat_cfg.f0_min_hz,
        feat_cfg.f0_max_hz,
        feat_cfg.mfcc_n,
        feat_cfg.mel_filters,
        feat_cfg.cmvn,
    )


def save_pre(pre: AudioPre, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy.savez always appends .npz to the basename - pass without suffix.
    np.savez(
        path.with_suffix(""),
        sp_mask=pre.sp_mask,
        f0=pre.f0,
        voiced_flag=pre.voiced_flag,
        voiced_in_speech=pre.voiced_in_speech,
        rms=pre.rms,
        mfcc=pre.mfcc,
        formants_per_frame=pre.formants_per_frame,
        centroid_per_frame=pre.centroid_per_frame,
    )


def load_pre(path: Path) -> AudioPre:
    with np.load(path) as z:
        return AudioPre(
            sp_mask=z["sp_mask"],
            f0=z["f0"],
            voiced_flag=z["voiced_flag"],
            voiced_in_speech=z["voiced_in_speech"],
            rms=z["rms"],
            mfcc=z["mfcc"],
            formants_per_frame=z["formants_per_frame"],
            centroid_per_frame=z["centroid_per_frame"],
        )


def feature_names(feat_cfg: AudioFeaturesCfg) -> list[str]:
    names = list(feat_cfg.include)
    names.extend([f"mfcc_mean_{i}" for i in range(feat_cfg.mfcc_n_means)])
    return names


def _compute_named(
    name: str,
    f0: np.ndarray,
    voiced_in_speech: np.ndarray,
    sp_mask: np.ndarray,
    rms: np.ndarray,
    formants_per_frame: np.ndarray | None = None,
    centroid_per_frame: np.ndarray | None = None,
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
    if name in ("f1_median", "f2_median", "f3_median"):
        if formants_per_frame is None:
            raise RuntimeError(f"{name}: formant pre-computation skipped")
        idx = int(name[1]) - 1
        return _formant_median_log(formants_per_frame, idx, voiced_in_speech)
    if name in ("f2_f1_ratio", "f3_f1_ratio"):
        if formants_per_frame is None:
            raise RuntimeError(f"{name}: formant pre-computation skipped")
        num_idx = int(name[1]) - 1
        return _formant_ratio(formants_per_frame, num_idx, 0, voiced_in_speech)
    if name == "spectral_centroid_log_mean":
        if centroid_per_frame is None:
            raise RuntimeError("spectral_centroid_log_mean: centroid pre-computation skipped")
        return _centroid_log_mean(centroid_per_frame, voiced_in_speech)
    raise ValueError(f"unknown feature: {name}")


def _formants_per_frame(
    x: np.ndarray,
    sr: int,
    hop: int,
    win: int,
    lpc_order: int = 12,
    n_formants: int = 4,
    f_min_hz: float = 90.0,
    f_max_hz: float | None = None,
    voiced_mask: np.ndarray | None = None,
) -> np.ndarray:
    if f_max_hz is None:
        f_max_hz = sr * 0.5 - 100.0
    x = x.astype(np.float32)
    n_samples = len(x)
    if n_samples < win:
        return np.full((1, n_formants), np.nan, dtype=np.float32)
    n_frames = 1 + (n_samples - win) // hop

    if voiced_mask is None:
        active = np.ones(n_frames, dtype=bool)
    else:
        m = min(n_frames, len(voiced_mask))
        active = np.zeros(n_frames, dtype=bool)
        active[:m] = voiced_mask[:m]
    out = np.full((n_frames, n_formants), np.nan, dtype=np.float32)
    if not active.any():
        return out

    window = np.hamming(win).astype(np.float32)
    active_idx = np.flatnonzero(active)
    starts = active_idx * hop
    frames = np.stack([x[s : s + win] for s in starts]) * window  # (k, win)
    # Drop all-zero frames from LPC to avoid degenerate fits.
    frame_norm = np.abs(frames).max(axis=1)
    keep = frame_norm > 0
    if not keep.any():
        return out
    frames = frames[keep]
    active_idx = active_idx[keep]

    # Per-frame LPC (still loop - librosa.lpc batching is currently slower than
    # the loop). The loop is fast (~80 ms for 1000 frames).
    a = np.zeros((frames.shape[0], lpc_order + 1), dtype=np.float32)
    for i in range(frames.shape[0]):
        try:
            a[i] = librosa.lpc(frames[i], order=lpc_order)
        except Exception:
            a[i] = np.nan
    finite = np.all(np.isfinite(a), axis=1)
    if not finite.any():
        return out
    a = a[finite]
    active_idx = active_idx[finite]

    # Batched companion-matrix root finding.
    deg = lpc_order
    n = a.shape[0]
    comp = np.zeros((n, deg, deg), dtype=np.complex128)
    if deg > 1:
        comp[:, np.arange(deg - 1) + 1, np.arange(deg - 1)] = 1.0
    comp[:, :, -1] = -a[:, :-1] / a[:, -1:]
    roots = np.linalg.eigvals(comp)  # (n, deg)

    # Keep roots inside unit circle with positive imaginary part.
    mask = (np.imag(roots) > 0.0) & (np.abs(roots) < 1.0)
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs_hz = (angles * sr / (2.0 * np.pi)).astype(np.float32)
    band = (freqs_hz >= f_min_hz) & (freqs_hz <= f_max_hz)
    keep_mask = mask & band

    # For each frame, pull the kept frequencies, sort, keep lowest n_formants.
    for k, frame_i in enumerate(active_idx):
        fk = np.sort(freqs_hz[k][keep_mask[k]])[:n_formants]
        out[frame_i, : len(fk)] = fk
    return out


def _formant_median_log(
    formants: np.ndarray, idx: int, voiced_in_speech: np.ndarray
) -> float:
    n = min(len(formants), len(voiced_in_speech))
    if n == 0:
        return 0.0
    fk = formants[:n, idx]
    mask = voiced_in_speech[:n] & np.isfinite(fk) & (fk > 0)
    if not mask.any():
        return 0.0
    return float(np.median(np.log(fk[mask])))


def _formant_ratio(
    formants: np.ndarray, num_idx: int, den_idx: int, voiced_in_speech: np.ndarray
) -> float:
    n = min(len(formants), len(voiced_in_speech))
    if n == 0:
        return 0.0
    fnum = formants[:n, num_idx]
    fden = formants[:n, den_idx]
    mask = (
        voiced_in_speech[:n]
        & np.isfinite(fnum)
        & np.isfinite(fden)
        & (fden > 0)
        & (fnum > 0)
    )
    if not mask.any():
        return 0.0
    # Median of log-ratio is robust and aligned with the log-Hz medians above.
    return float(np.median(np.log(fnum[mask] / fden[mask])))


def _spectral_centroid_per_frame(
    x: np.ndarray, sr: int, hop: int, win: int
) -> np.ndarray:
    return librosa.feature.spectral_centroid(
        y=x.astype(np.float32),
        sr=sr,
        n_fft=2048,
        hop_length=hop,
        win_length=win,
    )[0]


def _centroid_log_mean(centroid: np.ndarray, voiced_in_speech: np.ndarray) -> float:
    n = min(len(centroid), len(voiced_in_speech))
    if n == 0:
        return 0.0
    c = centroid[:n]
    mask = voiced_in_speech[:n] & (c > 0)
    if not mask.any():
        return 0.0
    return float(np.mean(np.log(c[mask])))


# NOTE: MFCC means aggregation lives inside `aggregate()` now; the heavy MFCC
# computation moved to `extract_pre()` so it gets cached.
