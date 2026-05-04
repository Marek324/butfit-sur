
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

DATASET = Path("dataset")
OUT = Path("scripts/vad_compare/energy_zcr_segments.json")
SR = 16000

FRAME_MS = 25
HOP_MS = 10
PREEMPH = 0.97
NOISE_PCT = 10.0
ENERGY_MARGIN_DB = 8.0
ZCR_AID_MARGIN_DB = 4.0
ZCR_LOW = 0.02
ZCR_HIGH = 0.35
MIN_SPEECH_MS = 100
MIN_SILENCE_MS = 100


def read_wav_mono16k(p: Path) -> np.ndarray:
    with wave.open(str(p)) as f:
        if f.getframerate() != SR or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise RuntimeError(f"unexpected format for {p}: {f.getparams()}")
        n = f.getnframes()
        raw = f.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return x[idx]


def compute_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pre = np.empty_like(x)
    pre[0] = x[0]
    pre[1:] = x[1:] - PREEMPH * x[:-1]

    frame_len = int(SR * FRAME_MS / 1000)
    hop_len = int(SR * HOP_MS / 1000)
    frames = frame_signal(pre, frame_len, hop_len)

    eps = 1e-10
    energy = np.sum(frames * frames, axis=1)
    energy_db = 10.0 * np.log10(energy + eps)

    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    zcr = crossings / (frame_len - 1)

    return energy_db, zcr


def smooth_runs(
    mask: np.ndarray, min_speech_frames: int, min_silence_frames: int
) -> np.ndarray:
    out = mask.copy()

    # Drop tiny speech runs.
    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_speech_frames:
                out[i:j] = False
            i = j
        else:
            i += 1

    # Fill tiny silence runs.
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) < min_silence_frames:
                out[i:j] = True
            i = j
        else:
            i += 1

    return out


def mask_to_segments(
    mask: np.ndarray, hop_s: float, frame_s: float
) -> list[list[float]]:
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            start = i * hop_s
            end = (j - 1) * hop_s + frame_s
            segs.append([float(start), float(end)])
            i = j
        else:
            i += 1
    return segs


def vad_one(x: np.ndarray) -> tuple[list[list[float]], float, float]:
    energy_db, zcr = compute_features(x)
    if len(energy_db) == 0:
        return [], 0.0, 0.0

    noise_db = float(np.percentile(energy_db, NOISE_PCT))
    e_thresh = noise_db + ENERGY_MARGIN_DB
    e_thresh_loose = noise_db + ZCR_AID_MARGIN_DB

    strong = energy_db > e_thresh
    fricative_like = (energy_db > e_thresh_loose) & (zcr >= ZCR_LOW) & (zcr <= ZCR_HIGH)
    raw = strong | fricative_like

    min_speech_frames = max(1, MIN_SPEECH_MS // HOP_MS)
    min_silence_frames = max(1, MIN_SILENCE_MS // HOP_MS)
    smoothed = smooth_runs(raw, min_speech_frames, min_silence_frames)

    hop_s = HOP_MS / 1000.0
    frame_s = FRAME_MS / 1000.0
    segs = mask_to_segments(smoothed, hop_s, frame_s)

    dur_s = len(x) / SR
    speech_s = sum(e - s for s, e in segs)
    return segs, dur_s, speech_s


def main() -> int:
    argparse.ArgumentParser().parse_args()

    files = sorted(DATASET.rglob("*.wav"))
    print(f"Found {len(files)} WAV files")

    out = {
        "sample_rate": SR,
        "model": "energy_zcr",
        "params": {
            "frame_ms": FRAME_MS,
            "hop_ms": HOP_MS,
            "preemphasis": PREEMPH,
            "noise_percentile": NOISE_PCT,
            "energy_margin_db": ENERGY_MARGIN_DB,
            "zcr_aid_margin_db": ZCR_AID_MARGIN_DB,
            "zcr_low": ZCR_LOW,
            "zcr_high": ZCR_HIGH,
            "min_speech_ms": MIN_SPEECH_MS,
            "min_silence_ms": MIN_SILENCE_MS,
        },
        "files": {},
    }

    t0 = time.time()
    for i, p in enumerate(files, 1):
        x = read_wav_mono16k(p)
        segs, dur_s, speech_s = vad_one(x)
        rel = str(p.relative_to(DATASET))
        out["files"][rel] = {
            "duration_s": dur_s,
            "speech_segments_s": segs,
            "speech_ratio": speech_s / dur_s if dur_s > 0 else 0.0,
        }
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  ({time.time() - t0:.1f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
