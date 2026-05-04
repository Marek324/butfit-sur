
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import torch

DATASET = Path("dataset")
OUT = Path("scripts/vad_compare/silero_segments.json")
SR = 16000


def wav_paths() -> list[Path]:
    return sorted(DATASET.rglob("*.wav"))


def read_wav_mono16k(p: Path) -> torch.Tensor:
    with wave.open(str(p)) as f:
        if f.getframerate() != SR or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise RuntimeError(f"unexpected format for {p}: {f.getparams()}")
        n = f.getnframes()
        raw = f.readframes(n)
    arr = torch.frombuffer(raw, dtype=torch.int16).to(torch.float32) / 32768.0
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech-ms", type=int, default=100)
    ap.add_argument("--min-silence-ms", type=int, default=100)
    args = ap.parse_args()

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        print(
            "silero-vad not installed. Try: uv add --group dev silero-vad",
            file=sys.stderr,
        )
        return 2

    print("Loading Silero VAD model ...")
    model = load_silero_vad()

    files = wav_paths()
    print(f"Found {len(files)} WAV files")

    out: dict = {
        "sample_rate": SR,
        "model": "silero_vad",
        "params": {
            "threshold": args.threshold,
            "min_speech_ms": args.min_speech_ms,
            "min_silence_ms": args.min_silence_ms,
        },
        "files": {},
    }

    t0 = time.time()
    for i, p in enumerate(files, 1):
        audio = read_wav_mono16k(p)
        ts = get_speech_timestamps(
            audio,
            model,
            sampling_rate=SR,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            return_seconds=True,
        )
        dur_s = audio.numel() / SR
        speech = sum(seg["end"] - seg["start"] for seg in ts)
        rel = str(p.relative_to(DATASET))
        out["files"][rel] = {
            "duration_s": dur_s,
            "speech_segments_s": [[seg["start"], seg["end"]] for seg in ts],
            "speech_ratio": speech / dur_s if dur_s > 0 else 0.0,
        }
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  ({time.time() - t0:.1f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
