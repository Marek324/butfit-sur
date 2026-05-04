#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np

from fusion import load as load_fusion, score_fusion


def _read(path: Path) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, score, hard = line.split()
        out[name] = (float(score), int(hard))
    return out


def _write(path: Path, rows: dict[str, tuple[float, int]]) -> None:
    lines = [f"{n} {s:.6f} {h}" for n, (s, h) in rows.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-result", type=Path, default=Path("results/audio_logreg.txt"))
    ap.add_argument("--image-result", type=Path, default=Path("results/image_logreg.txt"))
    ap.add_argument(
        "--fusion-result", type=Path, default=Path("results/fusion_logreg.txt")
    )
    ap.add_argument(
        "--fusion-model",
        type=Path,
        default=Path("artifacts/fusion_baseline_logreg/model.pkl"),
    )
    ap.add_argument(
        "--collapse-score",
        type=float,
        default=1.535,
        help="Score value associated with VAD-empty inputs (autodetect not implemented)",
    )
    ap.add_argument(
        "--collapse-tol",
        type=float,
        default=0.005,
        help="Tolerance around --collapse-score for matching",
    )
    ap.add_argument(
        "--sentinel",
        type=float,
        default=-5.0,
        help="Replacement audio score for VAD-empty samples",
    )
    args = ap.parse_args()

    audio = _read(args.audio_result)
    image = _read(args.image_result)
    fusion_old = _read(args.fusion_result)

    if set(audio.keys()) != set(image.keys()) or set(audio.keys()) != set(fusion_old.keys()):
        raise SystemExit("audio/image/fusion result files have mismatched segment names")

    affected = [
        n for n, (s, _h) in audio.items()
        if abs(s - args.collapse_score) < args.collapse_tol
    ]
    print(f"Found {len(affected)} VAD-empty audio samples (score≈{args.collapse_score})")

    for orig in (args.audio_result, args.fusion_result):
        backup = orig.with_suffix(orig.suffix.replace(".txt", ".before_vad_fix.txt"))
        if not backup.exists():
            shutil.copy2(orig, backup)
            print(f"  backed up {orig} -> {backup}")

    audio_changed_to_negative = 0
    for name in affected:
        old_score, _old_hard = audio[name]
        audio[name] = (args.sentinel, 0)
        audio_changed_to_negative += 1
    print(f"Patched audio: {audio_changed_to_negative} entries -> score={args.sentinel}, hard=0")
    _write(args.audio_result, audio)

    fmodel = load_fusion(args.fusion_model)
    names_ordered = list(audio.keys())
    a_scores = np.array([audio[n][0] for n in names_ordered], dtype=np.float64)
    i_scores = np.array([image[n][0] for n in names_ordered], dtype=np.float64)
    fused = score_fusion(fmodel, a_scores, i_scores)
    fusion_new: dict[str, tuple[float, int]] = {}
    for n, s in zip(names_ordered, fused):
        fusion_new[n] = (float(s), int(s >= 0.0))
    _write(args.fusion_result, fusion_new)

    n_target_old = sum(1 for _, h in fusion_old.values() if h == 1)
    n_target_new = sum(1 for _, h in fusion_new.values() if h == 1)
    print()
    print("Fusion change summary:")
    print(f"  predicted target: {n_target_old} -> {n_target_new} ({n_target_new - n_target_old:+d})")

    flips = [
        n for n in names_ordered
        if fusion_old[n][1] != fusion_new[n][1]
    ]
    print(f"  hard-decision flips: {len(flips)}")
    flip_to_neg = sum(1 for n in flips if fusion_new[n][1] == 0)
    flip_to_pos = sum(1 for n in flips if fusion_new[n][1] == 1)
    print(f"    -> non-target: {flip_to_neg}")
    print(f"    -> target:     {flip_to_pos}")

    print()
    print("First 5 affected samples (audio + fusion):")
    for n in affected[:5]:
        old_a = "1.535"
        new_a = f"{args.sentinel}"
        old_f = fusion_old[n]
        new_f = fusion_new[n]
        print(
            f"  {n}: audio {old_a}->{new_a},  fusion {old_f[0]:+.3f}/{old_f[1]} -> {new_f[0]:+.3f}/{new_f[1]}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
