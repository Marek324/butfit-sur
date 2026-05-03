"""Sample enumeration, WAV loading, PNG loading.

Common conventions:
- Filename stem = `<segment_name>` in the result file.
- Speaker ID is field 1 of the underscore-separated stem.
- Session ID = "<speaker>_<session>"; samples in the same session must
  not straddle a CV fold boundary (instructor recommendation).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_SR = 16000
EXPECTED_PNG_SIZE = (80, 80)


@dataclass(frozen=True)
class Sample:
    path: Path
    name: str  # filename stem (= "<segment_name>" in the result file)
    label: int | None  # 1=target, 0=non-target, None=eval
    speaker_id: str | None  # field 1 of filename
    session_id: str | None  # "<speaker>_<session>"; None if filename can't be parsed


@dataclass(frozen=True)
class PairedSample:
    """Audio + image sample that share the same filename stem (used for fusion)."""

    audio_path: Path
    image_path: Path
    name: str
    label: int | None
    speaker_id: str | None
    session_id: str | None


def _parse_filename(stem: str) -> tuple[str | None, str | None]:
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None
    return parts[0], f"{parts[0]}_{parts[1]}"


def enumerate_split(
    dirs: list[Path],
    extension: str = ".wav",
    label_required: bool = True,
) -> list[Sample]:
    """Enumerate samples across the given directories.

    Label is inferred from each directory name:
      - starts with 'non_target_'  → label 0
      - starts with 'target_'      → label 1
      - otherwise                  → label None (eval mode), unless label_required
    """
    out: list[Sample] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"directory not found: {d}")
        name = d.name
        if name.startswith("non_target_"):
            label: int | None = 0
        elif name.startswith("target_"):
            label = 1
        elif label_required:
            raise ValueError(
                f"cannot infer label from {d!s}; expected target_* or non_target_* "
                f"(or pass label_required=False for eval data)"
            )
        else:
            label = None
        for p in sorted(d.glob(f"*{extension}")):
            spk, sess = _parse_filename(p.stem)
            out.append(
                Sample(
                    path=p, name=p.stem, label=label, speaker_id=spk, session_id=sess
                )
            )
    return out


def enumerate_paired(
    dirs: list[Path],
    label_required: bool = True,
) -> list[PairedSample]:
    """Enumerate (WAV, PNG) pairs that share a filename stem under the given dirs.

    Both files must exist next to each other; orphaned files are skipped and
    reported via assertion-style error if label_required=True. Useful for
    fusion training/scoring where the two modalities operate on the same
    underlying sample.
    """
    out: list[PairedSample] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"directory not found: {d}")
        name = d.name
        if name.startswith("non_target_"):
            label: int | None = 0
        elif name.startswith("target_"):
            label = 1
        elif label_required:
            raise ValueError(
                f"cannot infer label from {d!s}; expected target_* or non_target_* "
                f"(or pass label_required=False for eval data)"
            )
        else:
            label = None

        wavs = {p.stem: p for p in d.glob("*.wav")}
        pngs = {p.stem: p for p in d.glob("*.png")}
        common = sorted(wavs.keys() & pngs.keys())
        for stem in common:
            spk, sess = _parse_filename(stem)
            out.append(
                PairedSample(
                    audio_path=wavs[stem],
                    image_path=pngs[stem],
                    name=stem,
                    label=label,
                    speaker_id=spk,
                    session_id=sess,
                )
            )
    return out


def load_wav_mono16k(p: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as float32 in [-1, 1]. Asserts 16 kHz mono PCM16."""
    with wave.open(str(p)) as f:
        sr = f.getframerate()
        ch = f.getnchannels()
        sw = f.getsampwidth()
        if sr != EXPECTED_SR or ch != 1 or sw != 2:
            raise RuntimeError(
                f"unexpected WAV format for {p}: sr={sr} ch={ch} sampwidth={sw} "
                f"(expected {EXPECTED_SR}/mono/16-bit)"
            )
        n = f.getnframes()
        raw = f.readframes(n)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, sr


def load_png(p: Path, grayscale: bool = False) -> np.ndarray:
    """Read a PNG as float32 in [0, 1].

    Returns (H, W, 3) for RGB or (H, W) for grayscale. Strips alpha if present.
    """
    with Image.open(p) as im:
        if grayscale:
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        else:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr
