"""TOML experiment config + dataclass schema with strict unknown-key validation."""

from __future__ import annotations

import tomllib
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class DataCfg:
    target_id: str
    train_dirs: list[Path]
    dev_dirs: list[Path]


@dataclass(frozen=True)
class ValidationCfg:
    strategy: str = "holdout_dev"  # | "session_grouped_cv"
    cv_folds: int = 2


@dataclass(frozen=True)
class VadCfg:
    algorithm: str = "energy_zcr"
    frame_ms: int = 25
    hop_ms: int = 10
    preemphasis: float = 0.97
    noise_percentile: float = 10.0
    energy_margin_db: float = 8.0
    zcr_aid_margin_db: float = 4.0
    zcr_low: float = 0.02
    zcr_high: float = 0.35
    min_speech_ms: int = 100
    min_silence_ms: int = 100


@dataclass(frozen=True)
class AudioFeaturesCfg:
    f0_method: str = "pyin"  # librosa.pyin
    f0_min_hz: float = 60.0
    f0_max_hz: float = 400.0
    include: tuple[str, ...] = (
        "log_f0_mean",
        "log_f0_std",
        "jitter_local",
        "shimmer_local",
        "voicing_ratio",
    )
    mfcc_n_means: int = 5
    mfcc_n: int = 13
    mel_filters: int = 23
    cmvn: str = "per_utterance"


@dataclass(frozen=True)
class WaveformAugCfg:
    time_stretch_range: tuple[float, float] = (0.9, 1.1)
    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    polarity_flip_p: float = 0.5
    crop_min_s: float = 5.0
    eq_enabled: bool = True
    eq_gain_db_range: tuple[float, float] = (-3.0, 3.0)
    eq_freq_hz_range: tuple[float, float] = (200.0, 4000.0)
    eq_q: float = 1.0


@dataclass(frozen=True)
class FeatureSpaceAugCfg:
    enabled: bool = False
    sigma_per_dim_source: str = "non_target_within_speaker"
    sigma_multiplier: float = 1.0


@dataclass(frozen=True)
class AugTrainCfg:
    enabled: bool = True
    n_per_target: int = 15
    n_per_nontarget: int = 0
    seed: int = 42
    waveform: WaveformAugCfg = field(default_factory=WaveformAugCfg)
    feature_space: FeatureSpaceAugCfg = field(default_factory=FeatureSpaceAugCfg)


@dataclass(frozen=True)
class AugDevCfg:
    """Disjoint-envelope dev augmentation. Off by default; recommend disjoint
    ranges from train so dev still measures generalization, not invariance."""

    enabled: bool = False
    n_per_target: int = 5
    seed: int = 43
    waveform: WaveformAugCfg = field(
        default_factory=lambda: WaveformAugCfg(
            time_stretch_range=(0.80, 0.88),
            gain_db_range=(-12.0, -7.0),
            eq_freq_hz_range=(4000.0, 7000.0),
        )
    )


@dataclass(frozen=True)
class AudioCfg:
    vad: VadCfg = field(default_factory=VadCfg)
    features: AudioFeaturesCfg = field(default_factory=AudioFeaturesCfg)
    augment_train: AugTrainCfg = field(default_factory=AugTrainCfg)
    augment_dev: AugDevCfg = field(default_factory=AugDevCfg)


@dataclass(frozen=True)
class LogregCfg:
    C: float = 1.0
    class_weight: str = "balanced"  # "" → None (no weighting)
    penalty: str = "l2"
    max_iter: int = 1000


@dataclass(frozen=True)
class ClassifierCfg:
    type: str = "logreg"  # | "svm_rbf"
    logreg: LogregCfg = field(default_factory=LogregCfg)


@dataclass(frozen=True)
class OutputCfg:
    result_filename: str = "audio_logreg.txt"
    score: str = "logit"  # log-odds; monotonic, unbounded, sign-aligned to "1 = target"
    hard_threshold_logit: float = 0.0


@dataclass(frozen=True)
class Config:
    name: str
    modality: str  # "audio" | "image" | "fusion"
    data: DataCfg
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    output: OutputCfg = field(default_factory=OutputCfg)


def _convert(value: Any, target_type: Any) -> Any:
    if target_type is Path:
        return Path(value)
    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is list:
        return [_convert(v, args[0]) for v in value]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0]) for v in value)
        return tuple(_convert(v, t) for v, t in zip(value, args, strict=True))
    if isinstance(target_type, type) and is_dataclass(target_type):
        return _from_dict(target_type, value or {})
    return value


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"unknown keys in [{cls.__name__}]: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _convert(data[f.name], hints[f.name])
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ValueError(f"missing required field: {cls.__name__}.{f.name}")
    return cls(**kwargs)


def _flatten_audio_aug(audio: dict[str, Any]) -> dict[str, Any]:
    """Map [audio.augment.train] / [audio.augment.dev] in TOML to
    AudioCfg.augment_train / AudioCfg.augment_dev in the dataclass schema."""
    aug = audio.get("augment")
    if not isinstance(aug, dict):
        return audio
    out = {k: v for k, v in audio.items() if k != "augment"}
    if "train" in aug:
        out["augment_train"] = aug["train"]
    if "dev" in aug:
        out["augment_dev"] = aug["dev"]
    extra = set(aug) - {"train", "dev"}
    if extra:
        raise ValueError(f"unknown keys in [audio.augment]: {sorted(extra)}")
    return out


def load_config(path: Path | str) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if "audio" in raw:
        raw["audio"] = _flatten_audio_aug(raw["audio"])
    return _from_dict(Config, raw)
