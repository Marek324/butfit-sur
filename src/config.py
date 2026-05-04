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
    # Directories holding the unlabeled evaluation set (May 3rd release).
    # Used by `evaluate.py --split eval`. Empty list disables that mode.
    eval_dirs: list[Path] = field(default_factory=list)


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
    # Additive white-Gaussian noise at a randomly drawn SNR. SNR is computed
    # against the post-gain signal RMS, so quieter variants stay equally noisy
    # in absolute terms. Set both endpoints to a high value (e.g. (60, 60)) to
    # disable.
    noise_snr_db_range: tuple[float, float] = (15.0, 30.0)


@dataclass(frozen=True)
class AudioCfg:
    vad: VadCfg = field(default_factory=VadCfg)
    features: AudioFeaturesCfg = field(default_factory=AudioFeaturesCfg)


@dataclass(frozen=True)
class ImageFeaturesCfg:
    method: str = "hog"  # | "raw_flatten"
    image_size: tuple[int, int] = (80, 80)
    grayscale: bool = True
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    hog_block_norm: str = "L2-Hys"
    pca_components: int = 0  # 0 ⇒ no PCA reduction


@dataclass(frozen=True)
class ImageAugCfg:
    horizontal_flip_p: float = 0.5
    rotation_deg_range: tuple[float, float] = (-10.0, 10.0)
    brightness_range: tuple[float, float] = (0.85, 1.15)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    translate_pct: float = 0.05
    crop_pct: float = 0.0
    # Additive Gaussian pixel noise (per-channel, IID). Sigma is in [0, 1]
    # image-value units; (0.0, 0.0) disables noise entirely.
    noise_sigma_range: tuple[float, float] = (0.0, 0.04)


@dataclass(frozen=True)
class ImageCfg:
    features: ImageFeaturesCfg = field(default_factory=ImageFeaturesCfg)


@dataclass(frozen=True)
class LogregCfg:
    C: float = 1.0
    class_weight: str = "balanced"  # "" → None (no weighting)
    penalty: str = "l2"
    max_iter: int = 1000


@dataclass(frozen=True)
class SvmCfg:
    C: float = 1.0
    class_weight: str = "balanced"
    kernel: str = "rbf"  # informational; classifier dispatches on type
    gamma: str = "scale"  # | a stringified float, e.g. "0.05"


@dataclass(frozen=True)
class ClassifierCfg:
    type: str = "logreg"  # | "svm_rbf" | "svm_linear"
    logreg: LogregCfg = field(default_factory=LogregCfg)
    svm: SvmCfg = field(default_factory=SvmCfg)


@dataclass(frozen=True)
class FusionCfg:
    audio_artifact: str = ""  # name of artifacts/<dir> for the audio system to fuse
    image_artifact: str = ""  # name of artifacts/<dir> for the image system to fuse
    score_normalize: str = "none"  # | "zscore" - applied before fusion logreg
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    train_on: str = (
        "dev"  # which split to use as fusion training data ("dev" | "cv_oof")
    )


@dataclass(frozen=True)
class CacheCfg:
    enabled: bool = True
    root: Path = Path("artifacts/cache")


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
    image: ImageCfg = field(default_factory=ImageCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    cache: CacheCfg = field(default_factory=CacheCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    seed: int = 42


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


def load_config(path: Path | str) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return _from_dict(Config, raw)


# --- Offline augmentation config (consumed by scripts/augment_dataset.py) ---


@dataclass(frozen=True)
class AudioAugmentConfig:
    modality: str  # must be "audio"
    source_target_dir: Path
    source_nontarget_dir: Path
    output_target_dir: Path
    output_nontarget_dir: Path
    n_per_target: int = 15
    n_per_nontarget: int = 0
    seed: int = 42
    waveform: WaveformAugCfg = field(default_factory=WaveformAugCfg)


@dataclass(frozen=True)
class ImageAugmentConfig:
    modality: str  # must be "image"
    source_target_dir: Path
    source_nontarget_dir: Path
    output_target_dir: Path
    output_nontarget_dir: Path
    n_per_target: int = 15
    n_per_nontarget: int = 0
    seed: int = 42
    envelope: ImageAugCfg = field(default_factory=ImageAugCfg)


def load_augment_config(
    path: Path | str,
) -> AudioAugmentConfig | ImageAugmentConfig:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    modality = raw.get("modality")
    if modality == "audio":
        return _from_dict(AudioAugmentConfig, raw)
    if modality == "image":
        return _from_dict(ImageAugmentConfig, raw)
    raise ValueError(
        f"augment config {path}: modality must be 'audio' or 'image', got {modality!r}"
    )
