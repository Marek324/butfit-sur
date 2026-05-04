import time
from pathlib import Path

import numpy as np

from cache import cache_key_path, config_hash, get_or_compute
from config import Config
from data import Sample, load_png, load_wav_mono16k


def feature_config_hash(cfg: Config, modality: str) -> str:
    if modality == "audio":
        return config_hash(cfg.audio.vad, cfg.audio.features)
    if modality == "image":
        return config_hash(cfg.image.features)
    raise ValueError(f"feature cache not defined for modality={modality!r}")


def extract_one_audio(s: Sample, cfg: Config) -> np.ndarray:
    from audio_features import extract 

    x, sr = load_wav_mono16k(s.path)
    return extract(x, sr, cfg.audio.vad, cfg.audio.features)


def extract_one_image(s: Sample, cfg: Config) -> np.ndarray:
    from image_features import extract 

    img = load_png(s.path, grayscale=cfg.image.features.grayscale)
    return extract(img, cfg.image.features)


def _per_sample_extract_cached(
    sample: Sample, cfg: Config, modality: str, cfg_hash: str
) -> np.ndarray:
    cache_path = cache_key_path(cfg.cache.root, modality, cfg_hash, sample.name)

    if modality == "audio":
        return _audio_two_stage_cached(sample, cfg, cache_path)

    def go() -> np.ndarray:
        if modality == "image":
            return extract_one_image(sample, cfg)
        raise ValueError(f"unknown modality: {modality}")

    return get_or_compute(
        cache_path, go, enabled=cfg.cache.enabled, source_path=sample.path
    )


def _audio_two_stage_cached(
    sample: Sample, cfg: Config, vector_cache_path: Path
) -> np.ndarray:
    from audio_features import (
        aggregate,
        extract_pre,
        load_pre,
        pre_config_hash,
        save_pre,
    )

    enabled = cfg.cache.enabled
    src_mtime = sample.path.stat().st_mtime

    if (
        enabled
        and vector_cache_path.is_file()
        and vector_cache_path.stat().st_mtime >= src_mtime
    ):
        return np.load(vector_cache_path)

    pre_hash = pre_config_hash(cfg.audio.vad, cfg.audio.features)
    pre_path = cache_key_path(
        cfg.cache.root, "audio_pre", pre_hash, sample.name
    ).with_suffix(".npz")
    pre = None
    if enabled and pre_path.is_file() and pre_path.stat().st_mtime >= src_mtime:
        try:
            pre = load_pre(pre_path)
        except Exception:
            pre = None
    if pre is None:
        x, sr = load_wav_mono16k(sample.path)
        pre = extract_pre(x, sr, cfg.audio.vad, cfg.audio.features)
        if enabled:
            save_pre(pre, pre_path)

    v = aggregate(pre, cfg.audio.features)
    if enabled:
        vector_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vector_cache_path.with_suffix(""), v)
    return v


def extract_set(
    samples: list[Sample],
    cfg: Config,
    modality: str,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str | None]]:
    if modality not in ("audio", "image"):
        raise ValueError(f"unsupported modality: {modality}")

    cfg_hash = feature_config_hash(cfg, modality)

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    names: list[str] = []
    spkrs: list[str | None] = []

    t0 = time.time()
    for i, s in enumerate(samples):
        v = _per_sample_extract_cached(s, cfg, modality, cfg_hash)
        Xs.append(v)
        ys.append(s.label if s.label is not None else -1)
        names.append(s.name)
        spkrs.append(s.speaker_id)
        if progress and ((i + 1) % 25 == 0 or i == len(samples) - 1):
            print(
                f"  base {modality}: {i + 1}/{len(samples)}  ({time.time() - t0:.1f}s)"
            )

    return np.stack(Xs), np.asarray(ys), names, spkrs
