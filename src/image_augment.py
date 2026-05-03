"""Image augmentations chosen to be approximately identity-preserving.

Geometric + photometric + sensor-style transforms — none of which alter the
perceived person:
  - horizontal flip (faces are roughly bilaterally symmetric)
  - small rotation (head tilt, head pose drift)
  - small translate (face crop drift)
  - brightness/contrast jitter (lighting variation)
  - additive Gaussian pixel noise (sensor noise, JPEG-style artifacts)
  - optional small crop

Deliberately NOT implemented: vertical flip, large rotations, color hue
rotation. Augmentations operate in image space — feature extraction happens
after.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import AffineTransform, warp

from config import ImageAugCfg


def augment(img: np.ndarray, cfg: ImageAugCfg, rng: np.random.Generator) -> np.ndarray:
    """Apply one random sample from the image-augmentation envelope.

    img: (H, W) or (H, W, 3) float in [0, 1]. Returns same dtype / shape.
    """
    out = img.astype(np.float32, copy=True)

    if rng.random() < cfg.horizontal_flip_p:
        out = np.ascontiguousarray(out[:, ::-1])

    rot_lo, rot_hi = cfg.rotation_deg_range
    tr_pct = cfg.translate_pct
    needs_warp = (rot_hi > rot_lo) or tr_pct > 0
    if needs_warp:
        deg = float(rng.uniform(rot_lo, rot_hi)) if rot_hi > rot_lo else 0.0
        h, w = out.shape[:2]
        tx = float(rng.uniform(-tr_pct, tr_pct)) * w if tr_pct > 0 else 0.0
        ty = float(rng.uniform(-tr_pct, tr_pct)) * h if tr_pct > 0 else 0.0
        out = _affine(out, angle_deg=deg, tx=tx, ty=ty)

    out = _photometric(out, cfg, rng)

    if cfg.crop_pct > 0:
        out = _random_crop(out, cfg.crop_pct, rng)

    sigma_lo, sigma_hi = cfg.noise_sigma_range
    if sigma_hi > 0:
        sigma = float(rng.uniform(sigma_lo, sigma_hi))
        if sigma > 0:
            out = np.clip(
                out + (sigma * rng.standard_normal(out.shape)).astype(np.float32),
                0.0,
                1.0,
            )

    return out.astype(np.float32, copy=False)


def _affine(img: np.ndarray, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    angle = np.deg2rad(angle_deg)
    # Rotate about center, then translate.
    to_center = AffineTransform(translation=(-cx, -cy))
    rot = AffineTransform(rotation=angle)
    from_center = AffineTransform(translation=(cx + tx, cy + ty))
    tform = to_center + rot + from_center
    return warp(img, tform.inverse, mode="edge", preserve_range=True).astype(np.float32)


def _photometric(
    img: np.ndarray, cfg: ImageAugCfg, rng: np.random.Generator
) -> np.ndarray:
    b_lo, b_hi = cfg.brightness_range
    if b_hi > b_lo or b_lo != 1.0:
        b = float(rng.uniform(b_lo, b_hi))
        img = np.clip(img * b, 0.0, 1.0)
    c_lo, c_hi = cfg.contrast_range
    if c_hi > c_lo or c_lo != 1.0:
        c = float(rng.uniform(c_lo, c_hi))
        if img.ndim == 3:
            mean = img.mean(axis=(0, 1), keepdims=True)
        else:
            mean = float(img.mean())
        img = np.clip((img - mean) * c + mean, 0.0, 1.0)
    return img


def _random_crop(
    img: np.ndarray, crop_pct: float, rng: np.random.Generator
) -> np.ndarray:
    h, w = img.shape[:2]
    ch = int(round(h * crop_pct))
    cw = int(round(w * crop_pct))
    if ch <= 0 and cw <= 0:
        return img
    top = int(rng.integers(0, ch + 1)) if ch > 0 else 0
    left = int(rng.integers(0, cw + 1)) if cw > 0 else 0
    bot = h - (ch - top)
    right = w - (cw - left)
    return img[top:bot, left:right]
