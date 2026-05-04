import numpy as np
from skimage import color
from skimage.feature import hog
from skimage.transform import resize

from config import ImageFeaturesCfg


def extract(img: np.ndarray, cfg: ImageFeaturesCfg) -> np.ndarray:
    img = _prepare(img, cfg)

    if cfg.method == "hog":
        feats = hog(
            img,
            orientations=cfg.hog_orientations,
            pixels_per_cell=tuple(cfg.hog_pixels_per_cell),
            cells_per_block=tuple(cfg.hog_cells_per_block),
            block_norm=cfg.hog_block_norm,
            feature_vector=True,
            channel_axis=-1 if img.ndim == 3 else None,
        )
        return feats.astype(np.float32)

    if cfg.method == "raw_flatten":
        return img.astype(np.float32).reshape(-1)

    raise ValueError(f"unknown image feature method: {cfg.method}")


def feature_dim(cfg: ImageFeaturesCfg) -> int:
    h, w = cfg.image_size
    if cfg.method == "raw_flatten":
        c = 1 if cfg.grayscale else 3
        return h * w * c
    if cfg.method == "hog":
        ph, pw = cfg.hog_pixels_per_cell
        bh, bw = cfg.hog_cells_per_block
        n_cells_y = h // ph
        n_cells_x = w // pw
        n_blocks_y = max(0, n_cells_y - bh + 1)
        n_blocks_x = max(0, n_cells_x - bw + 1)
        return n_blocks_y * n_blocks_x * bh * bw * cfg.hog_orientations
    raise ValueError(f"unknown image feature method: {cfg.method}")


def _prepare(img: np.ndarray, cfg: ImageFeaturesCfg) -> np.ndarray:
    if cfg.grayscale and img.ndim == 3:
        img = color.rgb2gray(img)
    target = tuple(cfg.image_size)
    if (img.shape[0], img.shape[1]) != target:
        img = resize(img, target, anti_aliasing=True)
    return img.astype(np.float32)
