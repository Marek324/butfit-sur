import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GmmModel:
    target: GaussianMixture
    ubm: GaussianMixture
    n_mfcc: int
    n_target_frames: int
    n_ubm_frames: int


def _frames_from_pre(mfcc_t_by_T: np.ndarray, sp_mask: np.ndarray) -> np.ndarray:
    if mfcc_t_by_T.ndim != 2:
        raise ValueError(f"expected 2-d MFCC matrix, got shape {mfcc_t_by_T.shape}")
    if sp_mask.shape[0] != mfcc_t_by_T.shape[1]:
        raise ValueError(
            f"sp_mask length {sp_mask.shape[0]} != MFCC frames {mfcc_t_by_T.shape[1]}"
        )
    return mfcc_t_by_T[:, sp_mask].T.astype(np.float64, copy=False)


def collect_frames(
    pre_paths: list[Path],
    load_pre,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in pre_paths:
        pre = load_pre(p)
        out.append(_frames_from_pre(pre.mfcc, pre.sp_mask))
    return out


def fit_gmm(
    frames_concat: np.ndarray,
    n_components: int,
    covariance_type: str = "diag",
    reg_covar: float = 1e-3,
    random_state: int = 0,
    max_iter: int = 200,
    n_init: int = 1,
) -> GaussianMixture:
    if frames_concat.ndim != 2:
        raise ValueError(f"expected 2-d frame matrix, got {frames_concat.shape}")
    if frames_concat.shape[0] < n_components * 5:
        raise ValueError(
            f"too few frames ({frames_concat.shape[0]}) for {n_components} mixtures"
        )
    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    gm.fit(frames_concat)
    return gm


def map_adapt_means(
    ubm: GaussianMixture, target_frames: np.ndarray, relevance_factor: float = 16.0
) -> GaussianMixture:
    if ubm.covariance_type != "diag":
        raise NotImplementedError(
            f"MAP adaptation only implemented for diagonal cov, got {ubm.covariance_type!r}"
        )
    if target_frames.ndim != 2 or target_frames.shape[1] != ubm.means_.shape[1]:
        raise ValueError(
            f"target_frames shape {target_frames.shape} incompatible with UBM "
            f"means shape {ubm.means_.shape}"
        )
    resp = ubm.predict_proba(target_frames)
    n_k = resp.sum(axis=0)
    weighted = resp.T @ target_frames
    safe_n = np.where(n_k > 1e-12, n_k, 1.0)
    x_bar = weighted / safe_n[:, None]
    alpha = n_k / (n_k + relevance_factor)
    new_means = alpha[:, None] * x_bar + (1 - alpha)[:, None] * ubm.means_

    adapted = GaussianMixture(
        n_components=ubm.n_components,
        covariance_type=ubm.covariance_type,
        random_state=ubm.random_state,
    )
    adapted.weights_ = ubm.weights_.copy()
    adapted.means_ = new_means.astype(ubm.means_.dtype, copy=False)
    adapted.covariances_ = ubm.covariances_.copy()
    adapted.precisions_cholesky_ = 1.0 / np.sqrt(adapted.covariances_)
    adapted.converged_ = True
    adapted.n_iter_ = 0
    adapted.lower_bound_ = float("nan")
    return adapted


def score_utterance(
    frames: np.ndarray, model: GmmModel
) -> float:
    if frames.shape[0] == 0:
        return float("nan")
    log_p_t = model.target.score_samples(frames)
    log_p_u = model.ubm.score_samples(frames)
    return float(np.mean(log_p_t - log_p_u))


def save(model: GmmModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load(path: Path) -> GmmModel:
    with path.open("rb") as f:
        return pickle.load(f)
