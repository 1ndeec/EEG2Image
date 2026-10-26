"""Labeled-support calibration inspired by SCORE.

Fit statistics and maps from known target-subject training-concept pairs.
Test EEG never contributes to fitting or the fixed CSLS reference bank.
This differs from original SCORE, which discovers unlabeled test correspondences."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ScoreCalibrationConfig:
    """Support-pool size, pair selection, Procrustes regularization, and CSLS settings."""
    enabled: bool = False
    pool_class_count: int = 160
    landmark_count: int = 80
    seed: int = 42
    regularization_ratio: float = 0.1
    csls_neighbors: int = 10
    epsilon: float = 1e-6

    def validate(self, class_count: int) -> None:
        if not 2 <= self.pool_class_count <= class_count:
            raise ValueError("pool_class_count must be between 2 and the training class count")
        if not 2 <= self.landmark_count < self.pool_class_count:
            raise ValueError("landmark_count must be at least 2 and below pool_class_count")
        if self.regularization_ratio < 0 or self.epsilon <= 0 or self.csls_neighbors < 1:
            raise ValueError("Invalid SCORE regularization, epsilon, or CSLS neighbor count")


def select_support_images(config: ScoreCalibrationConfig, class_count: int = 1654) -> tuple[int, ...]:
    """Choose one image per sampled class; reuse the identities across subjects."""
    config.validate(class_count)
    rng = np.random.default_rng(config.seed)
    classes = rng.choice(class_count, size=config.pool_class_count, replace=False)
    offsets = rng.integers(0, 10, size=config.pool_class_count)
    return tuple(sorted((classes * 10 + offsets).tolist()))


def _normalized(features: Tensor) -> Tensor:
    return F.normalize(features.float(), dim=-1)


def csls_scores(queries: Tensor, gallery: Tensor, reference_eeg: Tensor, neighbors: int) -> Tensor:
    """Correct cosine scores using the gallery and a fixed support-EEG reference bank."""
    queries, gallery, reference_eeg = map(_normalized, (queries, gallery, reference_eeg))
    cosine = queries @ gallery.T
    gallery_neighborhood = cosine.topk(min(neighbors, gallery.shape[0]), dim=1).values.mean(dim=1)
    eeg_neighborhood = (reference_eeg @ gallery.T).topk(
        min(neighbors, reference_eeg.shape[0]), dim=0
    ).values.mean(dim=0)
    return 2.0 * cosine - gallery_neighborhood[:, None] - eeg_neighborhood[None, :]


def correct_pair_margins(eeg: Tensor, images: Tensor, neighbors: int) -> Tensor:
    """Subtract the strongest incorrect image score from each known-pair score."""
    if eeg.shape[0] != images.shape[0] or eeg.shape[0] < 2:
        raise ValueError("CSLS landmark selection needs at least two matched support pairs")
    scores = csls_scores(eeg, images, eeg, neighbors)
    true_scores = scores.diagonal()
    competitors = scores.clone()
    competitors.fill_diagonal_(-torch.inf)
    return true_scores - competitors.max(dim=1).values


def _moments(eeg: Tensor, images: Tensor, epsilon: float) -> dict[str, Tensor]:
    return {
        "eeg_mean": eeg.mean(dim=0),
        "eeg_std": eeg.std(dim=0, correction=0),
        "image_mean": images.mean(dim=0),
        "image_std": images.std(dim=0, correction=0),
        "epsilon": torch.tensor(epsilon, dtype=eeg.dtype),
    }


def _match_moments(eeg: Tensor, statistics: dict[str, Tensor]) -> Tensor:
    eps = statistics["epsilon"]
    return ((eeg - statistics["eeg_mean"]) / (statistics["eeg_std"] + eps)
            * (statistics["image_std"] + eps) + statistics["image_mean"])


def _fit_orthogonal_map(eeg: Tensor, images: Tensor, ratio: float) -> dict[str, Tensor]:
    """Solve uniformly weighted, centered Procrustes with an identity penalty."""
    eeg_center, image_center = eeg.mean(dim=0), images.mean(dim=0)
    cross = (eeg - eeg_center).T @ (images - image_center)
    strength = ratio * torch.linalg.svdvals(cross)[0]
    matrix = cross + strength * torch.eye(cross.shape[0], dtype=cross.dtype)
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return {"rotation": u @ vh, "eeg_center": eeg_center, "image_center": image_center}


def _apply_map(eeg: Tensor, fitted: dict[str, Tensor]) -> Tensor:
    return (eeg - fitted["eeg_center"]) @ fitted["rotation"] + fitted["image_center"]


def _retrieval_metrics(scores: Tensor, targets: Tensor) -> dict[str, float]:
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == targets[:, None]).nonzero(as_tuple=False)[:, 1]
    return {"r1": float((ranks < 1).float().mean()), "r5": float((ranks < 5).float().mean())}


@torch.no_grad()
def evaluate_score_calibration(
    support_eeg: Tensor,
    support_images: Tensor,
    test_eeg: Tensor,
    gallery: Tensor,
    targets: Tensor,
    support_image_ids: tuple[int, ...],
    config: ScoreCalibrationConfig,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate uncalibrated, random-pair, and CSLS-selected maps with both rankers.

    Support rows follow support_image_ids; targets index gallery rows.
    Moment matching and SVD run on CPU in float64."""
    config.validate(1654)
    if len(support_image_ids) != config.pool_class_count:
        raise ValueError("Support image IDs do not match the configured pool size")
    support_eeg, support_images, test_eeg, gallery = (
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in (support_eeg, support_images, test_eeg, gallery)
    )
    targets = targets.detach().to(device="cpu", dtype=torch.long)
    if support_eeg.shape != support_images.shape or test_eeg.shape[1] != gallery.shape[1]:
        raise ValueError("EEG and image embedding dimensions must match")
    if targets.shape[0] != test_eeg.shape[0]:
        raise ValueError("Target count does not match test EEG count")
    if support_eeg.shape[0] != len(support_image_ids):
        raise ValueError("Support feature count does not match support image count")

    metrics: dict[str, float] = {}
    cosine = _normalized(test_eeg) @ _normalized(gallery).T
    baseline_scores = {
        "cosine": cosine,
        "csls": csls_scores(test_eeg, gallery, support_eeg, config.csls_neighbors),
    }
    for ranking, scores in baseline_scores.items():
        for name, value in _retrieval_metrics(scores, targets).items():
            metrics[f"uncalibrated_{ranking}_{name}"] = value

    statistics = _moments(support_eeg, support_images, config.epsilon)
    matched_support = _match_moments(support_eeg, statistics)
    matched_test = _match_moments(test_eeg, statistics)
    rng = np.random.default_rng(config.seed + 1)
    random_rows = tuple(sorted(rng.choice(
        len(support_image_ids), size=config.landmark_count, replace=False
    ).tolist()))
    margins = correct_pair_margins(matched_support, support_images, config.csls_neighbors)
    selected_rows = tuple(sorted(torch.argsort(
        margins, descending=True, stable=True
    )[:config.landmark_count].tolist()))
    artifacts: dict[str, object] = {
        "support_image_ids": list(support_image_ids),
        "statistics": statistics,
        "csls_correct_pair_margins": margins.float(),
        "methods": {},
    }
    for method, rows in (("random", random_rows), ("csls_selected", selected_rows)):
        fitted = _fit_orthogonal_map(
            matched_support[list(rows)], support_images[list(rows)],
            config.regularization_ratio,
        )
        corrected_test = _apply_map(matched_test, fitted)
        corrected_support = _apply_map(matched_support, fitted)
        scores_by_ranker = {
            "cosine": _normalized(corrected_test) @ _normalized(gallery).T,
            "csls": csls_scores(
                corrected_test, gallery, corrected_support, config.csls_neighbors
            ),
        }
        for ranking, scores in scores_by_ranker.items():
            for name, value in _retrieval_metrics(scores, targets).items():
                metrics[f"{method}_{ranking}_{name}"] = value
        artifacts["methods"][method] = {
            "landmark_image_ids": [support_image_ids[row] for row in rows],
            "map": fitted,
            "fixed_csls_eeg_bank": corrected_support.float(),
        }
    return metrics, artifacts


@torch.no_grad()
def evaluate_chained_score_calibration(
    support_eeg: Tensor,
    support_images: Tensor,
    test_eeg: Tensor,
    gallery: Tensor,
    targets: Tensor,
    support_image_ids: tuple[int, ...],
    config: ScoreCalibrationConfig,
    stages: tuple[tuple[str, int] | tuple[str, int, float], ...],
) -> tuple[dict[str, float], dict[str, object]]:
    """Fit successive maps to transformed support EEG and apply them to queries.

    Each stage may reuse support pairs and choose its own regularization.
    Test queries are transformed but never used to estimate a stage."""
    if not stages:
        return {}, {"stages": []}
    config.validate(1654)
    if len(support_image_ids) != config.pool_class_count:
        raise ValueError("Support image IDs do not match the configured pool size")
    parsed_stages: list[tuple[str, int, float]] = []
    for stage in stages:
        if len(stage) == 2:
            method, count = stage
            rho = config.regularization_ratio
        elif len(stage) == 3:
            method, count, rho = stage
        else:
            raise ValueError("Each chain stage must contain a method, landmark count, and optional rho")
        if (method not in {"random", "csls_selected"}
                or not isinstance(count, int) or isinstance(count, bool) or count < 2
                or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0):
            raise ValueError("Invalid chain method, landmark count, or rho")
        parsed_stages.append((method, count, float(rho)))
    if any(count > len(support_image_ids) for _, count, _ in parsed_stages):
        raise ValueError("A chain stage cannot use more landmarks than the support pool")
    if support_eeg.shape != support_images.shape or test_eeg.shape[1] != gallery.shape[1]:
        raise ValueError("EEG and image embedding dimensions must match")
    if support_eeg.shape[0] != len(support_image_ids) or targets.shape[0] != test_eeg.shape[0]:
        raise ValueError("Support or target count does not match the features")

    support_eeg, support_images, test_eeg, gallery = (
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in (support_eeg, support_images, test_eeg, gallery)
    )
    targets = targets.detach().to(device="cpu", dtype=torch.long)
    statistics = _moments(support_eeg, support_images, config.epsilon)
    current_support = _match_moments(support_eeg, statistics)
    current_test = _match_moments(test_eeg, statistics)
    available = list(range(len(support_image_ids)))
    rng = np.random.default_rng(config.seed + 1)
    metrics: dict[str, float] = {}
    artifacts: dict[str, object] = {"statistics": statistics, "stages": []}

    for stage_number, (method, count, rho) in enumerate(parsed_stages, start=1):
        if method == "random":
            rows = sorted(int(row) for row in rng.choice(available, size=count, replace=False))
        else:
            candidate_scores = correct_pair_margins(
                current_support[available], support_images[available], config.csls_neighbors
            )
            chosen = torch.argsort(candidate_scores, descending=True, stable=True)[:count]
            rows = sorted(available[int(index)] for index in chosen)
        fitted = _fit_orthogonal_map(
            current_support[rows], support_images[rows], rho
        )
        current_support = _apply_map(current_support, fitted)
        current_test = _apply_map(current_test, fitted)
        scores = {
            "cosine": _normalized(current_test) @ _normalized(gallery).T,
            "csls": csls_scores(current_test, gallery, current_support, config.csls_neighbors),
        }
        for ranking, values in scores.items():
            for metric, value in _retrieval_metrics(values, targets).items():
                metrics[f"chain_{stage_number}_{ranking}_{metric}"] = value
        artifacts["stages"].append({
            "stage": stage_number,
            "selection": method,
            "rho": rho,
            "landmark_image_ids": [support_image_ids[row] for row in rows],
            "map": fitted,
            "fixed_csls_eeg_bank": current_support.float(),
        })
    return metrics, artifacts
