"""Multi-positive alignment and supervised source-subject recovery episodes.

The support maps are fitted without gradients. Recovery gradients reach
query encoders through the fixed maps, using images outside the support pool."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .score_calibration import _fit_orthogonal_map, _match_moments, _moments, correct_pair_margins


def multi_positive_loss(
    eeg: Tensor, visual: Tensor, image_indices: Tensor, temperature: float
) -> Tensor:
    """Sum bidirectional cross-entropy with equal mass on all same-image pairs."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if eeg.ndim != 2 or eeg.shape != visual.shape:
        raise ValueError("EEG and visual embeddings must have the same [batch, dim] shape")
    if image_indices.ndim != 1 or image_indices.shape[0] != eeg.shape[0]:
        raise ValueError("image_indices must have one entry per EEG/image pair")
    with torch.autocast(device_type=eeg.device.type, enabled=False):
        queries = F.normalize(eeg.float(), dim=-1)
        targets = F.normalize(visual.float(), dim=-1)
        logits = queries @ targets.T / temperature
        positives = image_indices[:, None].eq(image_indices[None, :]).float()
        positives = positives / positives.sum(dim=1, keepdim=True)
        eeg_to_image = -(positives * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        image_to_eeg = -(positives.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
        return eeg_to_image + image_to_eeg


@torch.no_grad()
def select_subject_csls_landmarks(
    support_eeg: Tensor,
    support_visual: Tensor,
    landmark_count: int,
    neighbors: int,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return support-row indices independently for each training subject.

    EEG has shape [subjects, support_images, dim]; visual features have shape
    [support_images, dim]. Selection uses known-pair CSLS margins."""
    if support_eeg.ndim != 3 or support_visual.ndim != 2:
        raise ValueError("Expected support EEG [subjects, images, dim] and images [images, dim]")
    if support_eeg.shape[1:] != support_visual.shape:
        raise ValueError("Support EEG and image dimensions must match")
    if not 2 <= landmark_count <= support_visual.shape[0]:
        raise ValueError("landmark_count must be between 2 and support image count")
    if neighbors < 1 or epsilon <= 0:
        raise ValueError("neighbors and epsilon must be positive")
    visual = support_visual.detach().float()
    image_mean = visual.mean(dim=0)
    image_std = visual.std(dim=0, correction=0)
    selected_rows = []
    for eeg in support_eeg:
        eeg = eeg.detach().float()
        matched = (
            (eeg - eeg.mean(dim=0)) / (eeg.std(dim=0, correction=0) + epsilon)
            * (image_std + epsilon) + image_mean
        )
        margins = correct_pair_margins(matched, visual, neighbors)
        selected_rows.append(
            torch.argsort(margins, descending=True, stable=True)[:landmark_count]
            .sort().values
        )
    return torch.stack(selected_rows)


@torch.no_grad()
def fit_supervised_support_map(
    support_eeg: Tensor,
    support_visual: Tensor,
    landmark_rows: Tensor | list[int],
    identity_ratio: float,
    epsilon: float = 1e-6,
) -> dict[str, Tensor]:
    """Estimate moments from the full support pool and rotation from selected rows."""
    if support_eeg.ndim != 2 or support_eeg.shape != support_visual.shape:
        raise ValueError("Matched support EEG and images must have the same [images, dim] shape")
    if support_eeg.shape[0] < 2 or identity_ratio < 0 or epsilon <= 0:
        raise ValueError("Need at least two pairs, nonnegative identity ratio, positive epsilon")
    eeg = support_eeg.detach().to(device="cpu", dtype=torch.float64)
    visual = support_visual.detach().to(device="cpu", dtype=torch.float64)
    rows = torch.as_tensor(landmark_rows, dtype=torch.long, device="cpu")
    if (rows.ndim != 1 or rows.numel() < 2 or rows.unique().numel() != rows.numel()
            or int(rows.min()) < 0 or int(rows.max()) >= eeg.shape[0]):
        raise ValueError("Landmark rows must be at least two distinct support-pool indices")
    statistics = _moments(eeg, visual, epsilon)
    matched = _match_moments(eeg, statistics)
    fitted = _fit_orthogonal_map(matched[rows], visual[rows], identity_ratio)
    values = {
        **statistics,
        "center_eeg": fitted["eeg_center"],
        "center_image": fitted["image_center"],
        "rotation": fitted["rotation"],
    }
    return {name: value.float() for name, value in values.items()}


def support_to_query_recovery_loss(
    query_eeg: Tensor,
    query_visual: Tensor,
    fitted: dict[str, Tensor],
    temperature: float,
) -> Tensor:
    """Retrieve disjoint query images after applying a fixed, support-fitted map."""
    if query_eeg.ndim != 2 or query_eeg.shape != query_visual.shape:
        raise ValueError("Query EEG and image features must have the same [images, dim] shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with torch.autocast(device_type=query_eeg.device.type, enabled=False):
        query = query_eeg.float()
        gallery = query_visual.float()
        matched = (
            (query - fitted["eeg_mean"]) / (fitted["eeg_std"] + fitted["epsilon"])
            * (fitted["image_std"] + fitted["epsilon"]) + fitted["image_mean"]
        )
        recovered = (matched - fitted["center_eeg"]) @ fitted["rotation"] + fitted["center_image"]
        logits = F.normalize(recovered, dim=-1) @ F.normalize(gallery, dim=-1).T
        labels = torch.arange(query.shape[0], device=query.device)
        return F.cross_entropy(logits / temperature, labels)
