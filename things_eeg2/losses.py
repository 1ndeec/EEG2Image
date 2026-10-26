"""SAMGA retrieval and coarse-stage distribution alignment."""

from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import OptimizationConfig


def symmetric_contrastive_loss(
    eeg: Tensor,
    visual: Tensor,
    temperature: float,
) -> Tensor:
    """Average diagonal-pair cross-entropy in both retrieval directions.

    All off-diagonal scores remain in the denominator, including repeated
    images from other subjects. Inputs are [batch, embedding_dim]."""
    eeg = F.normalize(eeg, dim=-1)
    visual = F.normalize(visual, dim=-1)
    logits = eeg @ visual.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _pairwise_squared_distance(left: Tensor, right: Tensor) -> Tensor:
    distances = (
        left.square().sum(dim=1, keepdim=True)
        + right.square().sum(dim=1).unsqueeze(0)
        - 2.0 * left @ right.T
    )
    return distances.clamp_min(0.0)


def _multi_rbf(left: Tensor, right: Tensor, scales: Sequence[float]) -> Tensor:
    distance = _pairwise_squared_distance(left, right)
    kernels = [torch.exp(-distance / (2.0 * scale * scale)) for scale in scales]
    return torch.stack(kernels).mean(dim=0)


def mmd_loss(eeg: Tensor, visual: Tensor, scales: Sequence[float]) -> Tensor:
    """Compare EEG and visual distributions with a mixture of RBF kernels."""
    batch = eeg.shape[0]
    if batch < 2:
        return eeg.new_zeros(())
    eeg_kernel = _multi_rbf(eeg, eeg, scales)
    visual_kernel = _multi_rbf(visual, visual, scales)
    cross_kernel = _multi_rbf(eeg, visual, scales)
    within_denominator = batch * (batch - 1)
    eeg_within = (eeg_kernel.sum() - eeg_kernel.diagonal().sum()) / within_denominator
    visual_within = (
        visual_kernel.sum() - visual_kernel.diagonal().sum()
    ) / within_denominator
    return (eeg_within + visual_within - 2.0 * cross_kernel.mean()).clamp_min(0.0)


def _mmd_weight(
    epoch: int, optimization: OptimizationConfig, coarse_phase_start_epoch: int = 1,
) -> float:
    """Interpolate the MMD weight within the current coarse training phase."""
    if optimization.coarse_epochs <= coarse_phase_start_epoch:
        return optimization.mmd_weight_end
    progress = (epoch - coarse_phase_start_epoch) / (
        optimization.coarse_epochs - coarse_phase_start_epoch
    )
    progress = min(1.0, max(0.0, progress))
    return optimization.mmd_weight_start + progress * (
        optimization.mmd_weight_end - optimization.mmd_weight_start
    )
