"""TSConv EEG encoding and subject-aware mixtures of intermediate visual features.

The visual backbone lives in visual_lora.py; this module accepts its per-layer
feature vectors and maps both modalities into one retrieval space."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EEGEncoderConfig:
    """Temporal/spatial convolution sizes and EEG projection width."""
    channels: int = 63
    samples: int = 250
    temporal_filters: int = 40
    temporal_kernel: int = 25
    pooling_kernel: int = 51
    pooling_stride: int = 5
    dropout: float = 0.5
    feature_dim: int = 1024


@dataclass(frozen=True)
class VisualRouterConfig:
    """Visual depths, projection widths, and subject/global routing settings."""
    layer_ids: tuple[int, ...] = (20, 24, 28, 32, 36)
    raw_feature_dim: int = 3200
    intermediate_dim: int = 1024
    alignment_dim: int = 512
    prior_center: int = 28
    prior_strength: float = 1.0
    temperature: float = 1.0
    subject_dropout: float = 0.3
    layer_dropout: float = 0.0
    max_subject_id: int = 10


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings serialized alongside every training checkpoint."""
    eeg: EEGEncoderConfig = field(default_factory=EEGEncoderConfig)
    visual: VisualRouterConfig = field(default_factory=VisualRouterConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualProjection(nn.Module):
    """Linear projection with a residual MLP and layer normalization."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, output_dim)
        self.residual = nn.Sequential(
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        return self.norm(x + self.residual(x))


class TSConvEncoder(nn.Module):
    """Encode EEG [batch, channels, samples] into [batch, feature_dim]."""

    def __init__(self, config: EEGEncoderConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config
        if cfg.temporal_kernel > cfg.samples:
            raise ValueError("temporal_kernel cannot exceed EEG sample count")
        convolved = cfg.samples - cfg.temporal_kernel + 1
        pooled = (convolved - cfg.pooling_kernel) // cfg.pooling_stride + 1
        if pooled < 1:
            raise ValueError("TSConv pooling reduces the time axis to zero")
        self.output_time_points = pooled
        self.flattened_dim = cfg.temporal_filters * pooled

        self.backbone = nn.Sequential(
            nn.Conv2d(1, cfg.temporal_filters, (1, cfg.temporal_kernel)),
            nn.AvgPool2d((1, cfg.pooling_kernel), (1, cfg.pooling_stride)),
            nn.BatchNorm2d(cfg.temporal_filters),
            nn.ELU(),
            nn.Conv2d(
                cfg.temporal_filters,
                cfg.temporal_filters,
                (cfg.channels, 1),
            ),
            nn.BatchNorm2d(cfg.temporal_filters),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv2d(cfg.temporal_filters, cfg.temporal_filters, (1, 1)),
        )
        self.projection = ResidualProjection(
            self.flattened_dim,
            cfg.feature_dim,
            cfg.dropout,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        x = self.backbone(eeg.unsqueeze(1))
        return self.projection(x.flatten(start_dim=1))


class SubjectAwareLayerRouter(nn.Module):
    """Combine a global depth prior with residuals learned for training subjects.

    Use force_global=True for subjects that were not seen during training."""

    def __init__(self, config: VisualRouterConfig) -> None:
        super().__init__()
        self.config = config
        layer_ids = torch.tensor(config.layer_ids, dtype=torch.float32)
        self.register_buffer("layer_ids", layer_ids, persistent=True)
        if len(config.layer_ids) > 1:
            ordered = sorted(config.layer_ids)
            step = min(b - a for a, b in zip(ordered, ordered[1:]) if b > a)
        else:
            step = 1
        initial = -config.prior_strength * (
            (layer_ids - float(config.prior_center)).abs() / float(step)
        )
        self.global_logits = nn.Parameter(initial)
        self.subject_residuals = nn.Embedding(
            config.max_subject_id + 1,
            len(config.layer_ids),
        )
        nn.init.zeros_(self.subject_residuals.weight)

    def forward(
        self,
        subject_ids: Tensor | None,
        *,
        force_global: bool,
    ) -> Tensor:
        if subject_ids is None:
            logits = self.global_logits.unsqueeze(0)
        else:
            logits = self.global_logits.unsqueeze(0).expand(subject_ids.shape[0], -1)
            if not force_global:
                residual = self.subject_residuals(subject_ids.long())
                if self.training and self.config.subject_dropout > 0:
                    keep = (
                        torch.rand(subject_ids.shape[0], 1, device=subject_ids.device)
                        >= self.config.subject_dropout
                    )
                    residual = residual * keep
                logits = logits + residual
        return torch.softmax(logits / self.config.temperature, dim=-1)

    def global_weights(self) -> Tensor:
        return torch.softmax(self.global_logits / self.config.temperature, dim=-1)


class MultiGranularityVisualTarget(nn.Module):
    """Project [batch, layers, width] features and mix them with router weights."""

    def __init__(self, config: VisualRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_projector = nn.Linear(
            config.raw_feature_dim,
            config.intermediate_dim,
        )
        self.layer_projectors = nn.ModuleList(
            nn.Linear(config.intermediate_dim, config.alignment_dim)
            for _ in config.layer_ids
        )
        self.router = SubjectAwareLayerRouter(config)

    def _apply_layer_dropout(self, weights: Tensor) -> Tensor:
        probability = self.config.layer_dropout
        if not self.training or probability <= 0:
            return weights
        keep = torch.rand_like(weights) >= probability
        empty = ~keep.any(dim=-1, keepdim=True)
        keep = torch.where(empty, torch.ones_like(keep), keep)
        masked = weights * keep
        return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        visual_features: Tensor,
        subject_ids: Tensor | None = None,
        *,
        force_global: bool = False,
    ) -> tuple[Tensor, Tensor]:
        expected = (len(self.config.layer_ids), self.config.raw_feature_dim)
        if visual_features.ndim != 3 or visual_features.shape[1:] != expected:
            raise ValueError(
                f"Expected visual features [B,{expected[0]},{expected[1]}], "
                f"got {tuple(visual_features.shape)}"
            )
        shared = self.pre_projector(visual_features)
        projected = torch.stack(
            [projector(shared[:, index]) for index, projector in enumerate(self.layer_projectors)],
            dim=1,
        )
        weights = self.router(subject_ids, force_global=force_global)
        if weights.shape[0] == 1 and projected.shape[0] != 1:
            weights = weights.expand(projected.shape[0], -1)
        weights = self._apply_layer_dropout(weights)
        return (projected * weights.unsqueeze(-1)).sum(dim=1), weights


class SAMGAModel(nn.Module):
    """EEG and visual branches followed by the same linear alignment encoder."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.eeg_encoder = TSConvEncoder(cfg.eeg)
        self.eeg_projector = nn.Linear(
            cfg.eeg.feature_dim,
            cfg.visual.alignment_dim,
        )
        self.visual_target = MultiGranularityVisualTarget(cfg.visual)
        self.shared_encoder = nn.Linear(
            cfg.visual.alignment_dim,
            cfg.visual.alignment_dim,
        )

    def encode_eeg(self, eeg: Tensor) -> Tensor:
        return self.shared_encoder(self.eeg_projector(self.eeg_encoder(eeg)))

    def encode_visual(
        self,
        visual_features: Tensor,
        subject_ids: Tensor | None = None,
        *,
        force_global: bool = False,
    ) -> tuple[Tensor, Tensor]:
        target, weights = self.visual_target(
            visual_features,
            subject_ids,
            force_global=force_global,
        )
        return self.shared_encoder(target), weights

    def freeze_shared_encoder(self) -> None:
        """Freeze the common alignment map at the coarse-to-fine transition."""
        self.shared_encoder.requires_grad_(False)
        self.shared_encoder.eval()

    def train(self, mode: bool = True) -> "SAMGAModel":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.shared_encoder.parameters()):
            self.shared_encoder.eval()
        return self


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
