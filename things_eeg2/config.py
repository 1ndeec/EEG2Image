"""Experiment configuration and checks performed once before training."""

from dataclasses import dataclass, field, replace
import importlib.util
from pathlib import Path
import sys

from .model import ModelConfig
from .paths import workspace_path
from .score_calibration import ScoreCalibrationConfig
from .visual_lora import VisualLoRAConfig


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths, held-out subject, and image-by-subject batching."""
    dataset_root: Path = Path("training_data/nm000232")
    visual_feature_dir: Path = Path(
        "training_data/nm000232/stimuli/internvit_multilevel_20_24_28_32_36"
    )
    held_out_subject: int = 10
    num_workers: int = 0
    pin_memory: bool = True
    evaluate_train_subjects_on_test: bool = True
    cross_subject_batches: bool = True
    images_per_batch: int = 112


@dataclass(frozen=True)
class OptimizationConfig:
    """Training stages, loss weights, and deterministic EEG augmentation."""
    epochs: int = 60
    coarse_epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-4
    fine_learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    temperature: float = 0.07
    loss_mode: str = "samga"
    paired_loss_version: str = "samga_full_batch_v1"
    score_alignment_scale: float = 1.0
    score_recovery_weight: float = 0.03
    score_recovery_temperature: float = 0.07
    score_recovery_identity_ratio: float = 0.1
    score_recovery_csls_neighbors: int = 10
    score_recovery_landmark_schedule: str = "per_epoch_v1"
    mmd_weight_start: float = 0.9
    mmd_weight_end: float = 0.5
    mmd_kernel_scales: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    smooth_kernel_size: int = 5
    smooth_channel_probability: float = 0.3
    frozen_smoothing: bool = True
    use_amp: bool = True
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    """Complete experiment settings saved in config.json and checkpoints."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visual_lora: VisualLoRAConfig = field(default_factory=VisualLoRAConfig)
    score: ScoreCalibrationConfig = field(default_factory=ScoreCalibrationConfig)
    checkpoint_root: Path = Path("checkpoints/things_eeg2_samga")
    run_name: str = "samga_9_to_1"
    resume_checkpoint: Path | None = None
    coarse_phase_start_epoch: int = 1


def resolve_paths(config: RunConfig) -> RunConfig:
    """Resolve data, pretrained weights, and checkpoint paths against the workspace."""
    return replace(
        config,
        data=replace(config.data, dataset_root=workspace_path(config.data.dataset_root),
                     visual_feature_dir=workspace_path(config.data.visual_feature_dir)),
        visual_lora=replace(config.visual_lora, checkpoint=workspace_path(config.visual_lora.checkpoint)),
        checkpoint_root=workspace_path(config.checkpoint_root),
        resume_checkpoint=(workspace_path(config.resume_checkpoint)
                           if config.resume_checkpoint is not None else None),
    )


def validate_run_config(config: RunConfig) -> tuple[int, ...]:
    """Validate an experiment and return its selected one-based LoRA blocks."""
    lora_blocks = ()
    opt = config.optimization
    if not 0 <= opt.coarse_epochs <= opt.epochs or opt.epochs < 1:
        raise ValueError("Require 0 <= coarse_epochs <= epochs and at least one epoch")
    if opt.temperature <= 0:
        raise ValueError("temperature must be positive")
    if opt.smooth_channel_probability > 0 and (opt.smooth_kernel_size < 1 or opt.smooth_kernel_size % 2 == 0):
        raise ValueError("smooth_kernel_size must be a positive odd integer")
    if config.score.enabled:
        config.score.validate(1654)
    if config.optimization.loss_mode not in {"samga", "score"}:
        raise ValueError("loss_mode must be 'samga' or 'score'")
    if config.optimization.paired_loss_version != "samga_full_batch_v1":
        raise ValueError("Unsupported paired_loss_version")
    if config.optimization.loss_mode == "score":
        if config.optimization.score_recovery_landmark_schedule not in {"per_epoch_v1", "fixed_v1"}:
            raise ValueError("Invalid SCORE recovery landmark schedule")
        if not config.data.cross_subject_batches:
            raise ValueError("SCORE loss requires cross_subject_batches=True")
        if (config.data.images_per_batch < 2
                or (config.visual_lora.enabled and config.visual_lora.images_per_batch < 2)):
            raise ValueError("SCORE recovery requires at least two images per batch")
        if (config.optimization.score_alignment_scale <= 0
                or config.optimization.score_recovery_weight < 0
                or config.optimization.score_recovery_temperature <= 0
                or config.optimization.score_recovery_identity_ratio < 0
                or config.optimization.score_recovery_csls_neighbors < 1):
            raise ValueError("Invalid SCORE recovery settings")
        if config.optimization.score_recovery_weight > 0 and not config.score.enabled:
            raise ValueError(
                "Support-to-query SCORE recovery requires score.enabled=True "
                "to define labeled support images separately from recovery queries"
            )
    if config.visual_lora.enabled:
        lora_blocks = config.visual_lora.selected_blocks(config.model.visual.layer_ids)
        if config.visual_lora.stage == "coarse" and config.optimization.coarse_epochs == 0:
            raise ValueError("Coarse-only visual LoRA requires at least one coarse epoch")
        if importlib.util.find_spec("open_clip") is None:
            raise RuntimeError(
                "Visual LoRA needs open_clip_torch in this Python kernel "
                f"({sys.executable}). Install it before starting the coarse stage."
            )
        if not config.data.cross_subject_batches:
            raise ValueError("Visual LoRA requires cross_subject_batches=True")
        if config.model.visual.raw_feature_dim != 768:
            raise ValueError("Visual LoRA requires 768-dimensional ViT-B/16 features")
        expected_feature_dir = f"vit_b16_laion2b_layers_2_10_{config.visual_lora.pooling}"
        if config.data.visual_feature_dir.name != expected_feature_dir:
            raise ValueError(
                "Visual LoRA pooling must match the cached feature directory: "
                f"expected {expected_feature_dir}"
            )
    return lora_blocks
