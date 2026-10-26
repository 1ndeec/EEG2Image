"""Coarse-to-fine training, checkpointing, and retrieval evaluation.

Configuration, datasets, and losses are defined in separate modules and
re-exported here for existing notebook and analysis imports."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ModelConfig, SAMGAModel, count_parameters
from .config import DataConfig, OptimizationConfig, RunConfig, resolve_paths, validate_run_config
from .data import (
    ALL_SUBJECTS, TRAIN_IMAGE_COUNT, TEST_IMAGE_COUNT, CrossSubjectImageBatchSampler,
    SAMGADataset, VisualFeatureBank, _make_loader, _smooth_channels, prepare_data,
    required_visual_feature_files, visual_feature_status,
)
from .losses import _mmd_weight, mmd_loss, symmetric_contrastive_loss
from .score_calibration import (
    ScoreCalibrationConfig,
    evaluate_score_calibration,
)
from .score_training import (
    fit_supervised_support_map,
    multi_positive_loss,
    select_subject_csls_landmarks,
    support_to_query_recovery_loss,
)
from .visual_lora import LiveViTFeatures, VisualLoRAConfig


SCORE_METHODS = ("uncalibrated", "random", "csls_selected")
SCORE_RANKINGS = ("cosine", "csls")
SCORE_R1_VARIANTS = tuple(
    f"{method}_{ranking}" for method in SCORE_METHODS for ranking in SCORE_RANKINGS
)


def _best_score_metrics(history: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Find the best completed epoch independently for each ranking and metric."""
    best: dict[str, dict[str, float | int]] = {}
    for method in SCORE_METHODS:
        for ranking in SCORE_RANKINGS:
            for metric in ("r1", "r5"):
                name = f"{method}_{ranking}_{metric}"
                key = f"validation_score_{name}"
                rows = (
                    row for row in history
                    if isinstance(row.get(key), (int, float))
                    and math.isfinite(float(row[key]))
                )
                winner = max(rows, key=lambda row: float(row[key]), default=None)
                if winner is not None:
                    best[name] = {"epoch": int(winner["epoch"]), "value": float(winner[key])}
    return best


def _best_score_r1_by_variant(history: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Select per-variant Top-1 checkpoints from recorded epoch metrics."""
    best = _best_score_metrics(history)
    return {
        variant: best[f"{variant}_r1"]
        for variant in SCORE_R1_VARIANTS
        if f"{variant}_r1" in best
    }


@torch.no_grad()
def _encode_global_gallery(
    model: SAMGAModel,
    visual_bank: VisualFeatureBank,
    device: torch.device,
    batch_size: int,
    live_features: Tensor | None = None,
    *,
    normalize_output: bool = True,
) -> Tensor:
    """Encode candidate images with global routing, in metadata order."""
    outputs: list[Tensor] = []
    for start in range(0, visual_bank.image_count, batch_size):
        stop = min(start + batch_size, visual_bank.image_count)
        if live_features is None:
            features = np.stack([visual_bank[index] for index in range(start, stop)])
            tensor = torch.from_numpy(features).to(device, non_blocking=True)
        else:
            tensor = live_features[start:stop].to(device, non_blocking=True)
        encoded, _ = model.encode_visual(tensor, force_global=True)
        outputs.append((F.normalize(encoded, dim=-1) if normalize_output else encoded).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate(
    model: SAMGAModel,
    dataset: SAMGADataset,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    live_visual_features: Tensor | None = None,
) -> dict[str, float]:
    """Rank held-out EEG against the full candidate gallery without subject residuals."""
    model.eval()
    gallery = _encode_global_gallery(
        model, dataset.visual, device, loader.batch_size or 200, live_visual_features
    )
    ranks: list[Tensor] = []
    losses: list[Tensor] = []
    class_hits: list[Tensor] = []
    gallery_classes = (
        torch.arange(dataset.images_per_subject) // 10
        if dataset.split == "training"
        else torch.arange(dataset.images_per_subject)
    )
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        targets = batch["image_index"].long()
        query_classes = batch["class_index"].long()
        encoded = F.normalize(model.encode_eeg(eeg), dim=-1).cpu()
        similarities = encoded @ gallery.T
        losses.append(F.cross_entropy(similarities / temperature, targets, reduction="none"))
        order = similarities.argsort(dim=1, descending=True)
        rank = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        ranks.append(rank)
        top_prediction = order[:, 0]
        class_hits.append(gallery_classes[top_prediction] == query_classes)
    all_ranks = torch.cat(ranks)
    return {
        "loss": float(torch.cat(losses).mean()),
        "r1": float((all_ranks < 1).float().mean()),
        "r5": float((all_ranks < 5).float().mean()),
        "ic": float(torch.cat(class_hits).float().mean()),
    }


@torch.no_grad()
def evaluate_score(
    model: SAMGAModel,
    support_dataset: SAMGADataset,
    support_loader: DataLoader,
    test_dataset: SAMGADataset,
    test_loader: DataLoader,
    device: torch.device,
    config: ScoreCalibrationConfig,
    support_image_ids: tuple[int, ...],
    live_visual: LiveViTFeatures | None,
    live_test_features: Tensor | None,
    *,
    amp_enabled: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    support_eeg, support_images, test_eeg, gallery, targets = collect_score_features(
        model, support_dataset, support_loader, test_dataset, test_loader,
        device, support_image_ids, live_visual, live_test_features,
        amp_enabled=amp_enabled,
    )
    return evaluate_score_calibration(
        support_eeg, support_images, test_eeg, gallery, targets, support_image_ids, config,
    )


def _select_training_recovery_landmarks(
    support_eeg: Tensor,
    support_visual: Tensor,
    train_subjects: Sequence[int],
    support_image_ids: tuple[int, ...],
    landmark_count: int,
    neighbors: int,
    epsilon: float,
    schedule: str,
    previous: dict[int, tuple[int, ...]] | None,
) -> dict[int, tuple[int, ...]]:
    """Reselect support pairs each epoch unless the fixed-selection ablation is set."""
    if schedule == "fixed_v1" and previous is not None:
        return previous
    rows_by_subject = select_subject_csls_landmarks(
        support_eeg, support_visual, landmark_count, neighbors, epsilon,
    ).tolist()
    return {
        subject_id: tuple(support_image_ids[row] for row in rows)
        for subject_id, rows in zip(train_subjects, rows_by_subject)
    }


@torch.no_grad()
def collect_score_features(
    model: SAMGAModel,
    support_dataset: SAMGADataset,
    support_loader: DataLoader,
    test_dataset: SAMGADataset,
    test_loader: DataLoader,
    device: torch.device,
    support_image_ids: tuple[int, ...],
    live_visual: LiveViTFeatures | None,
    live_test_features: Tensor | None,
    *,
    amp_enabled: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Encode labeled target support and test queries while preserving their row order."""
    model.eval()
    if live_visual is None:
        support_visual_features = None
    else:
        support_visual_features = live_visual.support_feature_bank(
            list(support_image_ids), amp_enabled=amp_enabled
        )
    support_eeg: list[Tensor] = []
    support_images: list[Tensor] = []
    observed_ids: list[int] = []
    for batch in support_loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        support_eeg.append(model.encode_eeg(eeg).float().cpu())
        ids = batch["image_index"].long().tolist()
        observed_ids.extend(ids)
        if support_visual_features is None:
            visual = batch["visual"].to(device, non_blocking=True)
        else:
            positions = [support_image_ids.index(image_id) for image_id in ids]
            visual = support_visual_features[positions].to(device)
        encoded, _ = model.encode_visual(visual, force_global=True)
        support_images.append(encoded.float().cpu())
    if tuple(observed_ids) != support_image_ids:
        raise RuntimeError("Support loader changed the calibration image ordering")

    test_eeg: list[Tensor] = []
    targets: list[Tensor] = []
    for batch in test_loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        test_eeg.append(model.encode_eeg(eeg).float().cpu())
        targets.append(batch["image_index"].long())
    gallery = _encode_global_gallery(
        model, test_dataset.visual, device, test_loader.batch_size or TEST_IMAGE_COUNT,
        live_test_features, normalize_output=False,
    )
    return (
        torch.cat(support_eeg), torch.cat(support_images), torch.cat(test_eeg),
        gallery, torch.cat(targets),
    )


@torch.no_grad()
def collect_training_support_features(
    model: SAMGAModel,
    support_dataset: SAMGADataset,
    support_loader: DataLoader,
    support_image_ids: tuple[int, ...],
    train_subjects: tuple[int, ...],
    device: torch.device,
    live_visual: LiveViTFeatures | None,
    *,
    amp_enabled: bool,
) -> tuple[Tensor, Tensor]:
    """Encode the common support pool separately for every training subject."""
    model.eval()
    if live_visual is None:
        raw_visual = torch.from_numpy(np.stack([
            support_dataset.visual[image_id] for image_id in support_image_ids
        ]))
    else:
        raw_visual = live_visual.support_feature_bank(
            list(support_image_ids), amp_enabled=amp_enabled
        )
    visual_chunks: list[Tensor] = []
    size = min(200, len(support_image_ids))
    for start in range(0, len(support_image_ids), size):
        raw = raw_visual[start:start + size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            encoded, _ = model.encode_visual(raw, force_global=True)
        visual_chunks.append(encoded.float().cpu())

    eeg_chunks: list[Tensor] = []
    observed_pairs: list[tuple[int, int]] = []
    for batch in support_loader:
        observed_pairs.extend(zip(
            batch["subject_id"].tolist(), batch["image_index"].tolist()
        ))
        eeg = batch["eeg"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            eeg_chunks.append(model.encode_eeg(eeg).float().cpu())
    expected_pairs = [
        (subject, image_id)
        for subject in train_subjects for image_id in support_image_ids
    ]
    if observed_pairs != expected_pairs:
        raise RuntimeError("Training calibration loader changed subject/image ordering")
    eeg_by_subject = torch.cat(eeg_chunks).reshape(
        len(train_subjects), len(support_image_ids), -1
    )
    return eeg_by_subject, torch.cat(visual_chunks)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _optimizer(
    model: SAMGAModel,
    config: OptimizationConfig,
    learning_rate: float,
    live_visual: LiveViTFeatures | None = None,
    visual_learning_rate: float | None = None,
):
    groups: list[dict[str, Any]] = [{
        "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
        "lr": learning_rate,
    }]
    if live_visual is not None:
        groups.append({
            "params": live_visual.trainable_parameters(),
            "lr": (live_visual.config.learning_rate
                   if visual_learning_rate is None else visual_learning_rate),
        })
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )


@torch.no_grad()
def _append_subject_router_weights(
    model: SAMGAModel,
    train_subjects: Sequence[int],
    path: Path,
    epoch: int,
) -> dict[str, Any]:
    router = model.visual_target.router
    subject_ids = torch.tensor(
        train_subjects, dtype=torch.long, device=router.global_logits.device
    )


    logits = router.global_logits.unsqueeze(0) + router.subject_residuals(subject_ids)
    subject_weights = torch.softmax(logits / router.config.temperature, dim=-1)
    record = {
        "epoch": epoch,
        "layer_ids": list(router.config.layer_ids),
        "global": router.global_weights().detach().cpu().tolist(),
        "subjects": {
            str(subject): weights
            for subject, weights in zip(
                train_subjects, subject_weights.detach().cpu().tolist()
            )
        },
    }
    with path.open("a", encoding="utf-8") as log_file:
        if log_file.tell() == 0:
            log_file.write(
                "Router weights (subject dropout disabled; columns follow layer order)\n"
            )
        log_file.write(
            f"\nEpoch {epoch:03d} | layers: "
            + " ".join(str(layer) for layer in record["layer_ids"])
            + "\n"
        )
        log_file.write(
            "  global     "
            + "  ".join(f"{weight:.6f}" for weight in record["global"])
            + "\n"
        )
        for subject, weights in record["subjects"].items():
            log_file.write(
                f"  subject-{int(subject):02d} "
                + "  ".join(f"{weight:.6f}" for weight in weights)
                + "\n"
            )
        log_file.flush()
        os.fsync(log_file.fileno())
    return record


def run_training(config: RunConfig) -> Path:
    """Train one held-out-subject fold and return its checkpoint directory."""
    config = resolve_paths(config)
    lora_blocks = validate_run_config(config)
    random.seed(config.optimization.seed)
    np.random.seed(config.optimization.seed)
    torch.manual_seed(config.optimization.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.optimization.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.visual_lora.enabled and config.visual_lora.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "Visual LoRA requires CUDA, but this notebook kernel reports "
            f"torch.cuda.is_available()=False (Python: {sys.executable}; "
            f"PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}). "
            "Use a CUDA-enabled PyTorch installation in this exact kernel, then restart it."
        )

    prepared = prepare_data(config)
    if config.data.cross_subject_batches:
        batch_sampler = CrossSubjectImageBatchSampler(
            prepared.train_dataset.images_per_subject,
            len(prepared.train_subjects),
            config.data.images_per_batch,
            seed=config.optimization.seed,
        )
        train_loader = DataLoader(
            prepared.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        )
        fine_batch_sampler = CrossSubjectImageBatchSampler(
            prepared.train_dataset.images_per_subject,
            len(prepared.train_subjects),
            config.visual_lora.images_per_batch,
            seed=config.optimization.seed,
        ) if config.visual_lora.enabled else None
        fine_train_loader = DataLoader(
            prepared.train_dataset,
            batch_sampler=fine_batch_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        ) if fine_batch_sampler is not None else None
    else:
        train_loader = _make_loader(
            prepared.train_dataset,
            config.data,
            config.optimization,
            shuffle=True,
        )
        fine_train_loader = None
    test_loader = _make_loader(
        prepared.test_dataset,
        config.data,
        config.optimization,
        shuffle=False,
        batch_size=200,
    )
    train_subjects_test_loader = None
    if prepared.train_subjects_test_dataset is not None:
        train_subjects_test_loader = _make_loader(
            prepared.train_subjects_test_dataset,
            config.data,
            config.optimization,
            shuffle=False,
            batch_size=200,
        )
    support_loader = None
    if prepared.support_dataset is not None:
        support_loader = _make_loader(
            prepared.support_dataset,
            config.data,
            config.optimization,
            shuffle=False,
            batch_size=min(config.score.pool_class_count, 200),
        )
    train_support_dataset = None
    train_support_loader = None
    if (config.optimization.loss_mode == "score"
            and config.optimization.score_recovery_weight > 0):
        train_support_dataset = SAMGADataset(
            config.data, config.model, prepared.train_subjects,
            "training", prepared.support_image_ids,
        )
        train_support_loader = _make_loader(
            train_support_dataset, config.data, config.optimization,
            shuffle=False, batch_size=min(config.score.pool_class_count, 200),
        )

    model = SAMGAModel(config.model).to(device)
    optimizer = _optimizer(model, config.optimization, config.optimization.learning_rate)
    amp_enabled = config.optimization.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_r1 = -math.inf
    best_score_r1 = -math.inf
    best_score_r1_by_variant: dict[str, dict[str, float | int]] = {}
    train_recovery_landmark_ids: dict[int, tuple[int, ...]] | None = None
    resume_source_directory: Path | None = None
    live_visual: LiveViTFeatures | None = None

    def create_live_visual() -> None:
        nonlocal live_visual
        if live_visual is None:
            live_visual = LiveViTFeatures(
                config.visual_lora,
                config.data.dataset_root,
                config.model.visual.layer_ids,
                device,
            )
            visual_trainable = sum(
                parameter.numel() for parameter in live_visual.trainable_parameters()
            )
            print(
                f"Live ViT trainable parameters: {visual_trainable:,} | "
                f"blocks {live_visual.lora_blocks} | "
                f"LayerNorm {config.visual_lora.tune_layer_norm} | "
                f"K {config.visual_lora.lora_key} | "
                f"attention output {config.visual_lora.lora_output} | "
                f"MLP {config.visual_lora.lora_mlp}",
                flush=True,
            )

    def start_fine_stage() -> None:
        nonlocal optimizer, live_visual
        model.freeze_shared_encoder()
        if config.visual_lora.enabled:
            create_live_visual()
        optimizer = _optimizer(
            model, config.optimization, config.optimization.fine_learning_rate,
            live_visual if config.visual_lora.stage in {"fine", "both"} else None,
        )

    if config.visual_lora.enabled and config.visual_lora.stage in {"coarse", "both"}:
        create_live_visual()
        optimizer = _optimizer(
            model, config.optimization, config.optimization.learning_rate, live_visual,
            visual_learning_rate=config.visual_lora.coarse_learning_rate,
        )

    if config.resume_checkpoint is not None:
        resume_source_directory = config.resume_checkpoint.parent
        payload = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        previous_optimization = payload.get("config", {}).get("optimization", {})
        previous_loss_mode = previous_optimization.get("loss_mode", "samga")
        if previous_loss_mode != config.optimization.loss_mode:
            raise ValueError(
                f"Cannot resume {previous_loss_mode} loss with {config.optimization.loss_mode} loss"
            )
        if config.optimization.loss_mode == "score":
            previous_schedule = previous_optimization.get(
                "score_recovery_landmark_schedule", "fixed_v1"
            )
            if previous_schedule != config.optimization.score_recovery_landmark_schedule:
                raise ValueError(
                    "Cannot resume SCORE checkpoint with a different recovery "
                    "landmark schedule; start a new run"
                )
        if (config.optimization.loss_mode == "samga"
                and previous_optimization.get("paired_loss_version")
                != config.optimization.paired_loss_version
                and payload.get("config", {}).get("data", {}).get("cross_subject_batches", True)):
            raise ValueError(
                "Checkpoint paired-loss version differs from the current full-batch loss; "
                "start a new run."
            )
        previous_score = payload.get("config", {}).get("score", {})
        if previous_score != _jsonable(asdict(config.score)):
            if previous_score or config.score.enabled:
                raise ValueError(
                    "Resume checkpoint SCORE settings differ"
                )
        checkpoint_subject = payload.get(
            "held_out_subject", payload.get("config", {}).get("data", {}).get("held_out_subject")
        )
        if checkpoint_subject != config.data.held_out_subject:
            raise ValueError(
                f"Cannot resume subject {config.data.held_out_subject:02d} from "
                f"{config.resume_checkpoint}: checkpoint holds out subject "
                f"{checkpoint_subject}."
            )
        resumed_epoch = int(payload["epoch"])
        saved_landmarks = payload.get("train_recovery_landmark_image_ids")
        if saved_landmarks is not None:
            if isinstance(saved_landmarks, dict):
                train_recovery_landmark_ids = {
                    int(subject): tuple(int(index) for index in indices)
                    for subject, indices in saved_landmarks.items()
                }
            else:
                raise ValueError("Recovery checkpoint must store selected image IDs for each source subject")
            if set(train_recovery_landmark_ids) != set(prepared.train_subjects):
                raise ValueError("Resume checkpoint recovery subjects differ")
            for indices in train_recovery_landmark_ids.values():
                if (len(indices) != config.score.landmark_count
                        or len(set(indices)) != len(indices)
                        or not set(indices).issubset(prepared.support_image_ids)):
                    raise ValueError("Resume checkpoint training recovery landmarks differ")
        adapter_state = payload.get("visual_lora_state_dict")
        reset_optimizer = False
        old_coarse_epochs = int(previous_optimization.get("coarse_epochs", config.optimization.coarse_epochs))
        if (
            resumed_epoch > old_coarse_epochs
            and config.optimization.coarse_epochs > resumed_epoch
        ):


            reset_optimizer = True
            print(
                f"Checkpoint epoch {resumed_epoch} finished its original coarse stage "
                f"at epoch {old_coarse_epochs}; starting a new coarse/fine cycle "
                "with fresh optimizer state.",
                flush=True,
            )
        if config.visual_lora.enabled and not adapter_state:
            if resumed_epoch > old_coarse_epochs:
                raise ValueError("Fine-stage checkpoint has no visual adapter weights; resume a coarse checkpoint instead")
            reset_optimizer = reset_optimizer or config.visual_lora.stage in {"coarse", "both"}
        model.load_state_dict(payload["model_state_dict"])
        start_epoch = resumed_epoch + 1
        history = list(payload.get("history", []))
        best_r1 = float(payload.get("best_validation_r1", -math.inf))
        best_score_r1 = float(payload.get("best_score_r1", -math.inf))
        best_score_r1_by_variant = _best_score_r1_by_variant(history)
        if config.visual_lora.enabled and adapter_state:
            old_lora = dict(payload.get("config", {}).get("visual_lora") or {})
            defaults = _jsonable(asdict(VisualLoRAConfig()))
            old_lora = {**defaults, **old_lora}
            new_lora = _jsonable(asdict(config.visual_lora))


            for key in ("learning_rate", "coarse_learning_rate", "stage"):
                old_lora.pop(key, None)
                new_lora.pop(key, None)
            if old_lora != new_lora:
                raise ValueError("Resume checkpoint visual LoRA settings differ from current config")
            old_stage = (payload.get("config", {}).get("visual_lora") or {}).get("stage", "fine")
            if old_stage != config.visual_lora.stage:
                reset_optimizer = True
                print("Visual LoRA stage changed; initializing an optimizer for the new stage.", flush=True)
        if resumed_epoch > config.optimization.coarse_epochs and (not reset_optimizer or adapter_state):
            start_fine_stage()
        if live_visual is not None and adapter_state:
            live_visual.load_adapter_state_dict(adapter_state)
        if not reset_optimizer:
            optimizer.load_state_dict(payload["optimizer_state_dict"])


            in_fine_stage = resumed_epoch > config.optimization.coarse_epochs
            optimizer.param_groups[0]["lr"] = (
                config.optimization.fine_learning_rate if in_fine_stage
                else config.optimization.learning_rate
            )
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = (
                    config.visual_lora.learning_rate if in_fine_stage
                    else (config.visual_lora.coarse_learning_rate
                          or config.visual_lora.learning_rate)
                )
        if not reset_optimizer and "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = config.checkpoint_root / f"{config.run_name}_{stamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    if resume_source_directory is not None:


        source_best = resume_source_directory / "best.pt"
        if source_best.is_file() and math.isfinite(best_r1):
            prior_best = torch.load(source_best, map_location="cpu", weights_only=False, mmap=True)
            prior_best_history = prior_best.get("history") or []
            if (
                int(prior_best["epoch"]) <= resumed_epoch
                and prior_best_history
                and math.isclose(
                    float(prior_best_history[-1]["validation_test_r1"]), best_r1,
                    rel_tol=0.0, abs_tol=1e-9,
                )
            ):
                shutil.copy2(source_best, run_directory / "best.pt")
            del prior_best
        source_manifest = resume_source_directory / "score_best_r1_checkpoints.json"
        source_winners = (
            json.loads(source_manifest.read_text(encoding="utf-8"))
            if source_manifest.is_file() else {}
        )
        for variant in SCORE_R1_VARIANTS:
            if variant not in best_score_r1_by_variant:
                continue
            name = f"best_score_{variant}.pt"
            source = resume_source_directory / name
            source_winner = source_winners.get(variant, {})
            expected_winner = best_score_r1_by_variant[variant]
            if (
                source.is_file()
                and source_winner.get("checkpoint") == name
                and source_winner.get("epoch") == expected_winner["epoch"]
                and source_winner.get("value") == expected_winner["value"]
            ):
                shutil.copy2(source, run_directory / name)
    (run_directory / "config.json").write_text(
        json.dumps(_jsonable(asdict(config)), indent=2),
        encoding="utf-8",
    )
    if config.score.enabled:
        (run_directory / "score_support_images.json").write_text(
            json.dumps({
                "seed": config.score.seed,
                "pool_class_count": config.score.pool_class_count,
                "landmark_count": config.score.landmark_count,
                "image_indices": list(prepared.support_image_ids),
                "class_indices": [index // 10 for index in prepared.support_image_ids],
                "source_training_images": prepared.train_dataset.images_per_subject,
                "target_test_images": TEST_IMAGE_COUNT,
            }, indent=2), encoding="utf-8",
        )

    total, trainable = count_parameters(model)
    print(f"Device: {device} | SAMGA parameters: {total:,} | trainable: {trainable:,}")
    print(f"Train subjects: {prepared.train_subjects} | held out: {config.data.held_out_subject:02d}")
    print(f"Train samples: {len(prepared.train_dataset):,} (averaged repetitions)")
    if config.data.cross_subject_batches:
        print(
            f"Train batching: {config.data.images_per_batch} images x "
            f"{len(prepared.train_subjects)} subjects = "
            f"{config.data.images_per_batch * len(prepared.train_subjects)} "
            "EEG trials per full batch"
        )
        if config.visual_lora.enabled:
            print(
                f"Live ViT ({config.visual_lora.stage} stage): "
                f"{config.visual_lora.images_per_batch} images x "
                f"{len(prepared.train_subjects)} subjects; base Q/V LoRA rank "
                f"{config.visual_lora.rank} on ViT blocks {lora_blocks}; "
                f"alpha {config.visual_lora.alpha:g}; max adapted-weight update ratio "
                f"{config.visual_lora.max_update_ratio}; gradient-cache image chunk "
                f"{config.visual_lora.gradient_cache_microbatch_size}; visual LR "
                f"coarse={(config.visual_lora.coarse_learning_rate or config.visual_lora.learning_rate) if config.visual_lora.stage in {'coarse', 'both'} else 'off'}, "
                f"fine={config.visual_lora.learning_rate if config.visual_lora.stage in {'fine', 'both'} else 'off'}"
            )
    else:
        print(f"Train batching: {config.optimization.batch_size} shuffled EEG trials")
    print(f"Training loss: {config.optimization.loss_mode}")
    if config.optimization.loss_mode == "score":
        print(
            f"SCORE alignment scale {config.optimization.score_alignment_scale:g}; "
            f"support-to-query recovery weight {config.optimization.score_recovery_weight:g}, "
            f"temperature {config.optimization.score_recovery_temperature:g}, "
            f"identity ratio {config.optimization.score_recovery_identity_ratio:g}, "
            f"{config.optimization.score_recovery_landmark_schedule} landmark "
            f"CSLS neighbors "
            f"{config.optimization.score_recovery_csls_neighbors}"
        )
    print(f"Validation/test samples: {len(prepared.test_dataset):,}")
    if config.score.enabled:
        print(
            f"SCORE support: {len(prepared.support_image_ids)} fixed images, "
            f"{config.score.landmark_count} map landmarks; source subjects keep "
            "these images in ordinary training; held-out subject used only for evaluation"
        )
    print(f"Run directory: {run_directory.resolve()}")

    router_log = run_directory / "router_subject_weights.txt"
    _append_subject_router_weights(
        model, prepared.train_subjects, router_log, start_epoch - 1
    )


    # A checkpoint at the coarse boundary still needs the fine-stage transition.
    stage_two_started = start_epoch > config.optimization.coarse_epochs + 1
    for epoch in range(start_epoch, config.optimization.epochs + 1):
        if isinstance(train_loader.batch_sampler, CrossSubjectImageBatchSampler):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        if fine_train_loader is not None:
            fine_train_loader.batch_sampler.set_epoch(epoch - 1)
        if epoch > config.optimization.coarse_epochs and not stage_two_started:
            start_fine_stage()
            stage_two_started = True
            print(
                f"Epoch {epoch:03d}: froze shared encoder and switched to "
                f"fine-stage LR {config.optimization.fine_learning_rate:g}"
            )

        recovery_maps: dict[int, dict[str, Tensor]] = {}
        if train_support_dataset is not None and train_support_loader is not None:
            support_eeg, support_visual = collect_training_support_features(
                model, train_support_dataset, train_support_loader,
                prepared.support_image_ids, prepared.train_subjects,
                device, live_visual, amp_enabled=amp_enabled,
            )
            train_recovery_landmark_ids = _select_training_recovery_landmarks(
                support_eeg, support_visual, prepared.train_subjects,
                prepared.support_image_ids, config.score.landmark_count,
                config.optimization.score_recovery_csls_neighbors,
                config.score.epsilon,
                config.optimization.score_recovery_landmark_schedule,
                train_recovery_landmark_ids,
            )
            if epoch == start_epoch:
                print(
                    f"Training recovery: {config.score.landmark_count} "
                    "CSLS-selected calibration images per source subject; "
                    f"selection schedule {config.optimization.score_recovery_landmark_schedule}",
                    flush=True,
                )
            landmark_log = run_directory / "train_recovery_landmarks.json"
            landmark_record = {
                "epoch": epoch,
                "selection": "true_pair_csls_margin_per_source_subject",
                "schedule": config.optimization.score_recovery_landmark_schedule,
                "image_indices_by_subject": {
                    str(subject): list(indices)
                    for subject, indices in train_recovery_landmark_ids.items()
                },
            }
            with (run_directory / "train_recovery_landmarks_by_epoch.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(landmark_record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            pending_log = landmark_log.with_suffix(".json.partial")
            pending_log.write_text(json.dumps(landmark_record, indent=2), encoding="utf-8")
            pending_log.replace(landmark_log)
            for subject_position, subject_id in enumerate(prepared.train_subjects):
                landmark_rows = [
                    prepared.support_image_ids.index(image_id)
                    for image_id in train_recovery_landmark_ids[subject_id]
                ]
                fitted = fit_supervised_support_map(
                    support_eeg[subject_position], support_visual, landmark_rows,
                    config.optimization.score_recovery_identity_ratio,
                    config.score.epsilon,
                )
                recovery_maps[subject_id] = {
                    name: value.to(device, non_blocking=True)
                    for name, value in fitted.items()
                }

        model.train()
        train_visual_this_epoch = (
            live_visual is not None
            and (epoch <= config.optimization.coarse_epochs
                 or config.visual_lora.stage in {"fine", "both"})
        )
        if live_visual is not None:
            live_visual.visual.train(train_visual_this_epoch)
        total_loss = 0.0
        total_retrieval = 0.0
        total_mmd = 0.0
        total_recovery = 0.0
        total_recovery_landmarks = 0
        batches = 0
        recovery_support_ids = (
            torch.tensor(prepared.support_image_ids, device=device)
            if recovery_maps else None
        )
        epoch_loader = fine_train_loader if live_visual is not None else train_loader
        for batch in epoch_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            subjects = batch["subject_id"].to(device, non_blocking=True)
            image_indices = batch["image_index"].to(device, non_blocking=True)
            eeg = _smooth_channels(
                eeg,
                config.optimization.smooth_kernel_size,
                config.optimization.smooth_channel_probability,
                subject_ids=subjects,
                image_indices=image_indices,
                seed=config.optimization.seed,
                frozen=config.optimization.frozen_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            feature_leaf: Tensor | None = None
            unique_image_ids: list[int] | None = None
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if live_visual is None:
                    visual = batch["visual"].to(device, non_blocking=True)
                else:
                    unique, inverse = torch.unique(
                        image_indices, sorted=True, return_inverse=True
                    )
                    unique_image_ids = unique.cpu().tolist()
                    if not train_visual_this_epoch:
                        with torch.no_grad():
                            unique_features = live_visual.encode(unique_image_ids, "train")
                    elif config.visual_lora.gradient_cache_microbatch_size is None:
                        unique_features = live_visual.encode(unique_image_ids, "train")
                    else:
                        feature_leaf = live_visual.encode_gradient_cached(
                            unique_image_ids, amp_enabled=amp_enabled
                        )
                        unique_features = feature_leaf
                    visual = unique_features.index_select(0, inverse)
                recovery_landmarks = 0
                if recovery_maps:
                    source_subjects = torch.unique(subjects, sorted=True)
                    pseudo_subject = source_subjects[
                        torch.randint(source_subjects.numel(), (), device=device)
                    ]
                    pseudo_mask = subjects.eq(pseudo_subject)
                    source_mask = ~pseudo_mask
                    query_mask = pseudo_mask & ~torch.isin(
                        image_indices, recovery_support_ids
                    )
                    eeg_embedding = model.encode_eeg(eeg[source_mask])
                    visual_embedding, _ = model.encode_visual(
                        visual[source_mask], subjects[source_mask]
                    )
                    aligned_images = image_indices[source_mask]
                else:
                    eeg_embedding = model.encode_eeg(eeg)
                    visual_embedding, _ = model.encode_visual(visual, subjects)
                    aligned_images = image_indices
                recovery = eeg_embedding.new_zeros(())
                if config.optimization.loss_mode == "score":
                    retrieval = config.optimization.score_alignment_scale * multi_positive_loss(
                        eeg_embedding, visual_embedding, aligned_images,
                        config.optimization.temperature,
                    )
                    if recovery_maps:
                        pseudo_images = image_indices[query_mask]
                        if pseudo_images.numel() > 1:


                            # Pseudo-target queries must not update source BatchNorm statistics.
                            was_training = model.eeg_encoder.training
                            model.eeg_encoder.eval()
                            try:
                                query_eeg_embedding = model.encode_eeg(eeg[query_mask])
                            finally:
                                model.eeg_encoder.train(was_training)
                            global_visual, _ = model.encode_visual(
                                visual[query_mask], force_global=True
                            )
                            recovery = support_to_query_recovery_loss(
                                query_eeg_embedding, global_visual,
                                recovery_maps[int(pseudo_subject.item())],
                                config.optimization.score_recovery_temperature,
                            )
                            recovery_landmarks = len(
                                train_recovery_landmark_ids[int(pseudo_subject.item())]
                            )
                else:
                    retrieval = symmetric_contrastive_loss(
                        eeg_embedding,
                        visual_embedding,
                        config.optimization.temperature,
                    )
                if epoch <= config.optimization.coarse_epochs:
                    mmd = mmd_loss(
                        eeg_embedding.float(), visual_embedding.float(),
                        config.optimization.mmd_kernel_scales,
                    )
                    weight = _mmd_weight(
                        epoch, config.optimization, config.coarse_phase_start_epoch
                    )
                    loss = weight * mmd + (1.0 - weight) * retrieval
                else:
                    mmd = retrieval.new_zeros(())
                    weight = 0.0
                    loss = retrieval
                if config.optimization.loss_mode == "score":


                    # Recovery is added in both stages; MMD remains coarse-only.
                    loss = loss + config.optimization.score_recovery_weight * recovery
            scaler.scale(loss).backward()
            if feature_leaf is not None:
                if feature_leaf.grad is None or unique_image_ids is None:
                    raise RuntimeError("Visual feature gradient was not produced")
                live_visual.backward_gradient_cached(
                    unique_image_ids, feature_leaf.grad, amp_enabled=amp_enabled
                )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            total_retrieval += float(retrieval.detach())
            total_mmd += float(mmd.detach())
            total_recovery += float(recovery.detach())
            total_recovery_landmarks += recovery_landmarks
            batches += 1

        router_snapshot = _append_subject_router_weights(
            model, prepared.train_subjects, router_log, epoch
        )
        live_test_features = (
            live_visual.test_feature_bank(amp_enabled=amp_enabled)
            if live_visual is not None else None
        )
        test_metrics = evaluate(
            model,
            prepared.test_dataset,
            test_loader,
            device,
            config.optimization.temperature,
            live_test_features,
        )
        score_metrics: dict[str, float] = {}
        score_artifacts: dict[str, object] | None = None
        if support_loader is not None and prepared.support_dataset is not None:
            score_metrics, score_artifacts = evaluate_score(
                model, prepared.support_dataset, support_loader,
                prepared.test_dataset, test_loader, device, config.score,
                prepared.support_image_ids, live_visual, live_test_features,
                amp_enabled=amp_enabled,
            )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_total": total_loss / batches,
            "train_retrieval": total_retrieval / batches,
            "train_mmd": total_mmd / batches,
            "train_recovery": total_recovery / batches,
            "train_recovery_landmarks": total_recovery_landmarks / batches,
            "loss_mode": config.optimization.loss_mode,
            "mmd_weight": weight,
            "validation_test_loss": test_metrics["loss"],
            "validation_test_r1": test_metrics["r1"],
            "validation_test_r5": test_metrics["r5"],
            "validation_test_ic": test_metrics["ic"],
            "global_router_weights": router_snapshot["global"],
            "subject_router_weights": router_snapshot["subjects"],
        }
        record.update({f"validation_score_{name}": value for name, value in score_metrics.items()})
        if (
            prepared.train_subjects_test_dataset is not None
            and train_subjects_test_loader is not None
        ):
            metrics = evaluate(
                model,
                prepared.train_subjects_test_dataset,
                train_subjects_test_loader,
                device,
                config.optimization.temperature,
                live_test_features,
            )
            for name, value in metrics.items():
                record[f"validation_train_subjects_test_{name}"] = value
        history.append(record)
        (run_directory / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        if score_metrics:
            score_best = {
                "held_out_subject": config.data.held_out_subject,
                "evaluated_epochs": sum(
                    "validation_score_uncalibrated_cosine_r1" in row for row in history
                ),
                "best_by_metric": _best_score_metrics(history),
            }
            (run_directory / "score_best_by_metric.json").write_text(
                json.dumps(score_best, indent=2), encoding="utf-8"
            )

        improved_score_variants: list[str] = []
        for variant in SCORE_R1_VARIANTS:
            value = score_metrics.get(f"{variant}_r1")
            if value is None or not math.isfinite(value):
                continue
            previous = best_score_r1_by_variant.get(variant)
            if previous is None or value > float(previous["value"]):
                best_score_r1_by_variant[variant] = {"epoch": epoch, "value": value}
                improved_score_variants.append(variant)

        calibrated_r1 = {
            f"{method}_{ranking}": score_metrics[f"{method}_{ranking}_r1"]
            for method in ("random", "csls_selected")
            for ranking in SCORE_RANKINGS
            if f"{method}_{ranking}_r1" in score_metrics
        }
        winning_variant = max(calibrated_r1, key=calibrated_r1.get) if calibrated_r1 else None
        improved_calibrated_score = (
            winning_variant is not None
            and calibrated_r1[winning_variant] > best_score_r1
        )
        if improved_calibrated_score:
            best_score_r1 = calibrated_r1[winning_variant]

        checkpoint = {
            "epoch": epoch,
            "held_out_subject": config.data.held_out_subject,
            "train_subjects": list(prepared.train_subjects),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "router_layer_ids": router_snapshot["layer_ids"],
            "subject_router_weights": router_snapshot["subjects"],
            "best_validation_r1": max(best_r1, test_metrics["r1"]),
            "best_score_r1": best_score_r1,
            "best_score_r1_by_variant": dict(best_score_r1_by_variant),
            "config": _jsonable(asdict(config)),
            "visual_encoder_config": {
                "backbone": "ViT-B-16",
                "visual_router": _jsonable(asdict(config.model.visual)),
                "visual_lora": _jsonable(asdict(config.visual_lora)),
            },
        }
        if live_visual is not None:
            checkpoint["visual_lora_state_dict"] = live_visual.adapter_state_dict()
        if train_recovery_landmark_ids is not None:
            checkpoint["train_recovery_landmark_image_ids"] = {
                str(subject): list(indices)
                for subject, indices in train_recovery_landmark_ids.items()
            }
        if score_artifacts is not None:
            checkpoint["score_calibration"] = score_artifacts
            checkpoint["score_best_variant_this_epoch"] = winning_variant
        for variant in improved_score_variants:
            torch.save(checkpoint, run_directory / f"best_score_{variant}.pt")
        if improved_calibrated_score:
            torch.save(checkpoint, run_directory / "best_score.pt")
        torch.save(checkpoint, run_directory / "last.pt")
        if test_metrics["r1"] > best_r1:
            best_r1 = test_metrics["r1"]
            checkpoint["best_validation_r1"] = best_r1
            torch.save(checkpoint, run_directory / "best.pt")
        if score_metrics:
            (run_directory / "score_best_r1_checkpoints.json").write_text(
                json.dumps({
                    variant: {
                        **winner,
                        "checkpoint": (
                            f"best_score_{variant}.pt"
                            if (run_directory / f"best_score_{variant}.pt").is_file()
                            else None
                        ),
                    }
                    for variant, winner in best_score_r1_by_variant.items()
                }, indent=2),
                encoding="utf-8",
            )

        router = ", ".join(f"{value:.3f}" for value in record["global_router_weights"])
        recovery_log = (
            f"recovery {record['train_recovery']:.4f} "
            f"landmarks {record['train_recovery_landmarks']:.1f} "
            if config.optimization.loss_mode == "score" else ""
        )
        print(
            f"Subject {config.data.held_out_subject:02d} | Epoch {epoch:03d} | "
            f"train total {record['train_total']:.4f} "
            f"retrieval {record['train_retrieval']:.4f} mmd {record['train_mmd']:.4f} "
            f"{recovery_log}w {weight:.3f} | held-out/test loss {test_metrics['loss']:.4f} "
            f"R@1 {test_metrics['r1']:.4f} R@5 {test_metrics['r5']:.4f} "
            f"IC {test_metrics['ic']:.4f} | router [{router}]"
        )
        if score_metrics:
            print(
                "  SCORE test R@1/R@5 | " + " | ".join(
                    f"{method} {ranking}: "
                    f"{score_metrics[f'{method}_{ranking}_r1']:.4f}/"
                    f"{score_metrics[f'{method}_{ranking}_r5']:.4f}"
                    for method in ("uncalibrated", "random", "csls_selected")
                    for ranking in ("cosine", "csls")
                ),
                flush=True,
            )

    if config.score.enabled:
        for name, winner in _best_score_metrics(history).items():
            print(f"Best SCORE {name}: {winner['value']:.4f} at epoch {winner['epoch']:03d}")
    if live_visual is not None:
        live_visual.close()
    return run_directory
