"""Memory-mapped EEG/visual arrays and reproducible cross-subject batches."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .config import DataConfig, OptimizationConfig, RunConfig
from .model import ModelConfig
from .paths import workspace_path
from .score_calibration import select_support_images

TRAIN_IMAGE_COUNT = 16_540
TEST_IMAGE_COUNT = 200
ALL_SUBJECTS = tuple(range(1, 11))


def _subject_dir(root: Path, subject: int) -> Path:
    return workspace_path(root) / "derivatives" / "nice_250hz" / f"sub-{subject:02d}"


def _eeg_path(root: Path, subject: int, split: str) -> Path:
    return _subject_dir(root, subject) / f"averaged_eeg_{split}.npy"


def required_visual_feature_files(
    data: DataConfig,
    model: ModelConfig,
) -> tuple[Path, ...]:
    return tuple(
        workspace_path(data.visual_feature_dir) / f"image_{split}_layer{layer}.npy"
        for split in ("train", "test")
        for layer in model.visual.layer_ids
    )


def visual_feature_status(data: DataConfig, model: ModelConfig) -> dict[str, Any]:
    """Inspect extraction manifests and selected files without loading features."""
    files = required_visual_feature_files(data, model)
    incomplete: list[str] = []
    for split, expected_count in (("train", TRAIN_IMAGE_COUNT), ("test", TEST_IMAGE_COUNT)):
        progress = workspace_path(data.visual_feature_dir) / f"image_{split}_progress.json"
        if progress.is_file():
            try:
                state = json.loads(progress.read_text(encoding="utf-8"))
                if int(state.get("completed_images", -1)) != expected_count:
                    incomplete.append(
                        f"{progress}: {state.get('completed_images', 0)}/{expected_count} images"
                    )
            except (OSError, ValueError, TypeError) as exc:
                incomplete.append(f"{progress}: invalid progress file ({exc})")
        elif data.visual_feature_dir.name.startswith("vit_b16_laion2b_"):
            incomplete.append(f"{progress}: missing progress file")
    return {
        "ready": all(path.is_file() for path in files) and not incomplete,
        "present": [str(path) for path in files if path.is_file()],
        "missing": [str(path) for path in files if not path.is_file()],
        "incomplete": incomplete,
        "expected_train_shape": [1654, 10, model.visual.raw_feature_dim],
        "expected_test_shape": [200, 1, model.visual.raw_feature_dim],
    }


class VisualFeatureBank:
    """Read selected feature layers without loading the full dataset into RAM."""

    def __init__(self, data: DataConfig, model: ModelConfig, split: str) -> None:
        if split not in {"training", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        file_split = "train" if split == "training" else "test"
        expected_images = TRAIN_IMAGE_COUNT if split == "training" else TEST_IMAGE_COUNT
        progress = workspace_path(data.visual_feature_dir) / f"image_{file_split}_progress.json"
        if progress.is_file():
            state = json.loads(progress.read_text(encoding="utf-8"))
            if int(state.get("completed_images", -1)) != expected_images:
                raise ValueError(
                    f"Visual extraction is incomplete: {progress} reports "
                    f"{state.get('completed_images')} of {expected_images} images"
                )
            available_layers = state.get("layer_ids", [])
            if not set(model.visual.layer_ids).issubset(set(available_layers)):
                raise ValueError(
                    f"Selected layers {model.visual.layer_ids} are not all in {progress}"
                )
        elif data.visual_feature_dir.name.startswith("vit_b16_laion2b_"):
            raise FileNotFoundError(f"Missing extraction progress file: {progress}")
        arrays: list[np.ndarray] = []
        for layer in model.visual.layer_ids:
            path = workspace_path(data.visual_feature_dir) / f"image_{file_split}_layer{layer}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing SAMGA visual feature: {path}\n"
                    "Run the visual-feature extractor before training."
                )
            array = np.load(path, mmap_mode="r")
            if array.shape[-1] != model.visual.raw_feature_dim:
                raise ValueError(
                    f"{path} has feature dimension {array.shape[-1]}, expected "
                    f"{model.visual.raw_feature_dim}"
                )
            if int(np.prod(array.shape[:-1])) != expected_images:
                raise ValueError(
                    f"{path} contains {int(np.prod(array.shape[:-1]))} images, "
                    f"expected {expected_images}; shape={array.shape}"
                )
            arrays.append(array.reshape(expected_images, array.shape[-1]))
        self.arrays = tuple(arrays)
        self.image_count = expected_images

    def __getitem__(self, image_index: int) -> np.ndarray:
        return np.stack(
            [array[image_index] for array in self.arrays],
            axis=0,
        ).astype(np.float32, copy=False)


class SAMGADataset(Dataset[dict[str, Tensor]]):
    """Pair each averaged EEG recording with the matching image and subject ID.

    Training rows are concept-major: ten images per concept. Test rows contain
    one candidate image per concept. Files are opened read-only."""

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        subjects: Sequence[int],
        split: str,
        image_indices: Sequence[int] | None = None,
    ) -> None:
        self.subjects = tuple(int(subject) for subject in subjects)
        self.split = split
        if not self.subjects:
            raise ValueError("At least one subject is required")
        self.eeg_arrays = tuple(
            np.load(_eeg_path(data.dataset_root, subject, split), mmap_mode="r")
            for subject in self.subjects
        )
        expected = (
            TRAIN_IMAGE_COUNT if split == "training" else TEST_IMAGE_COUNT,
            model.eeg.channels,
            model.eeg.samples,
        )
        for subject, array in zip(self.subjects, self.eeg_arrays):
            if array.shape != expected:
                raise ValueError(
                    f"Subject {subject:02d} {split} EEG shape is {array.shape}, "
                    f"expected {expected}"
                )
        self.visual = VisualFeatureBank(data, model, split)
        self.image_indices = (
            tuple(range(expected[0])) if image_indices is None
            else tuple(int(index) for index in image_indices)
        )
        if (not self.image_indices or len(set(self.image_indices)) != len(self.image_indices)
                or any(index < 0 or index >= expected[0] for index in self.image_indices)):
            raise ValueError("Dataset image indices must be nonempty, unique, and in range")
        self.images_per_subject = len(self.image_indices)

    def __len__(self) -> int:
        return len(self.subjects) * self.images_per_subject

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        subject_offset, local_image_index = divmod(index, self.images_per_subject)
        image_index = self.image_indices[local_image_index]
        subject_id = self.subjects[subject_offset]
        eeg = np.array(self.eeg_arrays[subject_offset][image_index], copy=True)
        visual = np.array(self.visual[image_index], copy=True)
        class_index = image_index // 10 if self.split == "training" else image_index
        return {
            "eeg": torch.from_numpy(eeg),
            "visual": torch.from_numpy(visual),
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
            "image_index": torch.tensor(image_index, dtype=torch.long),
            "class_index": torch.tensor(class_index, dtype=torch.long),
        }


class CrossSubjectImageBatchSampler(BatchSampler):
    """Shuffle image IDs, then include every training subject for each image."""

    def __init__(
        self,
        image_count: int,
        subject_count: int,
        images_per_batch: int,
        *,
        seed: int,
    ) -> None:
        if image_count < 1 or subject_count < 1 or images_per_batch < 1:
            raise ValueError("Image, subject, and per-batch counts must be positive")
        self.image_count = image_count
        self.subject_count = subject_count
        self.images_per_batch = images_per_batch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.image_count, generator=generator).tolist()
        for start in range(0, self.image_count, self.images_per_batch):
            selected = order[start : start + self.images_per_batch]
            yield [
                subject_position * self.image_count + image_index
                for image_index in selected
                for subject_position in range(self.subject_count)
            ]

    def __len__(self) -> int:
        return math.ceil(self.image_count / self.images_per_batch)


@dataclass
class PreparedData:
    """Training, test, and optional labeled-support views for one held-out fold."""
    train_subjects: tuple[int, ...]
    train_dataset: SAMGADataset
    test_dataset: SAMGADataset
    train_subjects_test_dataset: SAMGADataset | None
    support_dataset: SAMGADataset | None
    support_image_ids: tuple[int, ...]


def prepare_data(config: RunConfig) -> PreparedData:
    """Create disjoint subject views; target support always uses training concepts."""
    held_out = config.data.held_out_subject
    if held_out not in ALL_SUBJECTS:
        raise ValueError(f"held_out_subject must be in {ALL_SUBJECTS}")
    train_subjects = tuple(subject for subject in ALL_SUBJECTS if subject != held_out)
    support_image_ids: tuple[int, ...] = ()
    support = None
    if config.score.enabled:
        support_image_ids = select_support_images(config.score)
        support = SAMGADataset(
            config.data, config.model, (held_out,), "training", support_image_ids
        )
    train = SAMGADataset(
        config.data, config.model, train_subjects, "training"
    )
    test = SAMGADataset(config.data, config.model, (held_out,), "test")
    train_subjects_test = None
    if config.data.evaluate_train_subjects_on_test:
        train_subjects_test = SAMGADataset(
            config.data,
            config.model,
            train_subjects,
            "test",
        )
    return PreparedData(
        train_subjects, train, test, train_subjects_test, support, support_image_ids
    )


def _smooth_channels(
    eeg: Tensor,
    kernel_size: int,
    probability: float,
    *,
    subject_ids: Tensor,
    image_indices: Tensor,
    seed: int,
    frozen: bool,
) -> Tensor:
    """Apply a moving average to selected channels without modifying source arrays."""
    if probability <= 0:
        return eeg
    smoothed = F.avg_pool1d(
        eeg,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    if frozen:


        # Hashing makes the channel mask stable across epochs without a second EEG copy.
        channel_ids = torch.arange(eeg.shape[1], device=eeg.device).view(1, -1)
        keys = (
            subject_ids.long().view(-1, 1) * 73_856_093
            + image_indices.long().view(-1, 1) * 19_349_663
            + channel_ids * 83_492_791
            + int(seed)
        )
        hashed = torch.remainder(keys * 1_103_515_245 + 12_345, 2_147_483_647)
        mask = (hashed.float() / 2_147_483_647.0).unsqueeze(-1) < probability
    else:
        mask = torch.rand(eeg.shape[0], eeg.shape[1], 1, device=eeg.device) < probability
    return torch.where(mask, smoothed, eeg)


def _make_loader(
    dataset: Dataset,
    data: DataConfig,
    optimization: OptimizationConfig,
    *,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or optimization.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory and torch.cuda.is_available(),
    )
