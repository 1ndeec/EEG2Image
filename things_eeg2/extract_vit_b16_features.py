"""Cache class-token and mean-patch vectors from ViT-B/16 blocks 2-10.

Run python -m things_eeg2.extract_vit_b16_features. Image order comes from
image_metadata.npy; extraction reads image ZIPs and never modifies EEG."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


from .paths import WORKSPACE_ROOT

PROJECT_ROOT = WORKSPACE_ROOT


@dataclass(frozen=True)
class ExtractionConfig:
    """Checkpoint, output precision, pooling choices, and extraction batch size."""
    dataset_root: Path = PROJECT_ROOT / "training_data" / "nm000232"
    checkpoint: Path = (
        PROJECT_ROOT / "models" / "ViT-B-16-laion2b" / "open_clip_model.safetensors"
    )
    output_root: Path = PROJECT_ROOT / "training_data" / "nm000232" / "stimuli"
    layer_ids: tuple[int, ...] = tuple(range(2, 11))
    pooling_modes: tuple[str, ...] = ("cls", "patch_mean")
    batch_size: int = 32
    device: str = "auto"
    amp: bool = True
    output_dtype: str = "float16"


CONFIG = ExtractionConfig()


class ZippedImages:
    """Read stimulus images from ZIP files in the metadata-defined EEG row order."""
    def __init__(self, dataset_root: Path, split: str, preprocess) -> None:
        if split not in {"train", "test"}:
            raise ValueError(split)
        stimuli = dataset_root / "stimuli"
        metadata = np.load(stimuli / "image_metadata.npy", allow_pickle=True).item()
        concepts = metadata[f"{split}_img_concepts"]
        files = metadata[f"{split}_img_files"]
        if len(concepts) != len(files):
            raise ValueError(f"{split} metadata concepts/files lengths differ")
        self.members = [
            f"{'training' if split == 'train' else 'test'}_images/{concept}/{filename}"
            for concept, filename in zip(concepts, files)
        ]
        self.zip_path = stimuli / (
            "training_images.zip" if split == "train" else "test_images.zip"
        )
        self.preprocess = preprocess
        self._zip = zipfile.ZipFile(self.zip_path)
        zip_members = set(self._zip.namelist())
        missing = [member for member in self.members if member not in zip_members]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} metadata images missing from {self.zip_path}; "
                f"first missing: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, index: int):
        image_bytes = self._zip.read(self.members[index])
        with Image.open(BytesIO(image_bytes)) as image:
            return self.preprocess(image.convert("RGB"))

    def close(self) -> None:
        self._zip.close()


def _output_directory(config: ExtractionConfig, mode: str) -> Path:
    return config.output_root / f"vit_b16_laion2b_layers_2_10_{mode}"


def _state_path(output_dir: Path, split: str) -> Path:
    return output_dir / f"image_{split}_progress.json"


def _save_state(path: Path, state: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _open_outputs(config: ExtractionConfig, split: str, image_count: int):
    if split == "train" and image_count != 16_540:
        raise ValueError(f"Expected 16,540 train images, found {image_count}")
    if split == "test" and image_count != 200:
        raise ValueError(f"Expected 200 test images, found {image_count}")

    shape = (1654, 10, 768) if split == "train" else (200, 1, 768)
    dtype = np.dtype(config.output_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("output_dtype must be float16 or float32")
    arrays = {}
    offsets = []
    for mode in config.pooling_modes:
        output_dir = _output_directory(config, mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        state_file = _state_path(output_dir, split)
        expected_state = {
            "model": "ViT-B-16",
            "checkpoint": str(config.checkpoint.resolve()),
            "split": split,
            "mode": mode,
            "layer_ids": list(config.layer_ids),
            "shape": list(shape),
            "dtype": str(dtype),
        }
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            for key, value in expected_state.items():
                if state.get(key) != value:
                    raise ValueError(
                        f"Existing {state_file} has different {key}. "
                        "Use another output directory for a different extraction."
                    )
            offset = int(state["completed_images"])
            if not 0 <= offset <= image_count:
                raise ValueError(f"Invalid completed_images in {state_file}")
            file_mode = "r+"
        else:
            offset = 0
            file_mode = "w+"
            if any(
                (output_dir / f"image_{split}_layer{layer}.npy").exists()
                for layer in config.layer_ids
            ):
                raise FileExistsError(
                    f"{output_dir} contains layer files without {state_file}; "
                    "move those files before starting a new extraction"
                )
        offsets.append(offset)
        for layer in config.layer_ids:
            path = output_dir / f"image_{split}_layer{layer}.npy"
            array = np.lib.format.open_memmap(
                path, mode=file_mode, dtype=dtype, shape=shape
            )
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Unexpected shape/dtype in {path}: {array.shape}/{array.dtype}")
            arrays[(mode, layer)] = array.reshape(image_count, 768)
        if file_mode == "w+":
            _save_state(state_file, {**expected_state, "completed_images": 0})


    return arrays, min(offsets)


def _choose_device(config: ExtractionConfig, torch):
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if config.device == "auto" else config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(device)


def extract(config: ExtractionConfig = CONFIG) -> None:
    """Extract or resume feature caches and update their completion manifests."""
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install torch, torchvision, open_clip_torch, numpy, and pillow "
            "in the selected IDE environment before running this script"
        ) from exc

    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint missing: {config.checkpoint}")
    if config.layer_ids != tuple(range(2, 11)):
        raise ValueError("This extractor is intended to store blocks 2 through 10")
    if not config.pooling_modes or any(
        mode not in {"cls", "patch_mean"} for mode in config.pooling_modes
    ):
        raise ValueError("pooling_modes must contain cls and/or patch_mean")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = _choose_device(config, torch)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(config.checkpoint), device="cpu"
    )
    visual = model.visual.to(device).eval()
    del model
    if len(visual.transformer.resblocks) != 12 or visual.class_embedding.numel() != 768:
        raise ValueError("Checkpoint did not create a 12-block, 768-wide ViT-B/16")
    if not hasattr(visual, "forward_intermediates"):
        raise RuntimeError("This OpenCLIP version lacks forward_intermediates; update open_clip_torch")
    print(f"Device: {device}; checkpoint: {config.checkpoint}")
    print("Image preprocessing comes from OpenCLIP's ViT-B-16 evaluation transform")

    for split in ("train", "test"):
        images = ZippedImages(config.dataset_root, split, preprocess)
        try:
            arrays, start = _open_outputs(config, split, len(images))
            print(f"{split}: {len(images):,} images; resuming at {start:,}", flush=True)
            for begin in range(start, len(images), config.batch_size):
                end = min(begin + config.batch_size, len(images))
                batch = torch.stack([images[i] for i in range(begin, end)]).to(device)
                with torch.inference_mode():
                    with torch.autocast(
                        device_type=device.type,
                        enabled=config.amp and device.type == "cuda",
                    ):
                        result = visual.forward_intermediates(
                            batch,
                            indices=[layer - 1 for layer in config.layer_ids],
                            stop_early=True,
                            intermediates_only=True,
                            output_fmt="NLC",
                            output_extra_tokens=True,
                        )
                prefixes = result["image_intermediates_prefix"]
                patches = result["image_intermediates"]
                if len(prefixes) != len(config.layer_ids) or len(patches) != len(config.layer_ids):
                    raise RuntimeError("OpenCLIP returned an unexpected number of layers")
                for layer_index, layer in enumerate(config.layer_ids):
                    vectors = {
                        "cls": prefixes[layer_index][:, 0],
                        "patch_mean": patches[layer_index].mean(dim=1),
                    }
                    for mode in config.pooling_modes:
                        values = vectors[mode].float().cpu().numpy()
                        if values.shape != (end - begin, 768) or not np.isfinite(values).all():
                            raise RuntimeError(
                                f"Invalid {mode} layer {layer} vectors at images {begin}:{end}"
                            )
                        arrays[(mode, layer)][begin:end] = values
                for array in arrays.values():
                    array.flush()
                for mode in config.pooling_modes:
                    state_file = _state_path(_output_directory(config, mode), split)
                    state = json.loads(state_file.read_text(encoding="utf-8"))
                    state["completed_images"] = end
                    _save_state(state_file, state)
                if end == len(images) or end % max(config.batch_size * 20, 1) == 0:
                    print(f"{split}: {end:,}/{len(images):,}", flush=True)
                del batch, result
            print(f"{split}: complete", flush=True)
        finally:
            images.close()


if __name__ == "__main__":
    extract()
