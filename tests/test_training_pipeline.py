"""Exercise stage transitions and resume using a small, fully synthetic dataset."""

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from things_eeg2 import data, train
from things_eeg2.config import DataConfig, OptimizationConfig, RunConfig
from things_eeg2.model import EEGEncoderConfig, ModelConfig, VisualRouterConfig
from things_eeg2.score_calibration import ScoreCalibrationConfig
from things_eeg2.visual_lora import VisualLoRAConfig


class ToyVisualFeatures:
    """Provide differentiable image features without loading pretrained weights."""

    def __init__(self, config, dataset_root, layer_ids, device):
        self.config = config
        self.lora_blocks = config.blocks
        self.visual = nn.Linear(1, 1, bias=False).to(device)
        nn.init.zeros_(self.visual.weight)
        folder = dataset_root / "stimuli/vit_b16_laion2b_layers_2_10_cls"
        self.features = {
            split: torch.from_numpy(np.stack([
                np.load(folder / f"image_{split}_layer{layer}.npy")
                for layer in layer_ids
            ], axis=1)).to(device)
            for split in ("train", "test")
        }

    def trainable_parameters(self):
        return list(self.visual.parameters())

    def encode(self, indices, split):
        return self.features[split][indices] + self.visual.weight.reshape(())

    def test_feature_bank(self, *, amp_enabled):
        return self.encode(range(3), "test").detach()

    def support_feature_bank(self, indices, *, amp_enabled):
        return self.encode(indices, "train").detach()

    def adapter_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.visual.state_dict().items()}

    def load_adapter_state_dict(self, state):
        self.visual.load_state_dict(state)

    def close(self):
        pass


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(self.remove_fixture)
        rng = np.random.default_rng(11)
        for subject in (1, 2, 3):
            folder = self.root / "derivatives/nice_250hz" / f"sub-{subject:02d}"
            folder.mkdir(parents=True)
            for split, count in (("training", 6), ("test", 3)):
                np.save(folder / f"averaged_eeg_{split}.npy",
                        rng.normal(size=(count, 2, 40)).astype("float32"))
        features = self.root / "stimuli/vit_b16_laion2b_layers_2_10_cls"
        features.mkdir(parents=True)
        for split, count in (("train", 6), ("test", 3)):
            for layer in (1, 2):
                np.save(features / f"image_{split}_layer{layer}.npy",
                        rng.normal(size=(count, 768)).astype("float32"))
            (features / f"image_{split}_progress.json").write_text(json.dumps({
                "completed_images": count, "layer_ids": [1, 2], "shape": [count, 768],
            }))
        self.config = RunConfig(
            data=DataConfig(dataset_root=self.root, visual_feature_dir=features,
                            held_out_subject=3, images_per_batch=6),
            model=ModelConfig(
                eeg=EEGEncoderConfig(channels=2, samples=40, temporal_filters=2,
                                     temporal_kernel=5, pooling_kernel=3,
                                     pooling_stride=2, feature_dim=10),
                visual=VisualRouterConfig(layer_ids=(1, 2), raw_feature_dim=768,
                                          intermediate_dim=10, alignment_dim=6,
                                          prior_center=1, max_subject_id=3)),
            optimization=OptimizationConfig(epochs=2, coarse_epochs=1, use_amp=False,
                                            smooth_kernel_size=3),
            score=ScoreCalibrationConfig(enabled=True, pool_class_count=3,
                                         landmark_count=2, csls_neighbors=2),
            checkpoint_root=self.root / "checkpoints",
        )

    def remove_fixture(self):
        import gc
        import shutil
        gc.collect()
        shutil.rmtree(self.root, ignore_errors=True)

    def run_fixture(self, config):
        with ExitStack() as stack:
            for module in (data, train):
                for name, value in (("TRAIN_IMAGE_COUNT", 6), ("TEST_IMAGE_COUNT", 3),
                                    ("ALL_SUBJECTS", (1, 2, 3))):
                    stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(data, "select_support_images", return_value=(0, 2, 4)))
            stack.enter_context(patch.object(train, "LiveViTFeatures", ToyVisualFeatures))
            with redirect_stdout(StringIO()):
                output = train.run_training(config)
        return output, torch.load(output / "last.pt", map_location="cpu", weights_only=False)

    def test_each_visual_stage_keeps_adapters_in_the_checkpoint(self):
        for stage in ("coarse", "fine", "both"):
            with self.subTest(stage=stage):
                config = replace(self.config, run_name=stage, visual_lora=VisualLoRAConfig(
                    enabled=True, require_cuda=False, stage=stage, blocks=(1,),
                    images_per_batch=6, gradient_checkpointing=False,
                ))
                _, payload = self.run_fixture(config)
                self.assertEqual(payload["epoch"], 2)
                self.assertIn("weight", payload["visual_lora_state_dict"])
                self.assertEqual(payload["history"][1]["mmd_weight"], 0.0)
                self.assertTrue(torch.isfinite(payload["visual_lora_state_dict"]["weight"]).all())

    def test_coarse_boundary_resume_initializes_fine_only_adapters(self):
        coarse = replace(self.config, run_name="coarse", optimization=replace(
            self.config.optimization, epochs=1))
        folder, _ = self.run_fixture(coarse)
        fine = replace(self.config, run_name="continued", resume_checkpoint=folder / "last.pt",
                       visual_lora=VisualLoRAConfig(enabled=True, require_cuda=False,
                                                   blocks=(1,), images_per_batch=6))
        _, payload = self.run_fixture(fine)
        self.assertEqual([row["epoch"] for row in payload["history"]], [1, 2])
        self.assertIn("visual_lora_state_dict", payload)

    def test_missing_fine_adapter_is_rejected(self):
        config = replace(self.config, visual_lora=VisualLoRAConfig(
            enabled=True, require_cuda=False, blocks=(1,), images_per_batch=6))
        folder, payload = self.run_fixture(config)
        payload.pop("visual_lora_state_dict")
        invalid = folder / "missing_adapter.pt"
        torch.save(payload, invalid)
        with self.assertRaisesRegex(ValueError, "no visual adapter weights"):
            self.run_fixture(replace(config, resume_checkpoint=invalid))


if __name__ == "__main__":
    unittest.main()
