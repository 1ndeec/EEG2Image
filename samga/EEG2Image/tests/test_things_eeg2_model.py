from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from things_eeg2.model import ModelConfig, SAMGAModel
from things_eeg2.train import mmd_loss, symmetric_contrastive_loss


class SAMGAModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig()
        self.model = SAMGAModel(self.config)

    def test_eeg_and_visual_shapes(self) -> None:
        self.model.eval()
        eeg = torch.randn(3, 63, 250)
        visual = torch.randn(3, 5, 3200)
        subjects = torch.tensor([1, 2, 3])
        with torch.no_grad():
            eeg_embedding = self.model.encode_eeg(eeg)
            visual_embedding, weights = self.model.encode_visual(visual, subjects)
        self.assertEqual(tuple(eeg_embedding.shape), (3, 512))
        self.assertEqual(tuple(visual_embedding.shape), (3, 512))
        self.assertEqual(tuple(weights.shape), (3, 5))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))

    def test_tsconv_released_shape(self) -> None:
        self.assertEqual(self.model.eeg_encoder.output_time_points, 36)
        self.assertEqual(self.model.eeg_encoder.flattened_dim, 1440)

    def test_router_prior_is_centered_on_layer_28(self) -> None:
        expected_logits = torch.tensor([-2.0, -1.0, 0.0, -1.0, -2.0])
        torch.testing.assert_close(
            self.model.visual_target.router.global_logits.detach(),
            expected_logits,
        )

    def test_global_inference_ignores_subject_residuals(self) -> None:
        router = self.model.visual_target.router
        with torch.no_grad():
            router.subject_residuals.weight.normal_()
        self.model.eval()
        weights = router(torch.tensor([1, 9]), force_global=True)
        torch.testing.assert_close(weights[0], weights[1])

    def test_losses_are_finite(self) -> None:
        eeg = torch.randn(8, 512, requires_grad=True)
        visual = torch.randn(8, 512, requires_grad=True)
        retrieval = symmetric_contrastive_loss(eeg, visual, 0.07)
        mmd = mmd_loss(eeg, visual, (0.1, 0.5, 1.0))
        self.assertTrue(torch.isfinite(retrieval))
        self.assertTrue(torch.isfinite(mmd))
        (retrieval + mmd).backward()
        self.assertIsNotNone(eeg.grad)
        self.assertIsNotNone(visual.grad)

    def test_samga_retrieval_uses_full_similarity_matrix(self) -> None:
        # Rows 0 and 1 can represent two subjects viewing the same image.
        # The released SAMGA loss still keeps both off-diagonal scores.
        eeg = torch.tensor([[1., 0., 0.], [0.8, 0.6, 0.], [0., 0., 1.]])
        visual = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        logits = F.normalize(eeg, dim=-1) @ F.normalize(visual, dim=-1).T / 0.07
        labels = torch.arange(3)
        expected = (F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)) / 2
        torch.testing.assert_close(symmetric_contrastive_loss(eeg, visual, 0.07), expected)


if __name__ == "__main__":
    unittest.main()
