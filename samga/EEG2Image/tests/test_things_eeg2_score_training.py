from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from things_eeg2.score_training import (
    fit_supervised_support_map,
    multi_positive_loss,
    support_to_query_recovery_loss,
)
from things_eeg2.train import _select_training_recovery_landmarks


class ScoreTrainingLossTests(unittest.TestCase):
    def test_multi_positive_matches_symmetric_paper_equation(self) -> None:
        torch.manual_seed(21)
        eeg = torch.randn(6, 12, requires_grad=True)
        visual = torch.randn(6, 12, requires_grad=True)
        images = torch.tensor([4, 4, 4, 7, 7, 7])
        actual = multi_positive_loss(eeg, visual, images, 0.07)

        logits = F.normalize(eeg, dim=-1) @ F.normalize(visual, dim=-1).T / 0.07
        positives = images[:, None].eq(images[None, :]).float() / 3.0
        expected = (
            -(positives * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            -(positives.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(torch.isfinite(visual.grad).all())

    def test_support_recovery_stops_at_fit_but_trains_query_encoders(self) -> None:
        torch.manual_seed(22)
        support_eeg = torch.randn(8, 16, requires_grad=True)
        support_visual = torch.randn(8, 16, requires_grad=True)
        fitted = fit_supervised_support_map(
            support_eeg, support_visual, [0, 2, 4, 6], identity_ratio=0.1
        )
        self.assertFalse(any(value.requires_grad for value in fitted.values()))
        eeg = torch.randn(4, 16, requires_grad=True)
        visual = torch.randn(4, 16, requires_grad=True)
        loss = support_to_query_recovery_loss(eeg, visual, fitted, temperature=0.07)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(support_eeg.grad)
        self.assertIsNone(support_visual.grad)
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(torch.isfinite(visual.grad).all())
        self.assertGreater(float(eeg.grad.norm()), 0.0)
        self.assertGreater(float(visual.grad.norm()), 0.0)

    def test_training_landmarks_reselect_each_epoch_but_fixed_ablation_does_not(self) -> None:
        support = torch.zeros(2, 3, 4)
        visual = torch.zeros(3, 4)
        image_ids = (10, 20, 30)
        selections = [
            torch.tensor([[0, 1], [1, 2]]),
            torch.tensor([[1, 2], [0, 2]]),
        ]
        with patch("things_eeg2.train.select_subject_csls_landmarks", side_effect=selections) as choose:
            first = _select_training_recovery_landmarks(
                support, visual, (1, 2), image_ids, 2, 10, 1e-6,
                "per_epoch_v1", None,
            )
            second = _select_training_recovery_landmarks(
                support, visual, (1, 2), image_ids, 2, 10, 1e-6,
                "per_epoch_v1", first,
            )
            self.assertEqual(choose.call_count, 2)
            self.assertEqual(first, {1: (10, 20), 2: (20, 30)})
            self.assertEqual(second, {1: (20, 30), 2: (10, 30)})
        with patch("things_eeg2.train.select_subject_csls_landmarks") as choose:
            fixed = _select_training_recovery_landmarks(
                support, visual, (1, 2), image_ids, 2, 10, 1e-6,
                "fixed_v1", first,
            )
            self.assertIs(fixed, first)
            choose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
