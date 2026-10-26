from __future__ import annotations

import unittest

import torch

from things_eeg2.score_calibration import (
    ScoreCalibrationConfig,
    _apply_map,
    _fit_orthogonal_map,
    _match_moments,
    evaluate_chained_score_calibration,
)


class ChainedScoreCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.support = torch.randn(12, 4)
        self.images = torch.randn(12, 4)
        self.queries = torch.randn(4, 4)
        self.gallery = torch.randn(4, 4)
        self.targets = torch.arange(4)
        self.ids = tuple(range(12))
        self.config = ScoreCalibrationConfig(
            enabled=True, pool_class_count=12, landmark_count=4,
            seed=42, regularization_ratio=0.1, csls_neighbors=2,
        )

    def test_second_map_fits_transformed_support_with_its_own_rho(self) -> None:
        metrics, artifacts = evaluate_chained_score_calibration(
            self.support, self.images, self.queries, self.gallery,
            self.targets, self.ids, self.config,
            (("random", 4, 0.25), ("csls_selected", 4, 0.75)),
        )
        self.assertEqual(len(metrics), 8)
        self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))
        first, second = artifacts["stages"]
        self.assertEqual((first["rho"], second["rho"]), (0.25, 0.75))

        matched = _match_moments(self.support.double(), artifacts["statistics"])
        after_first = _apply_map(matched, first["map"])
        rows = second["landmark_image_ids"]
        expected = _fit_orthogonal_map(
            after_first[rows], self.images.double()[rows], 0.75,
        )
        for name in ("rotation", "eeg_center", "image_center"):
            torch.testing.assert_close(second["map"][name], expected[name])

    def test_fitting_does_not_depend_on_test_queries(self) -> None:
        arguments = (
            self.support, self.images, self.queries, self.gallery,
            self.targets, self.ids, self.config,
            (("random", 4, 0.25), ("csls_selected", 4, 0.75)),
        )
        _, original = evaluate_chained_score_calibration(*arguments)
        modified = list(arguments)
        modified[2] = self.queries * -3
        _, changed = evaluate_chained_score_calibration(*modified)
        for first, second in zip(original["stages"], changed["stages"]):
            self.assertEqual(first["landmark_image_ids"], second["landmark_image_ids"])
            torch.testing.assert_close(first["map"]["rotation"], second["map"]["rotation"])

    def test_stages_can_reuse_support_images(self) -> None:
        _, artifacts = evaluate_chained_score_calibration(
            self.support, self.images, self.queries, self.gallery,
            self.targets, self.ids, self.config, (("random", 8), ("random", 8)),
        )
        first, second = artifacts["stages"]
        self.assertEqual(len(set(first["landmark_image_ids"])), 8)
        self.assertEqual(len(set(second["landmark_image_ids"])), 8)
        self.assertTrue(set(first["landmark_image_ids"]) & set(second["landmark_image_ids"]))

    def test_rejects_more_landmarks_than_pool_in_one_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "support pool"):
            evaluate_chained_score_calibration(
                self.support, self.images, self.queries, self.gallery,
                self.targets, self.ids, self.config, (("random", 13),),
            )

    def test_rejects_negative_stage_rho(self) -> None:
        with self.assertRaisesRegex(ValueError, "rho"):
            evaluate_chained_score_calibration(
                self.support, self.images, self.queries, self.gallery,
                self.targets, self.ids, self.config, (("random", 4, -0.1),),
            )


if __name__ == "__main__":
    unittest.main()
