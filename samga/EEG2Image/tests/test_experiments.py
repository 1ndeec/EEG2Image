"""Check path-compatible resume discovery and optional-calibration summaries."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from things_eeg2.config import RunConfig, resolve_paths
from things_eeg2.experiments import resume_config_differences, run_loso
from things_eeg2.train import _jsonable


class ExperimentTests(unittest.TestCase):
    def test_relative_saved_paths_match_resolved_config(self):
        config = RunConfig()
        saved = _jsonable(asdict(config))
        self.assertEqual(resume_config_differences(saved, resolve_paths(config)), [])

    def test_resume_comparison_retains_loss_version_check(self):
        config = RunConfig()
        saved = _jsonable(asdict(config))
        saved["optimization"]["paired_loss_version"] = "different_loss"
        self.assertIn("optimization.paired_loss_version",
                      resume_config_differences(saved, config))

    def test_loso_summary_without_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(RunConfig(), checkpoint_root=root)

            def complete_fold(fold):
                folder = root / fold.run_name
                folder.mkdir()
                (folder / "history.json").write_text(json.dumps([{
                    "epoch": 1, "validation_test_r1": 0.25,
                    "validation_test_r5": 0.75, "validation_test_ic": 0.25,
                }]))
                return folder

            with patch("things_eeg2.experiments.run_training", side_effect=complete_fold):
                with redirect_stdout(StringIO()):
                    summary = run_loso(config, (1,), "test", 1, 0)
            self.assertEqual(summary["1"]["final_r1"], 0.25)
            self.assertEqual(summary["1"]["final_r5"], 0.75)
            self.assertTrue((root / "loso_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
