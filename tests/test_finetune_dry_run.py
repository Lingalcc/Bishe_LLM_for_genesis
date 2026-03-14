from __future__ import annotations

import unittest

from src.finetune_core.train import run_finetune_from_merged_config


class FinetuneDryRunTests(unittest.TestCase):
    def test_dry_run_skips_missing_split_files(self) -> None:
        cfg = {
            "finetune": {
                "train": {
                    "llamafactory_dir": "LlamaFactory",
                    "train_file": "tmp/nonexistent_train_for_test.json",
                    "val_file": "tmp/nonexistent_val_for_test.json",
                    "dry_run": True,
                    "finetune_method": "qlora",
                }
            }
        }

        result = run_finetune_from_merged_config(cfg)
        self.assertFalse(result["executed"])
        self.assertEqual(result["dataset_overrides"], [])

    def test_train_mode_prechecks_split_files(self) -> None:
        cfg = {
            "finetune": {
                "train": {
                    "llamafactory_dir": "LlamaFactory",
                    "train_file": "tmp/nonexistent_train_for_test.json",
                    "val_file": "tmp/nonexistent_val_for_test.json",
                    "dry_run": False,
                    "finetune_method": "qlora",
                }
            }
        }

        with self.assertRaises(FileNotFoundError) as ctx:
            run_finetune_from_merged_config(cfg)
        message = str(ctx.exception)
        self.assertIn("Missing required split files before finetune", message)
        self.assertIn("python cli.py data split", message)


if __name__ == "__main__":
    unittest.main()
