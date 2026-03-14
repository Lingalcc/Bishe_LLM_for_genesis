from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_core.split_dataset import run_split_from_merged_config


class SplitFromMergedConfigTests(unittest.TestCase):
    def test_split_uses_config_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "dataset.json"
            rows = [{"instruction": f"i-{i}", "output": '{"commands":[{"action":"wait"}]}'} for i in range(10)]
            src.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

            merged = {
                "dataset_prepare": {
                    "split": {
                        "input_file": str(src),
                        "out_dir": str(root / "splits"),
                        "train_ratio": 0.6,
                        "val_ratio": 0.2,
                        "test_ratio": 0.2,
                        "seed": 7,
                    }
                }
            }

            meta = run_split_from_merged_config(merged)
            self.assertEqual(meta["splits"]["train"]["num_samples"], 6)
            self.assertEqual(meta["splits"]["val"]["num_samples"], 2)
            self.assertEqual(meta["splits"]["test"]["num_samples"], 2)
            self.assertTrue(Path(meta["metadata_file"]).exists())


if __name__ == "__main__":
    unittest.main()
