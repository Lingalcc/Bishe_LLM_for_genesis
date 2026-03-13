from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_core.dataset_safety import enforce_train_eval_no_leakage
from src.data_core.split_dataset import SplitDatasetConfig, run_split_dataset


def _write_dataset(path: Path, n: int) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "instruction": f"instruction-{i}",
                "input": "",
                "output": f'{{"action":"wait","parameters":{{"steps":{i}}}}}',
            }
        )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


class DatasetSplitAndLeakageTests(unittest.TestCase):
    def test_ratio_split_correct(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "dataset.json"
            out = root / "splits"
            _write_dataset(src, 10)

            meta = run_split_dataset(
                SplitDatasetConfig(
                    input_file=src,
                    out_dir=out,
                    train_ratio=0.6,
                    val_ratio=0.2,
                    test_ratio=0.2,
                    seed=7,
                )
            )
            self.assertEqual(meta["splits"]["train"]["num_samples"], 6)
            self.assertEqual(meta["splits"]["val"]["num_samples"], 2)
            self.assertEqual(meta["splits"]["test"]["num_samples"], 2)

    def test_seed_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "dataset.json"
            _write_dataset(src, 20)

            meta_a = run_split_dataset(
                SplitDatasetConfig(input_file=src, out_dir=root / "split_a", seed=123)
            )
            meta_b = run_split_dataset(
                SplitDatasetConfig(input_file=src, out_dir=root / "split_b", seed=123)
            )
            self.assertEqual(
                meta_a["splits"]["train"]["sample_set_sha256"],
                meta_b["splits"]["train"]["sample_set_sha256"],
            )
            self.assertEqual(
                meta_a["splits"]["val"]["sample_set_sha256"],
                meta_b["splits"]["val"]["sample_set_sha256"],
            )
            self.assertEqual(
                meta_a["splits"]["test"]["sample_set_sha256"],
                meta_b["splits"]["test"]["sample_set_sha256"],
            )

    def test_train_test_same_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.json"
            val = root / "val.json"
            _write_dataset(train, 5)
            _write_dataset(val, 3)

            with self.assertRaises(ValueError):
                enforce_train_eval_no_leakage(
                    train_file=train,
                    val_file=val,
                    test_file=train,
                    strict=True,
                )

    def test_metadata_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "dataset.json"
            out = root / "splits"
            _write_dataset(src, 8)

            meta = run_split_dataset(
                SplitDatasetConfig(input_file=src, out_dir=out, seed=1)
            )
            self.assertIn("created_at_utc", meta)
            self.assertIn("seed", meta)
            self.assertIn("source", meta)
            self.assertIn("splits", meta)
            self.assertIn("overlap", meta)
            for key in ("train", "val", "test"):
                self.assertIn("path", meta["splits"][key])
                self.assertIn("num_samples", meta["splits"][key])
                self.assertIn("sha256", meta["splits"][key])
                self.assertIn("sample_set_sha256", meta["splits"][key])


if __name__ == "__main__":
    unittest.main()
