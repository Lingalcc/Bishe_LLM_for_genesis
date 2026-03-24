from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.finetune_core.train import run_finetune_from_merged_config


class FinetuneDryRunTests(unittest.TestCase):
    def _write_temp_train_config(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            tmp.write(content)
            path = Path(tmp.name)
            self.addCleanup(lambda: path.unlink(missing_ok=True))
            return path
        finally:
            tmp.close()

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

    def test_explicit_lora_config_overrides_base_default_qlora_method(self) -> None:
        config_path = self._write_temp_train_config(
            "\n".join(
                [
                    "model_name_or_path: model/Qwen_Qwen2.5-3B-Instruct",
                    "stage: sft",
                    "do_train: true",
                    "finetuning_type: lora",
                    "lora_rank: 8",
                    "",
                ]
            )
        )

        cfg = {
            "finetune": {
                "train": {
                    "llamafactory_dir": "LlamaFactory",
                    "config": str(config_path),
                    "dry_run": True,
                }
            }
        }

        result = run_finetune_from_merged_config(cfg)
        self.assertEqual(result["method"], "lora")

    def test_adalora_config_fails_fast_with_clear_error(self) -> None:
        config_path = self._write_temp_train_config(
            "\n".join(
                [
                    "model_name_or_path: model/Qwen_Qwen2.5-3B-Instruct",
                    "stage: sft",
                    "do_train: true",
                    "finetuning_type: lora",
                    "peft_type: adalora",
                    "adalora_init_r: 12",
                    "",
                ]
            )
        )

        cfg = {
            "finetune": {
                "train": {
                    "llamafactory_dir": "LlamaFactory",
                    "config": str(config_path),
                    "dry_run": True,
                }
            }
        }

        with self.assertRaises(ValueError) as ctx:
            run_finetune_from_merged_config(cfg)
        self.assertIn("AdaLoRA", str(ctx.exception))
        self.assertIn("LlamaFactory", str(ctx.exception))

    @mock.patch("torch.cuda.device_count", return_value=0)
    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_train_mode_fails_fast_when_cuda_is_unavailable(self, *_mocks: object) -> None:
        cfg = {
            "finetune": {
                "train": {
                    "llamafactory_dir": "LlamaFactory",
                    "train_file": "data_prepare/splits/train.json",
                    "val_file": "data_prepare/splits/val.json",
                    "dry_run": False,
                    "gpus": "0",
                    "finetune_method": "lora",
                }
            }
        }

        with self.assertRaises(RuntimeError) as ctx:
            run_finetune_from_merged_config(cfg)
        self.assertIn("CUDA", str(ctx.exception))
        self.assertIn("torch.cuda.is_available()", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
