from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.eval_core.inference_benchmark import InferenceBenchmarkConfig, run_inference_benchmark
from src.eval_core.inference_engines import build_inference_engine


class _DummyEngine:
    def generate(self, prompt: str) -> str:
        return json.dumps({"ok": True, "prompt": prompt}, ensure_ascii=False)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(p) for p in prompts]


class InferenceBenchmarkSmokeTests(unittest.TestCase):
    def test_build_inference_engine_for_vllm_passes_runtime_kwargs(self) -> None:
        with patch("src.eval_core.inference_engines.VLLMInferenceEngine") as mocked_cls:
            mocked_cls.return_value = object()

            engine = build_inference_engine(
                {
                    "backend": "vllm",
                    "model_path": "dummy-model",
                    "tokenizer_path": "dummy-tokenizer",
                    "quantization": "4bit",
                    "max_new_tokens": 64,
                    "temperature": 0.1,
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.85,
                    "trust_remote_code": False,
                    "require_gpu": True,
                }
            )

            self.assertIs(engine, mocked_cls.return_value)
            mocked_cls.assert_called_once_with(
                model_path="dummy-model",
                max_new_tokens=64,
                temperature=0.1,
                trust_remote_code=False,
                tokenizer_path="dummy-tokenizer",
                require_gpu=True,
                quantization="4bit",
                max_model_len=8192,
                gpu_memory_utilization=0.85,
                dtype=None,
            )

    def test_smoke_single_and_batch_outputs_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            single_json = root / "single.json"
            single_cfg = InferenceBenchmarkConfig(
                backend="transformers",
                model_path="dummy-model",
                batch_size=1,
                num_samples=3,
                output_json=str(single_json),
            )
            single_report = run_inference_benchmark(single_cfg, engine=_DummyEngine())

            for field in (
                "backend",
                "quantization",
                "batch_size",
                "num_samples",
                "avg_latency",
                "p50_latency",
                "p95_latency",
                "throughput",
                "peak_memory",
                "errors",
            ):
                self.assertIn(field, single_report)

            self.assertEqual(single_report["backend"], "transformers")
            self.assertEqual(single_report["batch_size"], 1)
            self.assertEqual(single_report["num_samples"], 3)
            self.assertEqual(single_report["token_count_method"], "heuristic_estimate")
            self.assertIsNone(single_report["avg_ttft_sec"])
            self.assertEqual(single_report["ttft_observed_batches"], 0)
            self.assertTrue(single_json.exists())

            batch_json = root / "batch.json"
            batch_csv = root / "batch.csv"
            batch_cfg = InferenceBenchmarkConfig(
                backend="vllm",
                model_path="dummy-model",
                batch_size=2,
                num_samples=5,
                output_json=str(batch_json),
                output_csv=str(batch_csv),
            )
            batch_report = run_inference_benchmark(batch_cfg, engine=_DummyEngine())

            self.assertEqual(batch_report["backend"], "vllm")
            self.assertEqual(batch_report["batch_size"], 2)
            self.assertEqual(batch_report["num_samples"], 5)
            self.assertEqual(batch_report["errors"], 0)
            self.assertEqual(batch_report["successful_samples"], 5)
            self.assertEqual(batch_report["prompt_sampling_strategy"], "sequential_repeat")
            self.assertGreater(batch_report["avg_input_tokens"], 0.0)
            self.assertGreater(batch_report["avg_e2e_time_per_output_token_sec"], 0.0)
            self.assertTrue(batch_json.exists())
            self.assertTrue(batch_csv.exists())

            persisted = json.loads(batch_json.read_text(encoding="utf-8"))
            self.assertEqual(persisted["num_batches"], 3)


if __name__ == "__main__":
    unittest.main()
