from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from src.data_core.generate import (
    DatasetGenerator,
    GenerateDatasetConfig,
    _build_runtime_paths,
    _sample_fingerprint,
)


class _StubGenerator(DatasetGenerator):
    def __init__(self, cfg: GenerateDatasetConfig, scripted_batches: list[list[dict]]) -> None:
        super().__init__(cfg)
        self._scripted_batches = scripted_batches
        self._cursor = 0

    def _prepare_batch_args(self, count: int) -> list[tuple[str, int, bool, int]]:
        return [("simple", count, False, 123)]

    def _generate_batch_from_args(
        self,
        difficulty: str,
        batch_size: int,
        with_state: bool,
        seed_hint: int,
        progress_callback=None,
        batch_label: str | None = None,
    ) -> tuple[str, list[dict]]:
        if self._cursor >= len(self._scripted_batches):
            return difficulty, []
        out = self._scripted_batches[self._cursor]
        self._cursor += 1
        return difficulty, out


class GenerateDedupTests(unittest.TestCase):
    def test_generate_all_deduplicates_and_refills(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "dummy-key-for-test"
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GenerateDatasetConfig(
                out_dir=Path(tmpdir),
                num_samples=3,
                max_workers=1,
                dedup_max_rounds=4,
                api_key_env="DEEPSEEK_API_KEY",
                api_key="",
            )
            a = {"instruction": "i1", "output": '{"commands":[{"action":"wait"}]}'}
            b = {"instruction": "i2", "output": '{"commands":[{"action":"wait"}]}'}
            c = {"instruction": "i3", "output": '{"commands":[{"action":"wait"}]}'}
            gen = _StubGenerator(
                cfg,
                scripted_batches=[
                    [a, a, b],
                    [b, c],
                ],
            )

            result = gen.generate_all()
            samples = result["samples"]
            stats = result["stats"]

            self.assertEqual(len(samples), 3)
            self.assertEqual(len({_sample_fingerprint(s) for s in samples}), 3)
            self.assertEqual(stats["duplicate_discarded"], 2)
            self.assertEqual(stats["dedup_rounds"], 2)
            self.assertTrue(stats["target_reached"])

    def test_generate_all_reports_progress(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "dummy-key-for-test"
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GenerateDatasetConfig(
                out_dir=Path(tmpdir),
                num_samples=3,
                batch_size=2,
                max_workers=1,
                dedup_max_rounds=3,
                api_key_env="DEEPSEEK_API_KEY",
                api_key="",
            )
            a = {"instruction": "i1", "output": '{"commands":[{"action":"wait"}]}'}
            b = {"instruction": "i2", "output": '{"commands":[{"action":"wait"}]}'}
            c = {"instruction": "i3", "output": '{"commands":[{"action":"wait"}]}'}
            updates: list[dict] = []
            gen = _StubGenerator(
                cfg,
                scripted_batches=[
                    [a, a],
                    [b],
                    [c],
                ],
            )

            result = gen.generate_all(progress_callback=updates.append)

            self.assertEqual(len(result["samples"]), 3)
            self.assertEqual(len(updates), 3)
            self.assertEqual(updates[0]["accepted_count"], 1)
            self.assertEqual(updates[0]["duplicate_count"], 1)
            self.assertEqual(updates[0]["unique_samples"], 1)
            self.assertEqual(updates[-1]["unique_samples"], 3)
            self.assertEqual(updates[-1]["remaining_samples"], 0)

    def test_generate_all_persists_after_each_batch(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "dummy-key-for-test"
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GenerateDatasetConfig(
                out_dir=Path(tmpdir),
                num_samples=3,
                batch_size=2,
                max_workers=1,
                api_key_env="DEEPSEEK_API_KEY",
                api_key="",
            )
            a = {"instruction": "i1", "output": '{"commands":[{"action":"wait"}]}'}
            b = {"instruction": "i2", "output": '{"commands":[{"action":"wait"}]}'}
            c = {"instruction": "i3", "output": '{"commands":[{"action":"wait"}]}'}
            gen = _StubGenerator(cfg, scripted_batches=[[a, b], [c]])
            runtime_paths = _build_runtime_paths(cfg)
            snapshots: list[tuple[int, int, int]] = []

            def _on_progress(update: dict) -> None:
                progress_payload = json.loads(runtime_paths["progress"].read_text(encoding="utf-8"))
                stats_payload = json.loads(runtime_paths["stats"].read_text(encoding="utf-8"))
                alpaca_payload = json.loads(runtime_paths["alpaca"].read_text(encoding="utf-8"))
                snapshots.append(
                    (
                        len(progress_payload["samples"]),
                        stats_payload["total_samples"],
                        len(alpaca_payload),
                    )
                )
                self.assertEqual(stats_payload["total_samples"], int(update["unique_samples"]))

            result = gen.generate_all(progress_callback=_on_progress, runtime_paths=runtime_paths)

            self.assertEqual(len(result["samples"]), 3)
            self.assertEqual(snapshots, [(2, 2, 2), (3, 3, 3)])

    def test_generate_all_resumes_from_progress_file(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "dummy-key-for-test"
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GenerateDatasetConfig(
                out_dir=Path(tmpdir),
                num_samples=3,
                batch_size=2,
                max_workers=1,
                api_key_env="DEEPSEEK_API_KEY",
                api_key="",
            )
            a = {"instruction": "i1", "output": '{"commands":[{"action":"wait"}]}'}
            b = {"instruction": "i2", "output": '{"commands":[{"action":"wait"}]}'}
            c = {"instruction": "i3", "output": '{"commands":[{"action":"wait"}]}'}
            runtime_paths = _build_runtime_paths(cfg)
            progress_payload = {
                "samples": [a, b],
                "stats": {
                    "total_samples": 2,
                    "target_samples": 3,
                    "api_calls": 1,
                    "invalid_discarded": 0,
                    "duplicate_discarded": 0,
                    "dedup_rounds": 1,
                    "difficulty_distribution": {"simple": 2, "medium": 0, "complex": 0},
                },
            }
            runtime_paths["out_dir"].mkdir(parents=True, exist_ok=True)
            runtime_paths["progress"].write_text(
                json.dumps(progress_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            gen = _StubGenerator(cfg, scripted_batches=[[c]])
            result = gen.generate_all(runtime_paths=runtime_paths)

            self.assertEqual(len(result["samples"]), 3)
            self.assertEqual(len({_sample_fingerprint(s) for s in result["samples"]}), 3)
            self.assertEqual(result["stats"]["api_calls"], 2)
            self.assertTrue(result["stats"]["resumed_from_progress"])
            persisted_progress = json.loads(runtime_paths["progress"].read_text(encoding="utf-8"))
            self.assertEqual(len(persisted_progress["samples"]), 3)


if __name__ == "__main__":
    unittest.main()
