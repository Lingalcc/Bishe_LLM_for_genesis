from __future__ import annotations

import os
import unittest

from src.data_core.generate import DatasetGenerator, GenerateDatasetConfig, _sample_fingerprint


class _StubGenerator(DatasetGenerator):
    def __init__(self, cfg: GenerateDatasetConfig, scripted_batches: list[list[dict]]) -> None:
        super().__init__(cfg)
        self._scripted_batches = scripted_batches
        self._cursor = 0

    def _prepare_batch_args(self, count: int) -> list[tuple[str, int, bool, int]]:
        return [("simple", count, False, 123)]

    def _generate_batch_from_args(
        self, difficulty: str, batch_size: int, with_state: bool, seed_hint: int,
    ) -> tuple[str, list[dict]]:
        if self._cursor >= len(self._scripted_batches):
            return difficulty, []
        out = self._scripted_batches[self._cursor]
        self._cursor += 1
        return difficulty, out


class GenerateDedupTests(unittest.TestCase):
    def test_generate_all_deduplicates_and_refills(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "dummy-key-for-test"
        cfg = GenerateDatasetConfig(
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


if __name__ == "__main__":
    unittest.main()
