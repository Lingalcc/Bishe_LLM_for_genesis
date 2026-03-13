from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval_core.accuracy import AccuracyEvalConfig, run_accuracy_eval
from src.eval_core.prompting import DEFAULT_EVAL_SYSTEM_PROMPT


class _FakeLocalEngine:
    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None

    def generate_chat(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return '{"action":"wait"}'


class EvalSystemPromptInjectionTests(unittest.TestCase):
    def _write_dataset(self, path: Path, *, sample_system: str | None) -> None:
        row = {
            "instruction": "open gripper",
            "output": '{"action":"wait"}',
        }
        if sample_system is not None:
            row["system"] = sample_system
        path.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")

    def _capture_local_system_prompt(self, *, cfg_system: str, sample_system: str | None) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_file = root / "dataset.json"
            report_file = root / "report_local.json"
            self._write_dataset(dataset_file, sample_system=sample_system)

            fake_engine = _FakeLocalEngine()
            cfg = AccuracyEvalConfig(
                dataset_file=dataset_file,
                report_file=report_file,
                num_samples=1,
                seed=1,
                mode="local",
                model_path="model/qwen2.5-3b-genesis-qlora",
                system_prompt=cfg_system,
            )
            with patch("src.eval_core.inference_engines.build_inference_engine", return_value=fake_engine):
                run_accuracy_eval(cfg)

            self.assertIsNotNone(fake_engine.last_messages)
            return fake_engine.last_messages[0]["content"]

    def _capture_api_system_prompt(self, *, cfg_system: str, sample_system: str | None) -> str:
        captured: dict[str, str] = {}

        def _fake_predict_once(**kwargs: object) -> str:
            messages = kwargs["messages"]
            if not isinstance(messages, list):
                raise TypeError("messages must be a list")
            captured["system"] = str(messages[0]["content"])
            return '{"action":"wait"}'

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_file = root / "dataset.json"
            report_file = root / "report_api.json"
            self._write_dataset(dataset_file, sample_system=sample_system)

            cfg = AccuracyEvalConfig(
                dataset_file=dataset_file,
                report_file=report_file,
                num_samples=1,
                seed=1,
                mode="api",
                system_prompt=cfg_system,
            )
            with (
                patch("src.eval_core.evaluate_toolcall_accuracy.resolve_api_key_from_env", return_value="sk-test"),
                patch("src.eval_core.evaluate_toolcall_accuracy.predict_once", side_effect=_fake_predict_once),
            ):
                run_accuracy_eval(cfg)

        return captured["system"]

    def test_cfg_system_prompt_has_highest_priority(self) -> None:
        cfg_prompt = "cfg prompt"
        sample_prompt = "sample prompt"
        self.assertEqual(
            self._capture_api_system_prompt(cfg_system=cfg_prompt, sample_system=sample_prompt),
            cfg_prompt,
        )

    def test_sample_system_prompt_used_when_cfg_empty(self) -> None:
        sample_prompt = "sample prompt"
        self.assertEqual(
            self._capture_api_system_prompt(cfg_system="  ", sample_system=sample_prompt),
            sample_prompt,
        )

    def test_default_system_prompt_used_when_cfg_and_sample_missing(self) -> None:
        self.assertEqual(
            self._capture_api_system_prompt(cfg_system="", sample_system=None),
            DEFAULT_EVAL_SYSTEM_PROMPT,
        )

    def test_api_and_local_use_same_prompt_resolution(self) -> None:
        cases = [
            {"cfg": "cfg first", "sample": "sample fallback", "expected": "cfg first"},
            {"cfg": "", "sample": "sample only", "expected": "sample only"},
            {"cfg": "", "sample": None, "expected": DEFAULT_EVAL_SYSTEM_PROMPT},
        ]
        for case in cases:
            with self.subTest(case=case):
                local_system = self._capture_local_system_prompt(
                    cfg_system=case["cfg"], sample_system=case["sample"],
                )
                api_system = self._capture_api_system_prompt(
                    cfg_system=case["cfg"], sample_system=case["sample"],
                )
                self.assertEqual(local_system, case["expected"])
                self.assertEqual(api_system, case["expected"])
                self.assertEqual(api_system, local_system)


if __name__ == "__main__":
    unittest.main()
