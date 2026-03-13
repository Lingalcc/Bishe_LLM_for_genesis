from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.utils.run_meta import record_run_meta


class RunMetaTests(unittest.TestCase):
    def test_run_meta_generated_with_required_fields_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report_dir = root / "reports"
            data_file = root / "dataset.json"
            data_file.write_text('[{"instruction":"a","output":"b"}]', encoding="utf-8")
            expected_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()

            merged_config = {
                "test": {
                    "accuracy_eval": {
                        "seed": 123,
                        "api_key": "sk-top-secret-abcdef",
                        "api_key_env": "OPENAI_API_KEY",
                        "report_file": str(report_dir / "accuracy_report.json"),
                    }
                },
                "headers": {
                    "Authorization": "Bearer sk-top-secret-abcdef",
                },
            }

            out_path = record_run_meta(
                report_dir,
                merged_config=merged_config,
                cli_args={"config": "x.yaml"},
                argv=["python", "run_accuracy.py", "--config", "x.yaml"],
                seed=123,
                data_paths=[data_file],
                extra_meta={"entry": "tests"},
            )

            self.assertTrue(out_path.exists())
            payload = json.loads(out_path.read_text(encoding="utf-8"))

            # 1) required top-level fields exist
            for key in (
                "git",
                "runtime",
                "dependencies",
                "environment",
                "gpu",
                "command",
                "config_snapshot",
                "timestamp",
                "random_seed",
                "data_files",
            ):
                self.assertIn(key, payload)

            # 2) critical field content
            self.assertIn("commit_hash", payload["git"])
            self.assertIn("dirty", payload["git"])
            self.assertIn("python_version", payload["runtime"])
            self.assertIn("CUDA_VISIBLE_DEVICES", payload["environment"])
            self.assertEqual(payload["random_seed"], 123)

            # 3) sensitive data redacted
            dumped = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("sk-top-secret-abcdef", dumped)
            self.assertIn("[REDACTED]", dumped)
            self.assertIn("OPENAI_API_KEY", dumped)

            # 4) config snapshot readable + data hash present
            self.assertIsInstance(payload["config_snapshot"], dict)
            self.assertIn("test", payload["config_snapshot"])
            self.assertTrue(payload["data_files"])
            self.assertEqual(payload["data_files"][0]["path"], str(data_file.resolve()))
            self.assertEqual(payload["data_files"][0]["sha256"], expected_hash)


if __name__ == "__main__":
    unittest.main()
