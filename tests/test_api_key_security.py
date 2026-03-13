from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

from src.data_core.api_client import resolve_api_key
from src.data_core.generate import DatasetGenerator, GenerateDatasetConfig
from src.utils.secrets import safe_json_dumps


class ApiKeySecurityTests(unittest.TestCase):
    def test_missing_env_raises_clear_error(self) -> None:
        secret_env = "OPENAI_API_KEY"
        original = os.environ.pop(secret_env, None)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                resolve_api_key(api_key="", api_key_env=secret_env)
            message = str(ctx.exception)
            self.assertIn(secret_env, message)
            self.assertIn("missing", message.lower())

            # Compatibility field is preserved, but real secret must still come from env var.
            with self.assertRaises(RuntimeError):
                resolve_api_key(api_key="sk-config-key-should-not-be-used", api_key_env=secret_env)
        finally:
            if original is not None:
                os.environ[secret_env] = original

    def test_logs_do_not_leak_api_key(self) -> None:
        api_key = "sk-very-secret-key-123456"
        os.environ["DEEPSEEK_API_KEY"] = api_key
        cfg = GenerateDatasetConfig(
            num_samples=1,
            max_retries=1,
            sleep_seconds=0.0,
            api_key_env="DEEPSEEK_API_KEY",
            api_key="",
        )
        generator = DatasetGenerator(cfg)
        logger_name = "src.data_core.generate"

        with self.assertLogs(logger_name, level=logging.WARNING) as captured:
            with patch("src.data_core.generate.call_chat_api", side_effect=RuntimeError(f"Bearer {api_key}")):
                generator.generate_batch(difficulty="simple", batch_size=1, with_state=False)

        joined = "\n".join(captured.output)
        self.assertNotIn(api_key, joined)
        self.assertIn("[REDACTED]", joined)

    def test_config_and_report_dump_are_redacted(self) -> None:
        api_key = "sk-top-secret-abcdef"
        payload = {
            "config": {
                "api_key": api_key,
                "api_key_env": "OPENAI_API_KEY",
                "headers": {"Authorization": f"Bearer {api_key}"},
            },
            "report": {
                "request_token": "abc-token-123",
                "ok": True,
            },
        }
        dumped = safe_json_dumps(payload, ensure_ascii=False, indent=2)
        self.assertNotIn(api_key, dumped)
        self.assertIn("[REDACTED]", dumped)
        self.assertIn("OPENAI_API_KEY", dumped)


if __name__ == "__main__":
    unittest.main()
