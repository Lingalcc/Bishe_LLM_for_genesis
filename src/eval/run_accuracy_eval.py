#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.unified_config import DEFAULT_CONFIG_PATH, build_cli_args, get_section, load_config


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run accuracy evaluation with unified config defaults."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Unified config YAML path.",
    )
    args, passthrough = parser.parse_known_args(argv)
    return args, passthrough


def _invoke_impl(argv: list[str]) -> None:
    from src.eval.evaluate_toolcall_accuracy import main as impl_main

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        impl_main()
    finally:
        sys.argv = original_argv


def build_accuracy_cli_defaults(config_path: Path) -> list[str]:
    cfg = load_config(config_path)
    section = get_section(cfg, "test", "accuracy_eval")

    return build_cli_args(
        section,
        option_keys=[
            "dataset_file",
            "predictions_file",
            "report_file",
            "num_samples",
            "seed",
            "api_base",
            "model",
            "api_key",
            "api_key_env",
            "temperature",
            "max_tokens",
            "timeout",
            "max_retries",
            "sleep_seconds",
        ],
    )


def run_accuracy_eval(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    passthrough: list[str] | None = None,
) -> None:
    defaults = build_accuracy_cli_defaults(config_path)
    extra_args = list(passthrough or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    _invoke_impl([*defaults, *extra_args])


def main(argv: list[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    run_accuracy_eval(config_path=args.config, passthrough=passthrough)


if __name__ == "__main__":
    main()
