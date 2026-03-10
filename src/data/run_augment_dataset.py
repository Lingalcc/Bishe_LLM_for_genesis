#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.unified_config import (
    DEFAULT_CONFIG_PATH,
    load_dataset_prepare_runtime_config,
)


DEFAULT_INPUT_FILE = "data_prepare/genesis_franka_toolcall_alpaca.json"
DEFAULT_OUTPUT_FILE = "data_prepare/genesis_franka_toolcall_alpaca_augmented.json"
DEFAULT_STATS_FILE = "data_prepare/genesis_franka_toolcall_augment_stats.json"
DEFAULT_OUTPUT_SHAREGPT_FILE = "data_prepare/genesis_franka_toolcall_sharegpt_augmented.json"
DEFAULT_SEED = 42
DEFAULT_NUM_SOURCE = 800
DEFAULT_AUG_PER_SAMPLE = 2
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5"
DEFAULT_API_KEY = ""
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5
DEFAULT_SLEEP_SECONDS = 0.2

AUGMENT_DEFAULTS: dict[str, Any] = {
    "input_file": DEFAULT_INPUT_FILE,
    "output_file": DEFAULT_OUTPUT_FILE,
    "stats_file": DEFAULT_STATS_FILE,
    "output_sharegpt_file": DEFAULT_OUTPUT_SHAREGPT_FILE,
    "seed": DEFAULT_SEED,
    "num_source": DEFAULT_NUM_SOURCE,
    "aug_per_sample": DEFAULT_AUG_PER_SAMPLE,
    "api_base": DEFAULT_API_BASE,
    "model": DEFAULT_MODEL,
    "api_key": DEFAULT_API_KEY,
    "api_key_env": DEFAULT_API_KEY_ENV,
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "timeout": DEFAULT_TIMEOUT,
    "max_retries": DEFAULT_MAX_RETRIES,
    "sleep_seconds": DEFAULT_SLEEP_SECONDS,
}


def _invoke_impl(argv: list[str]) -> None:
    from src.data.augment_genesis_franka_dataset_with_api import main as impl_main

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        impl_main()
    finally:
        sys.argv = original_argv


def build_augment_cli_defaults(config_path: Path) -> list[str]:
    section = load_dataset_prepare_runtime_config(
        config_path=config_path,
        section="augment",
        defaults=AUGMENT_DEFAULTS,
    )

    input_file = str(section.get("input_file", DEFAULT_INPUT_FILE))
    output_file = str(section.get("output_file", DEFAULT_OUTPUT_FILE))
    stats_file = str(section.get("stats_file", DEFAULT_STATS_FILE))
    output_sharegpt_file = str(section.get("output_sharegpt_file", DEFAULT_OUTPUT_SHAREGPT_FILE))
    seed = int(section.get("seed", DEFAULT_SEED))
    num_source = int(section.get("num_source", DEFAULT_NUM_SOURCE))
    aug_per_sample = int(section.get("aug_per_sample", DEFAULT_AUG_PER_SAMPLE))
    api_base = str(section.get("api_base", DEFAULT_API_BASE))
    model = str(section.get("model", DEFAULT_MODEL))
    api_key = str(section.get("api_key", DEFAULT_API_KEY))
    api_key_env = str(section.get("api_key_env", DEFAULT_API_KEY_ENV))
    temperature = float(section.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(section.get("max_tokens", DEFAULT_MAX_TOKENS))
    timeout = int(section.get("timeout", DEFAULT_TIMEOUT))
    max_retries = int(section.get("max_retries", DEFAULT_MAX_RETRIES))
    sleep_seconds = float(section.get("sleep_seconds", DEFAULT_SLEEP_SECONDS))

    return [
        "--input-file",
        input_file,
        "--output-file",
        output_file,
        "--stats-file",
        stats_file,
        "--output-sharegpt-file",
        output_sharegpt_file,
        "--seed",
        str(seed),
        "--num-source",
        str(num_source),
        "--aug-per-sample",
        str(aug_per_sample),
        "--api-base",
        api_base,
        "--model",
        model,
        "--api-key",
        api_key,
        "--api-key-env",
        api_key_env,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--timeout",
        str(timeout),
        "--max-retries",
        str(max_retries),
        "--sleep-seconds",
        str(sleep_seconds),
    ]


def run_augment(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    passthrough: list[str] | None = None,
) -> None:
    defaults = build_augment_cli_defaults(config_path)
    extra_args = list(passthrough or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    _invoke_impl([*defaults, *extra_args])


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Augment dataset with unified config defaults."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Unified config YAML path.",
    )
    args, passthrough = parser.parse_known_args(argv)
    return args, passthrough


def main(argv: list[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    run_augment(config_path=args.config, passthrough=passthrough)


if __name__ == "__main__":
    main()
