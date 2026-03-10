#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


DEFAULT_NUM_SAMPLES = 4000
DEFAULT_SEED = 42
DEFAULT_STATE_CONTEXT_RATIO = 0.7
DEFAULT_OUT_DIR = "data_prepare"
DEFAULT_ALPACA_FILE = "genesis_franka_toolcall_alpaca.json"
DEFAULT_SHAREGPT_FILE = "genesis_franka_toolcall_sharegpt.json"
DEFAULT_STATS_FILE = "genesis_franka_toolcall_stats.json"
DEFAULT_ACTION_MAP_FILE = "src/data/configs/action_map.default.json"
DEFAULT_ACTION_WEIGHTS: dict[str, int] | None = None

GENERATE_DEFAULTS: dict[str, Any] = {
    "num_samples": DEFAULT_NUM_SAMPLES,
    "seed": DEFAULT_SEED,
    "state_context_ratio": DEFAULT_STATE_CONTEXT_RATIO,
    "out_dir": DEFAULT_OUT_DIR,
    "alpaca_file": DEFAULT_ALPACA_FILE,
    "sharegpt_file": DEFAULT_SHAREGPT_FILE,
    "stats_file": DEFAULT_STATS_FILE,
    "action_map_file": DEFAULT_ACTION_MAP_FILE,
    "action_weights": DEFAULT_ACTION_WEIGHTS,
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Generate dataset with unified config defaults."
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
    from src.data.generate_genesis_franka_dataset import main as impl_main

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        impl_main()
    finally:
        sys.argv = original_argv


def build_generate_cli_defaults(config_path: Path) -> list[str]:
    section = load_dataset_prepare_runtime_config(
        config_path=config_path,
        section="generate",
        defaults=GENERATE_DEFAULTS,
    )

    num_samples = int(section.get("num_samples", DEFAULT_NUM_SAMPLES))
    seed = int(section.get("seed", DEFAULT_SEED))
    state_context_ratio = float(section.get("state_context_ratio", DEFAULT_STATE_CONTEXT_RATIO))
    out_dir = str(section.get("out_dir", DEFAULT_OUT_DIR))
    alpaca_file = str(section.get("alpaca_file", DEFAULT_ALPACA_FILE))
    sharegpt_file = str(section.get("sharegpt_file", DEFAULT_SHAREGPT_FILE))
    stats_file = str(section.get("stats_file", DEFAULT_STATS_FILE))
    action_map_file = section.get("action_map_file", DEFAULT_ACTION_MAP_FILE)
    action_weights = section.get("action_weights", DEFAULT_ACTION_WEIGHTS)

    defaults = [
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--state-context-ratio",
        str(state_context_ratio),
        "--out-dir",
        out_dir,
        "--alpaca-file",
        alpaca_file,
        "--sharegpt-file",
        sharegpt_file,
        "--stats-file",
        stats_file,
    ]
    if isinstance(action_weights, dict):
        defaults.extend(
            [
                "--action-map-json",
                json.dumps(action_weights, ensure_ascii=False),
            ]
        )
    elif action_map_file:
        defaults.extend(["--action-map-file", str(action_map_file)])
    return defaults


def run_generate(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    passthrough: list[str] | None = None,
) -> None:
    defaults = build_generate_cli_defaults(config_path)
    extra_args = list(passthrough or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    _invoke_impl([*defaults, *extra_args])


def main(argv: list[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    run_generate(config_path=args.config, passthrough=passthrough)


if __name__ == "__main__":
    main()
