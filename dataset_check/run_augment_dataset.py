#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.unified_config import DEFAULT_CONFIG_PATH, build_cli_args, get_section, load_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Augment dataset with unified config defaults."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Unified config JSON path.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    cfg = load_config(args.config)
    section = get_section(cfg, "dataset_prepare", "augment")

    defaults = build_cli_args(
        section,
        option_keys=[
            "input_file",
            "output_file",
            "stats_file",
            "output_sharegpt_file",
            "seed",
            "num_source",
            "aug_per_sample",
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

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    sys.argv = [sys.argv[0], *defaults, *passthrough]
    from pipeline.dataset_prepare.augment_genesis_franka_dataset_with_api import main as impl_main

    impl_main()


if __name__ == "__main__":
    main()
