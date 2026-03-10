#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_core.augment import run_augment_from_merged_config
from src.utils.config import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data augmentation experiment runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/01_data_exp/configs/augment.yaml"),
        help="Experiment-local override config YAML path.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Global base config YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )
    result = run_augment_from_merged_config(merged_config)

    print(f"[ok] input     : {result['input_file']} ({result['input_count']} rows)")
    print(f"[ok] augmented : +{result['augmented_count']} rows")
    print(f"[ok] output    : {result['output_file']} ({result['output_count']} rows)")
    print(f"[ok] sharegpt  : {result['output_sharegpt_file']} ({result['sharegpt_count']} rows)")
    print(f"[ok] stats     : {result['stats_file']}")


if __name__ == "__main__":
    main()
