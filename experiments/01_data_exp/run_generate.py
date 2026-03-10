#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_core.generate import run_generate_from_merged_config
from src.utils.config import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data generation experiment runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/01_data_exp/configs/generate.yaml"),
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
    output_paths = run_generate_from_merged_config(merged_config)

    print(f"[ok] alpaca  : {output_paths['alpaca_path']}")
    print(f"[ok] sharegpt: {output_paths['sharegpt_path']}")
    print(f"[ok] stats   : {output_paths['stats_path']}")


if __name__ == "__main__":
    main()
