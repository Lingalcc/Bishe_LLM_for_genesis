#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.eval_core.accuracy import run_accuracy_from_merged_config
from src.utils.config import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accuracy evaluation experiment runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/03_eval_exp/configs/accuracy.yaml"),
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
    report = run_accuracy_from_merged_config(merged_config)

    print(f"[ok] evaluated samples : {report['num_samples_evaluated']}")
    print(f"[ok] parse ok          : {report['parse_ok']} ({report['parse_ok_rate']:.4f})")
    print(f"[ok] exact match       : {report['exact_match']} ({report['exact_match_rate']:.4f})")
    print(f"[ok] action match      : {report['action_match']} ({report['action_match_rate']:.4f})")


if __name__ == "__main__":
    main()
