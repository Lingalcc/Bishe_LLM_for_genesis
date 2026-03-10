#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.finetune_core.train import run_finetune_from_merged_config
from src.utils.config import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune experiment runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/02_finetune_exp/configs/train.yaml"),
        help="Experiment-local override config YAML path.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Global base config YAML path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not execute training command.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to llamafactory-cli train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )

    result = run_finetune_from_merged_config(
        merged_config,
        dry_run_override=args.dry_run,
        extra_args=tuple(args.extra_args),
    )
    print(f"[finetune] working_dir: {result['working_dir']}")
    print(f"[finetune] method     : {result['method']}")
    print(f"[finetune] command    : {result['command_shell']}")
    if result.get("gpus") is not None:
        print(f"[finetune] GPUs       : {result['gpus']}")
    print(f"[finetune] executed   : {result['executed']}")


if __name__ == "__main__":
    main()
