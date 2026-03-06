#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMAFACTORY_DIR = REPO_ROOT / "LLaMA-Factory"
DEFAULT_CONFIG = DEFAULT_LLAMAFACTORY_DIR / "examples" / "train_lora" / "qwen3_lora_sft_genesis_toolcall.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLaMA-Factory fine-tuning for Genesis tool-call dataset."
    )
    parser.add_argument(
        "--llamafactory-dir",
        type=Path,
        default=DEFAULT_LLAMAFACTORY_DIR,
        help="Path to LLaMA-Factory project.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config path for training.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print command, do not execute training.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional args passed to llamafactory-cli train. Put them after '--'.",
    )
    return parser.parse_args()


def _resolve_config_path(config: Path, llamafactory_dir: Path) -> Path:
    if config.is_absolute():
        return config
    if config.exists():
        return config.resolve()
    candidate = (llamafactory_dir / config).resolve()
    return candidate


def _resolve_train_prefix() -> list[str]:
    if shutil.which("llamafactory-cli") is not None:
        return ["llamafactory-cli"]
    return [sys.executable, "-m", "llamafactory.cli"]


def main() -> None:
    args = parse_args()

    llamafactory_dir = args.llamafactory_dir.resolve()
    if not llamafactory_dir.exists():
        raise FileNotFoundError(f"LLaMA-Factory dir not found: {llamafactory_dir}")

    config_path = _resolve_config_path(args.config, llamafactory_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = _resolve_train_prefix() + ["train", str(config_path)] + extra_args

    env = os.environ.copy()
    src_dir = str((llamafactory_dir / "src").resolve())
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not old_pythonpath else f"{src_dir}:{old_pythonpath}"
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus

    print("[finetune] working_dir:", llamafactory_dir)
    print("[finetune] command    :", shlex.join(command))
    if args.gpus is not None:
        print("[finetune] GPUs       :", args.gpus)

    if args.dry_run:
        print("[finetune] dry-run enabled, command not executed.")
        return

    subprocess.run(command, cwd=str(llamafactory_dir), env=env, check=True)


if __name__ == "__main__":
    main()
