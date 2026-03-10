#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from src.utils.config import get_section, load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_LLAMAFACTORY_DIR = REPO_ROOT / "LLaMA-Factory"
DEFAULT_CONFIG = DEFAULT_LLAMAFACTORY_DIR / "examples" / "train_lora" / "qwen3_lora_sft_genesis_toolcall.yaml"
SUPPORTED_FINETUNE_METHODS = {"lora", "qlora", "dora", "galore"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLaMA-Factory fine-tuning for Genesis tool-call dataset."
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
        help="Unified YAML config path. Will read `finetune.train` section.",
    )
    parser.add_argument(
        "--llamafactory-dir",
        type=Path,
        default=None,
        help="Path to LLaMA-Factory project.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
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
        "--finetune-method",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_FINETUNE_METHODS),
        help="Fine-tuning method: lora / qlora / dora / galore.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional args passed to llamafactory-cli train. Put them after '--'.",
    )
    return parser.parse_args(argv)


def _resolve_config_path(config: Path, llamafactory_dir: Path) -> Path:
    if config.is_absolute():
        return config
    if config.exists():
        return config.resolve()
    repo_candidate = (REPO_ROOT / config).resolve()
    if repo_candidate.exists():
        return repo_candidate
    candidate = (llamafactory_dir / config).resolve()
    return candidate


def _resolve_train_prefix() -> list[str]:
    if shutil.which("llamafactory-cli") is not None:
        return ["llamafactory-cli"]
    return [sys.executable, "-m", "llamafactory.cli"]


def _load_finetune_train_section(pipeline_config: Path) -> dict[str, object]:
    if not pipeline_config.exists():
        return {}
    try:
        data = load_config(pipeline_config)
    except ValueError:
        return {}
    train = get_section(data, "finetune", "train")
    if not isinstance(train, dict):
        return {}
    return train


def _resolve_effective_settings(args: argparse.Namespace) -> tuple[Path, Path, str | None, bool, str]:
    section = _load_finetune_train_section(args.pipeline_config)

    llamafactory_dir_raw = args.llamafactory_dir or section.get("llamafactory_dir") or DEFAULT_LLAMAFACTORY_DIR
    config_raw = args.config or section.get("config") or DEFAULT_CONFIG
    gpus_raw = args.gpus if args.gpus is not None else section.get("gpus")
    finetune_method_raw = args.finetune_method or section.get("finetune_method") or "lora"
    dry_run = bool(args.dry_run or bool(section.get("dry_run", False)))

    llamafactory_dir = Path(str(llamafactory_dir_raw))
    if not llamafactory_dir.is_absolute() and args.llamafactory_dir is None:
        llamafactory_dir = REPO_ROOT / llamafactory_dir
    config_path = Path(str(config_raw))
    gpus = None if gpus_raw in (None, "") else str(gpus_raw)
    finetune_method = str(finetune_method_raw).strip().lower()
    if finetune_method not in SUPPORTED_FINETUNE_METHODS:
        raise ValueError(
            f"Unsupported finetune_method={finetune_method!r}. "
            f"Supported: {sorted(SUPPORTED_FINETUNE_METHODS)}"
        )

    return llamafactory_dir, config_path, gpus, dry_run, finetune_method


def _build_method_overrides(finetune_method: str) -> list[str]:
    overrides: list[str] = []
    if finetune_method == "lora":
        return overrides
    if finetune_method == "qlora":
        overrides.extend(["--quantization_bit", "4"])
        return overrides
    if finetune_method == "dora":
        overrides.extend(["--use_dora", "true"])
        return overrides
    if finetune_method == "galore":
        overrides.extend(["--use_galore", "true"])
        return overrides
    raise ValueError(f"Unsupported finetune method: {finetune_method}")


def run_finetune(
    *,
    pipeline_config: Path = DEFAULT_PIPELINE_CONFIG,
    llamafactory_dir: Path | None = None,
    config: Path | None = None,
    gpus: str | None = None,
    dry_run: bool = False,
    finetune_method: str | None = None,
    extra_args: list[str] | None = None,
) -> None:
    args = argparse.Namespace(
        pipeline_config=pipeline_config,
        llamafactory_dir=llamafactory_dir,
        config=config,
        gpus=gpus,
        dry_run=dry_run,
        finetune_method=finetune_method,
        extra_args=list(extra_args or []),
    )

    llamafactory_dir_raw, config_raw, gpus, dry_run, finetune_method = _resolve_effective_settings(args)

    llamafactory_dir = llamafactory_dir_raw.resolve()
    if not llamafactory_dir.exists():
        raise FileNotFoundError(f"LLaMA-Factory dir not found: {llamafactory_dir}")

    config_path = _resolve_config_path(config_raw, llamafactory_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    command_extra_args = list(args.extra_args)
    if command_extra_args and command_extra_args[0] == "--":
        command_extra_args = command_extra_args[1:]

    method_args = _build_method_overrides(finetune_method)
    command = _resolve_train_prefix() + ["train", str(config_path)] + method_args + command_extra_args

    env = os.environ.copy()
    src_dir = str((llamafactory_dir / "src").resolve())
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not old_pythonpath else f"{src_dir}:{old_pythonpath}"
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus

    print("[finetune] working_dir:", llamafactory_dir)
    print("[finetune] pipeline cfg:", args.pipeline_config)
    print("[finetune] method     :", finetune_method)
    if method_args:
        print("[finetune] method args:", shlex.join(method_args))
    print("[finetune] command    :", shlex.join(command))
    if gpus is not None:
        print("[finetune] GPUs       :", gpus)

    if dry_run:
        print("[finetune] dry-run enabled, command not executed.")
        return

    subprocess.run(command, cwd=str(llamafactory_dir), env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_finetune(
        pipeline_config=args.pipeline_config,
        llamafactory_dir=args.llamafactory_dir,
        config=args.config,
        gpus=args.gpus,
        dry_run=args.dry_run,
        finetune_method=args.finetune_method,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    main()
