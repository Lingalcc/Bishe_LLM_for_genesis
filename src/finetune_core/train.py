from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import get_section, load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_LLAMAFACTORY_DIR = REPO_ROOT / "LLaMA-Factory"
DEFAULT_CONFIG = (
    DEFAULT_LLAMAFACTORY_DIR
    / "examples"
    / "train_lora"
    / "qwen3_lora_sft_genesis_toolcall.yaml"
)
SUPPORTED_FINETUNE_METHODS = {"lora", "qlora", "dora", "galore"}


@dataclass(frozen=True)
class FinetuneConfig:
    pipeline_config: Path = DEFAULT_PIPELINE_CONFIG
    llamafactory_dir: Path | None = None
    config: Path | None = None
    gpus: str | None = None
    dry_run: bool = False
    finetune_method: str | None = None
    extra_args: tuple[str, ...] = ()


def _resolve_config_path(config: Path, llamafactory_dir: Path) -> Path:
    if config.is_absolute():
        return config
    if config.exists():
        return config.resolve()
    repo_candidate = (REPO_ROOT / config).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (llamafactory_dir / config).resolve()


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


def _resolve_effective_settings(cfg: FinetuneConfig) -> tuple[Path, Path, str | None, bool, str]:
    section = _load_finetune_train_section(cfg.pipeline_config)

    llamafactory_dir_raw = cfg.llamafactory_dir or section.get("llamafactory_dir") or DEFAULT_LLAMAFACTORY_DIR
    config_raw = cfg.config or section.get("config") or DEFAULT_CONFIG
    gpus_raw = cfg.gpus if cfg.gpus is not None else section.get("gpus")
    finetune_method_raw = cfg.finetune_method or section.get("finetune_method") or "lora"
    dry_run = bool(cfg.dry_run or bool(section.get("dry_run", False)))

    llamafactory_dir = Path(str(llamafactory_dir_raw))
    if not llamafactory_dir.is_absolute() and cfg.llamafactory_dir is None:
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
    if finetune_method == "lora":
        return []
    if finetune_method == "qlora":
        return ["--quantization_bit", "4"]
    if finetune_method == "dora":
        return ["--use_dora", "true"]
    if finetune_method == "galore":
        return ["--use_galore", "true"]
    raise ValueError(f"Unsupported finetune method: {finetune_method}")


def run_finetune(cfg: FinetuneConfig) -> dict[str, Any]:
    llamafactory_dir_raw, config_raw, gpus, dry_run, finetune_method = _resolve_effective_settings(cfg)

    llamafactory_dir = llamafactory_dir_raw.resolve()
    if not llamafactory_dir.exists():
        raise FileNotFoundError(f"LLaMA-Factory dir not found: {llamafactory_dir}")

    config_path = _resolve_config_path(config_raw, llamafactory_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    command_extra_args = list(cfg.extra_args)
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

    result = {
        "working_dir": str(llamafactory_dir),
        "pipeline_config": str(cfg.pipeline_config),
        "method": finetune_method,
        "method_args": method_args,
        "command": command,
        "command_shell": shlex.join(command),
        "gpus": gpus,
        "dry_run": dry_run,
    }

    if dry_run:
        result["executed"] = False
        return result

    subprocess.run(command, cwd=str(llamafactory_dir), env=env, check=True)
    result["executed"] = True
    return result


def run_finetune_from_merged_config(
    merged_config: dict[str, Any],
    *,
    extra_args: tuple[str, ...] = (),
    dry_run_override: bool | None = None,
) -> dict[str, Any]:
    section = get_section(merged_config, "finetune", "train")
    cfg = FinetuneConfig(
        pipeline_config=Path(section.get("pipeline_config", DEFAULT_PIPELINE_CONFIG)),
        llamafactory_dir=Path(section["llamafactory_dir"]) if section.get("llamafactory_dir") else None,
        config=Path(section["config"]) if section.get("config") else None,
        gpus=str(section["gpus"]) if section.get("gpus") is not None else None,
        dry_run=bool(section.get("dry_run", False)) if dry_run_override is None else dry_run_override,
        finetune_method=(str(section["finetune_method"]) if section.get("finetune_method") else None),
        extra_args=extra_args,
    )
    return run_finetune(cfg)
