from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import get_section, load_config

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_LLAMAFACTORY_DIR = (
    REPO_ROOT / "LlamaFactory" if (REPO_ROOT / "LlamaFactory").exists() else REPO_ROOT / "LLaMA-Factory"
)
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "02_finetune_exp" / "configs" / "llamafactory_train_lora_sft.yaml"
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


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to parse training config. Install dependency `pyyaml`.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping object: {path}")
    return data


def _is_repo_model_managed_path(raw_model_path: str) -> bool:
    normalized = raw_model_path.strip().replace("\\", "/")
    return (
        normalized.startswith("model/")
        or normalized.startswith("./model/")
        or normalized.startswith("../model/")
        or normalized == "model"
    )


def _infer_hf_model_id(model_dir: Path) -> str | None:
    dirname = model_dir.name
    if "_" not in dirname:
        return None
    org, repo = dirname.split("_", 1)
    if not org or not repo:
        return None
    return f"{org}/{repo}"


def _read_top_level_yaml_scalar(config_path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        matched = pattern.match(line)
        if not matched:
            continue
        value = matched.group(1).split("#", 1)[0].strip()
        if not value:
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value.strip()
    return None


def _resolve_yaml_paths_for_subprocess(config_path: Path) -> list[str]:
    """Resolve project-root-relative paths in LlamaFactory config to absolute, returning CLI overrides."""
    overrides: list[str] = []
    try:
        data = _load_yaml_file(config_path)
    except Exception:
        return overrides
    for key in ("model_name_or_path", "output_dir", "dataset_dir"):
        value = data.get(key, "")
        if not value or not isinstance(value, str):
            continue
        p = Path(value)
        if p.is_absolute():
            continue
        resolved = (REPO_ROOT / p).resolve()
        overrides.extend([f"--{key}", str(resolved)])
    return overrides


def _ensure_finetune_model_exists(config_path: Path) -> None:
    cfg: dict[str, Any] = {}
    try:
        cfg = _load_yaml_file(config_path)
    except Exception:
        cfg = {}

    model_name_or_path = str(cfg.get("model_name_or_path", "")).strip() if cfg else ""
    if not model_name_or_path:
        model_name_or_path = _read_top_level_yaml_scalar(config_path, "model_name_or_path") or ""
    if not model_name_or_path:
        return

    model_path_input = Path(model_name_or_path).expanduser()
    is_local_model_ref = model_path_input.is_absolute() or _is_repo_model_managed_path(model_name_or_path)
    if not is_local_model_ref:
        return

    target_model_path = (
        model_path_input.resolve()
        if model_path_input.is_absolute()
        else (REPO_ROOT / model_path_input).resolve()
    )
    model_root = (REPO_ROOT / "model").resolve()

    if target_model_path.exists():
        return

    if not model_root.exists():
        print(f"[finetune] 检测到模型目录不存在，自动创建: {model_root}")
        model_root.mkdir(parents=True, exist_ok=True)

    explicit_model_id = (
        (str(cfg.get("hf_model_id", "")).strip() if cfg else "")
        or (str(cfg.get("model_id", "")).strip() if cfg else "")
        or (_read_top_level_yaml_scalar(config_path, "hf_model_id") or "")
        or (_read_top_level_yaml_scalar(config_path, "model_id") or "")
    )
    inferred_model_id = _infer_hf_model_id(target_model_path)
    model_id = explicit_model_id or inferred_model_id
    if not model_id:
        raise FileNotFoundError(
            f"Model path does not exist: {target_model_path}. "
            "Cannot infer Hugging Face repo id. Set `hf_model_id` in training YAML."
        )

    revision = (str(cfg.get("model_revision", "")).strip() if cfg else "") or (
        _read_top_level_yaml_scalar(config_path, "model_revision") or ""
    )
    token = (str(cfg.get("hf_token", "")).strip() if cfg else "") or (
        _read_top_level_yaml_scalar(config_path, "hf_token") or ""
    )
    revision = revision or None
    token = token or None

    print(f"[finetune] 检测到模型不存在: {target_model_path}")
    print(f"[finetune] 自动下载 Hugging Face 模型: {model_id}")
    print(f"[finetune] 下载目录: {target_model_path}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install it with `pip install huggingface_hub`."
        ) from exc

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(target_model_path),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[finetune] 模型下载完成: {target_model_path}")


def run_finetune(cfg: FinetuneConfig) -> dict[str, Any]:
    from src.finetune_core.metrics import GPUMonitor, TrainingMetrics, find_trainer_state, parse_trainer_state

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
    path_overrides = _resolve_yaml_paths_for_subprocess(config_path)
    command = _resolve_train_prefix() + ["train", str(config_path)] + method_args + path_overrides + command_extra_args

    env = os.environ.copy()
    src_dir = str((llamafactory_dir / "src").resolve())
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not old_pythonpath else f"{src_dir}:{old_pythonpath}"
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus

    result: dict[str, Any] = {
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

    _ensure_finetune_model_exists(config_path)

    # Parse GPU indices for monitoring
    gpu_indices = [int(g) for g in gpus.split(",") if g.strip()] if gpus else [0]
    gpu_monitor = GPUMonitor(gpu_indices=gpu_indices, interval_sec=2.0)

    logger.info("Starting training: method=%s gpus=%s", finetune_method, gpus)
    gpu_monitor.start()
    t_start = time.time()

    try:
        subprocess.run(command, cwd=str(llamafactory_dir), env=env, check=True)
    finally:
        gpu_monitor.stop()

    elapsed = time.time() - t_start
    vram_summary = gpu_monitor.summary()

    # Try to extract loss curve from trainer_state.json
    # Look for output_dir in the LLaMA Factory train config
    output_dir = _extract_output_dir(config_path)
    loss_data: dict[str, Any] = {}
    if output_dir:
        state_path = find_trainer_state(output_dir)
        if state_path:
            loss_data = parse_trainer_state(state_path)
            logger.info("Parsed loss curve: %d steps from %s", len(loss_data.get("train_loss", {}).get("steps", [])), state_path)

    metrics = TrainingMetrics(
        method=finetune_method,
        total_time_sec=elapsed,
        total_steps=loss_data.get("total_steps", 0),
        total_epochs=loss_data.get("total_epochs", 0.0),
        final_loss=loss_data.get("final_loss", 0.0),
        min_loss=loss_data.get("min_loss", 0.0),
        min_loss_step=loss_data.get("min_loss_step", 0),
        loss_curve=loss_data.get("train_loss", {}),
        peak_vram_mb=vram_summary.get("peak_vram_mb", 0.0),
        avg_vram_mb=vram_summary.get("avg_vram_mb", 0.0),
        vram_detail=vram_summary,
    )

    result["executed"] = True
    result["training_time_sec"] = elapsed
    result["training_metrics"] = metrics.to_dict()
    return result


def _extract_output_dir(config_path: Path) -> Path | None:
    """Try to read output_dir from a LLaMA Factory YAML config."""
    try:
        data = _load_yaml_file(config_path)
        if isinstance(data, dict) and "output_dir" in data:
            p = Path(data["output_dir"])
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            return p
    except Exception:
        pass
    return None


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
