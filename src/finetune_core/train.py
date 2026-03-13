from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data_core.dataset_safety import enforce_train_eval_no_leakage
from src.utils.config import get_section, load_config

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_LLAMAFACTORY_DIR = (
    REPO_ROOT / "LlamaFactory" if (REPO_ROOT / "LlamaFactory").exists() else REPO_ROOT / "LLaMA-Factory"
)
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "02_finetune_exp" / "configs" / "llamafactory_train_lora_sft.yaml"
SUPPORTED_FINETUNE_METHODS = {"lora", "qlora", "dora", "galore"}

# Per-method default base configs — auto-selected when no explicit config is given.
# Each config is pre-tuned for the method (correct finetuning_type, quantization flags, etc.).
_CONFIGS_DIR = REPO_ROOT / "experiments" / "02_finetune_exp" / "configs"
_METHOD_DEFAULT_CONFIGS: dict[str, Path] = {
    "lora":   _CONFIGS_DIR / "llamafactory_train_lora_sft.yaml",
    "qlora":  _CONFIGS_DIR / "llamafactory_train_qlora_sft.yaml",
    "dora":   _CONFIGS_DIR / "llamafactory_train_dora_sft.yaml",
    "galore": _CONFIGS_DIR / "llamafactory_train_galore_sft.yaml",
}


@dataclass(frozen=True)
class FinetuneConfig:
    pipeline_config: Path = DEFAULT_PIPELINE_CONFIG
    llamafactory_dir: Path | None = None
    config: Path | None = None
    gpus: str | None = None
    dry_run: bool = False
    finetune_method: str | None = None
    dataset_overrides: tuple[str, ...] = ()
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
    gpus_raw = cfg.gpus if cfg.gpus is not None else section.get("gpus")
    finetune_method_raw = cfg.finetune_method or section.get("finetune_method") or "lora"
    dry_run = bool(cfg.dry_run or bool(section.get("dry_run", False)))

    llamafactory_dir = Path(str(llamafactory_dir_raw))
    if not llamafactory_dir.is_absolute() and cfg.llamafactory_dir is None:
        llamafactory_dir = REPO_ROOT / llamafactory_dir
    gpus = None if gpus_raw in (None, "") else str(gpus_raw)
    finetune_method = str(finetune_method_raw).strip().lower()
    if finetune_method not in SUPPORTED_FINETUNE_METHODS:
        raise ValueError(
            f"Unsupported finetune_method={finetune_method!r}. "
            f"Supported: {sorted(SUPPORTED_FINETUNE_METHODS)}"
        )

    # Auto-select the method-appropriate base config when nothing is explicitly specified.
    # This prevents the "selected galore but config is still lora/qlora" class of bugs.
    explicit_config = cfg.config or section.get("config")
    if explicit_config:
        config_raw = explicit_config
    else:
        config_raw = _METHOD_DEFAULT_CONFIGS.get(finetune_method, DEFAULT_CONFIG)
        logger.debug(
            "No explicit config provided; auto-selected config for method=%s: %s",
            finetune_method,
            config_raw,
        )

    config_path = Path(str(config_raw))
    return llamafactory_dir, config_path, gpus, dry_run, finetune_method


def _build_method_overrides(finetune_method: str) -> list[str]:
    """Return CLI key=value overrides that fully characterise the requested finetune method.

    These are appended *after* the base YAML config, so they take precedence over whatever
    the YAML says.  Each method sets every field it requires AND explicitly corrects fields
    that would conflict (e.g. GaLore must not have finetuning_type=lora).

    Support matrix
    --------------
    lora   : adapter-based, no quantization.
    qlora  : adapter-based + 4-bit NF4 quantization (requires bitsandbytes).
    dora   : adapter-based LoRA variant with weight-decomposition, no quantization.
    galore : full-parameter with gradient low-rank projection (requires finetuning_type=full).
    """
    if finetune_method == "lora":
        # Explicit: adapter mode, quantization disabled.
        return [
            "finetuning_type=lora",
        ]
    if finetune_method == "qlora":
        # Explicit: adapter mode + 4-bit NF4 quantization.
        return [
            "finetuning_type=lora",
            "quantization_bit=4",
            "quantization_type=nf4",
            "double_quantization=true",
        ]
    if finetune_method == "dora":
        # Explicit: LoRA adapter with use_dora=true; no quantization.
        return [
            "finetuning_type=lora",
            "use_dora=true",
        ]
    if finetune_method == "galore":
        # GaLore is an optimizer-level technique for full fine-tuning.
        # finetuning_type MUST be "full"; using "lora" silently breaks GaLore.
        return [
            "finetuning_type=full",
            "use_galore=true",
            "galore_target=all",
        ]
    raise ValueError(f"Unsupported finetune method: {finetune_method}")


def _validate_method_requirements(finetune_method: str, *, dry_run: bool = False) -> None:
    """Raise a clear error early if required packages for a method are absent.

    Skips import checks during dry-run so the command can be previewed without
    installing heavy CUDA dependencies.
    """
    if dry_run:
        return

    if finetune_method == "qlora":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "[qlora] Missing required package: bitsandbytes.\n"
                "Install with: pip install bitsandbytes\n"
                "bitsandbytes is mandatory for 4-bit NF4 quantization used by QLoRA."
            ) from None


def _parse_gpu_indices(cuda_visible_devices: str | None) -> list[int] | None:
    """Parse numeric GPU indices from CUDA_VISIBLE_DEVICES, or None if not numeric."""
    if not cuda_visible_devices:
        return None
    indices: list[int] = []
    for part in cuda_visible_devices.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            return None
        indices.append(int(token))
    return indices or None


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
        overrides.append(f"{key}={resolved}")
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


def _print_dry_run_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of what would be executed, for easy inspection."""
    sep = "-" * 60
    print(sep)
    print("[finetune] DRY-RUN — no training will be executed.")
    print(f"  method      : {result['method']}")
    print(f"  base_config : {result['base_config']}")
    print(f"  method_args : {result['method_args']}")
    print(f"  gpus        : {result['gpus']}")
    if result.get("dataset_overrides"):
        print(f"  dataset     : {result['dataset_overrides']}")
    print(f"  command     : {result['command_shell']}")
    print(sep)


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
    command = (
        _resolve_train_prefix()
        + ["train", str(config_path)]
        + method_args
        + path_overrides
        + list(cfg.dataset_overrides)
        + command_extra_args
    )

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
        "base_config": str(config_path),
        "method_args": method_args,
        "command": command,
        "command_shell": shlex.join(command),
        "gpus": gpus,
        "dry_run": dry_run,
        "dataset_overrides": list(cfg.dataset_overrides),
    }

    if dry_run:
        result["executed"] = False
        _print_dry_run_summary(result)
        return result

    _validate_method_requirements(finetune_method, dry_run=False)
    _ensure_finetune_model_exists(config_path)

    # Parse GPU indices for monitoring.
    # Prefer effective CUDA_VISIBLE_DEVICES from env to avoid mismatches.
    effective_visible_gpus = env.get("CUDA_VISIBLE_DEVICES")
    gpu_indices = _parse_gpu_indices(effective_visible_gpus)
    gpu_monitor = GPUMonitor(gpu_indices=gpu_indices, interval_sec=0.5)

    logger.info("Starting training: method=%s gpus=%s", finetune_method, gpus)
    t_start = time.time()

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(command, cwd=str(llamafactory_dir), env=env)
        gpu_monitor.set_target_pid(proc.pid)
        gpu_monitor.start()
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
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
        peak_delta_vram_mb=vram_summary.get("peak_delta_vram_mb", 0.0),
        avg_delta_vram_mb=vram_summary.get("avg_delta_vram_mb", 0.0),
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


def _build_split_dataset_overrides(train_file: Path, val_file: Path) -> tuple[str, ...]:
    """Create a runtime dataset_info.json so LLaMA Factory reads explicit train/val files."""
    runtime_dataset_dir = (REPO_ROOT / ".cache" / "llamafactory_dataset_splits").resolve()
    runtime_dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_info_path = runtime_dataset_dir / "dataset_info.json"
    dataset_info = {
        "__train_split__": {"file_name": str(train_file)},
        "__val_split__": {"file_name": str(val_file)},
    }
    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (
        f"dataset_dir={runtime_dataset_dir}",
        "dataset=__train_split__",
        "eval_dataset=__val_split__",
        "val_size=0.0",
    )


def run_finetune_from_merged_config(
    merged_config: dict[str, Any],
    *,
    extra_args: tuple[str, ...] = (),
    dry_run_override: bool | None = None,
) -> dict[str, Any]:
    section = get_section(merged_config, "finetune", "train")
    train_file = Path(section["train_file"]).expanduser().resolve() if section.get("train_file") else None
    val_file = Path(section["val_file"]).expanduser().resolve() if section.get("val_file") else None
    if (train_file is None) ^ (val_file is None):
        raise ValueError("finetune.train.train_file and finetune.train.val_file must be provided together.")

    leak_cfg = section.get("leakage_check", {}) if isinstance(section.get("leakage_check"), dict) else {}
    leakage_enabled = bool(leak_cfg.get("enabled", True))
    leakage_strict = bool(leak_cfg.get("strict", True))
    test_section = get_section(merged_config, "test", "accuracy_eval")
    test_file_raw = test_section.get("test_file") or test_section.get("dataset_file")
    test_file = Path(str(test_file_raw)).expanduser().resolve() if test_file_raw else None

    if leakage_enabled:
        enforce_train_eval_no_leakage(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            strict=leakage_strict,
            check_content_overlap=True,
        )

    dataset_overrides: tuple[str, ...] = ()
    if train_file is not None and val_file is not None:
        if not train_file.exists():
            raise FileNotFoundError(f"train_file not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"val_file not found: {val_file}")
        dataset_overrides = _build_split_dataset_overrides(train_file, val_file)

    cfg = FinetuneConfig(
        pipeline_config=Path(section.get("pipeline_config", DEFAULT_PIPELINE_CONFIG)),
        llamafactory_dir=Path(section["llamafactory_dir"]) if section.get("llamafactory_dir") else None,
        config=Path(section["config"]) if section.get("config") else None,
        gpus=str(section["gpus"]) if section.get("gpus") is not None else None,
        dry_run=bool(section.get("dry_run", False)) if dry_run_override is None else dry_run_override,
        finetune_method=(str(section["finetune_method"]) if section.get("finetune_method") else None),
        dataset_overrides=dataset_overrides,
        extra_args=extra_args,
    )
    return run_finetune(cfg)
