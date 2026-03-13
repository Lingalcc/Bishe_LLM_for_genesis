from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.secrets import redact_secrets, safe_json_dumps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPENDENCIES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "vllm",
    "pyyaml",
    "openai",
    "huggingface_hub",
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _to_jsonable(vars(value))
    return value


def _repo_git_info(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "commit_hash": None,
        "dirty": None,
        "branch": None,
    }
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["commit_hash"] = commit or None

        branch = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = branch or None

        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["dirty"] = bool(status.strip())
    except Exception:
        pass
    return info


def _dependency_versions(names: tuple[str, ...] | list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "detected": False,
        "count": 0,
        "devices": [],
        "source": None,
    }

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            devices: list[dict[str, Any]] = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            info.update(
                {
                    "detected": True,
                    "count": count,
                    "devices": devices,
                    "source": "torch",
                    "cuda_version": getattr(torch.version, "cuda", None),
                }
            )
            return info
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        devices = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx, name, mem_mb, driver = parts
            devices.append(
                {
                    "index": int(idx),
                    "name": name,
                    "total_memory_mb": int(float(mem_mb)),
                    "driver_version": driver,
                }
            )
        info.update(
            {
                "detected": bool(devices),
                "count": len(devices),
                "devices": devices,
                "source": "nvidia-smi",
            }
        )
    except Exception:
        pass

    return info


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_data_paths(data_paths: list[str | Path] | tuple[str | Path, ...] | None) -> list[Path]:
    if not data_paths:
        return []

    seen: set[Path] = set()
    normalized: list[Path] = []
    for item in data_paths:
        if item is None:
            continue
        raw = str(item).strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        normalized.append(p)
    return normalized


def _resolve_seed(explicit_seed: int | None, merged_config: dict[str, Any] | None) -> int | None:
    if explicit_seed is not None:
        return int(explicit_seed)
    if not isinstance(merged_config, dict):
        return None

    candidates = [
        merged_config.get("benchmark", {}).get("seed") if isinstance(merged_config.get("benchmark"), dict) else None,
        merged_config.get("test", {}).get("accuracy_eval", {}).get("seed")
        if isinstance(merged_config.get("test"), dict)
        and isinstance(merged_config.get("test", {}).get("accuracy_eval"), dict)
        else None,
        merged_config.get("finetune", {}).get("train", {}).get("seed")
        if isinstance(merged_config.get("finetune"), dict)
        and isinstance(merged_config.get("finetune", {}).get("train"), dict)
        else None,
        merged_config.get("dataset_prepare", {}).get("generate", {}).get("seed")
        if isinstance(merged_config.get("dataset_prepare"), dict)
        and isinstance(merged_config.get("dataset_prepare", {}).get("generate"), dict)
        else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except Exception:
            continue
    return None


def collect_run_meta(
    *,
    merged_config: dict[str, Any] | None = None,
    cli_args: dict[str, Any] | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    data_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    dependencies: list[str] | tuple[str, ...] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    normalized_data_paths = _normalize_data_paths(data_paths)

    data_files: list[dict[str, Any]] = []
    for p in normalized_data_paths:
        exists = p.exists() and p.is_file()
        item: dict[str, Any] = {
            "path": str(p),
            "exists": exists,
        }
        if exists:
            item["sha256"] = _sha256_file(p)
            item["size_bytes"] = p.stat().st_size
        data_files.append(item)

    dep_names = tuple(dependencies) if dependencies else DEFAULT_DEPENDENCIES

    meta = {
        "timestamp": {
            "utc": now.isoformat().replace("+00:00", "Z"),
            "epoch_sec": int(now.timestamp()),
        },
        "git": _repo_git_info(REPO_ROOT),
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": _dependency_versions(dep_names),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu": _gpu_info(),
        "command": {
            "argv": list(argv) if argv is not None else list(sys.argv),
            "cli_args": _to_jsonable(cli_args or {}),
        },
        "random_seed": _resolve_seed(seed, merged_config),
        "config_snapshot": _to_jsonable(merged_config or {}),
        "data_files": data_files,
    }

    if extra_meta:
        meta["extra"] = _to_jsonable(extra_meta)

    return redact_secrets(meta)


def write_run_meta(output_dir: str | Path, meta: dict[str, Any], filename: str = "run_meta.json") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(safe_json_dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def record_run_meta(
    output_dir: str | Path,
    *,
    merged_config: dict[str, Any] | None = None,
    cli_args: dict[str, Any] | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    data_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    dependencies: list[str] | tuple[str, ...] | None = None,
    extra_meta: dict[str, Any] | None = None,
    filename: str = "run_meta.json",
) -> Path:
    meta = collect_run_meta(
        merged_config=merged_config,
        cli_args=cli_args,
        argv=argv,
        seed=seed,
        data_paths=data_paths,
        dependencies=dependencies,
        extra_meta=extra_meta,
    )
    return write_run_meta(output_dir=output_dir, meta=meta, filename=filename)
