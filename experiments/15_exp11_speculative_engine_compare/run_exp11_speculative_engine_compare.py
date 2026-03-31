#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "speculative_engine_compare.yaml"
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_repo_root_on_pythonpath_env() -> None:
    repo_root = str(REPO_ROOT)
    pythonpath_items = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    if repo_root in pythonpath_items:
        return
    os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *pythonpath_items]) if pythonpath_items else repo_root


_ensure_repo_root_on_pythonpath_env()

from src.utils.config import load_config, load_merged_config
from src.utils.plotting import configure_report_matplotlib, pick_plot_text
from src.utils.run_meta import record_run_meta


DEFAULT_SYSTEM_PROMPT = (
    "你是 Franka 机械臂控制指令生成器。请把用户自然语言转换为可执行的 JSON action。"
    "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
    "只输出 JSON，不要输出解释。"
)
GPU_MEMORY_UTILIZATION_CAP = 0.99
GPU_MEMORY_UTILIZATION_FLOOR = 0.05
VRAM_POLL_INTERVAL_SEC = 0.2
POST_CASE_SLEEP_SECONDS = 10
DEFAULT_BACKENDS = ("vllm", "transformers")
DEFAULT_DECODING_MODES = ("standard", "speculative")
DEFAULT_EXPERIMENT_PROFILE = "exp8_aligned"
DEFAULT_VLLM_DTYPE = "float16"


@dataclass
class ResolvedModelRef:
    raw: str
    resolved: str
    is_local: bool


@dataclass
class LoadedModelBundle:
    model: Any
    base_model_ref: str
    adapter_path: str | None
    primary_device: str
    device_map: dict[str, str] | None
    cpu_offload_detected: bool
    load_strategy: str
    dtype_label: str
    attn_implementation: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验15 Exp11：按 Exp8 口径对比 vLLM 与 Transformers 的无投机 / 投机解码。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="实验配置文件路径。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG_PATH, help="基础配置文件路径。")
    parser.add_argument("--target-model-path", type=str, default=None, help="主模型路径。")
    parser.add_argument("--assistant-model-path", type=str, default=None, help="助手模型路径。")
    parser.add_argument("--dataset-path", type=str, default=None, help="测试集路径。")
    parser.add_argument("--num-samples", type=int, default=None, help="评测样本数。")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小，默认建议为 1。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数。")
    parser.add_argument("--max-model-len", type=int, default=None, help="模型上下文长度上限。")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度，建议保持 0.0。")
    parser.add_argument("--warmup-samples", type=int, default=None, help="预热样本数。")
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="逗号分隔的后端列表，支持 vllm,transformers。",
    )
    parser.add_argument(
        "--assistant-num-tokens",
        type=int,
        default=None,
        help="投机解码中每轮草稿 token 数。",
    )
    parser.add_argument(
        "--decoding-modes",
        type=str,
        default=None,
        help="逗号分隔的解码模式，支持 standard,speculative。",
    )
    parser.add_argument(
        "--assistant-confidence-threshold",
        type=float,
        default=None,
        help="Transformers assisted decoding 的 assistant_confidence_threshold。",
    )
    parser.add_argument(
        "--assistant-num-tokens-schedule",
        type=str,
        default=None,
        help="Transformers assisted decoding 的 num_assistant_tokens_schedule。",
    )
    parser.add_argument("--gpu-id", type=int, default=None, help="物理 GPU 编号，默认自动推断。")
    parser.add_argument("--preferred-cuda-device", type=int, default=None, help="优先使用的 CUDA 设备编号。")
    parser.add_argument(
        "--disable-same-gpu-placement",
        action="store_true",
        help="禁用 Transformers 主模型与助手模型优先落在同一张 GPU 上。",
    )
    parser.add_argument(
        "--disable-auto-device-map-fallback",
        action="store_true",
        help="当单卡加载失败时，禁用回退到 device_map=auto。",
    )
    parser.add_argument(
        "--total-gpu-memory-mb",
        type=int,
        default=None,
        help="手动指定 GPU 总显存（MB），默认自动探测。",
    )
    parser.add_argument("--report-path", type=str, default=None, help="JSON 报告输出路径。")
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _parse_backend_list(raw_text: str | None, default_values: tuple[str, ...]) -> list[str]:
    text = str(raw_text or "").strip()
    source = text if text else ",".join(default_values)
    supported = {"vllm", "transformers"}
    backends: list[str] = []
    seen: set[str] = set()
    for chunk in source.split(","):
        backend = chunk.strip().lower()
        if not backend:
            continue
        if backend not in supported:
            raise ValueError(f"不支持的 backend={backend!r}，仅支持 {sorted(supported)}。")
        if backend in seen:
            continue
        seen.add(backend)
        backends.append(backend)
    if not backends:
        raise ValueError("至少需要一个 backend。")
    return backends


def _parse_decoding_mode_list(raw_text: str | None, default_values: tuple[str, ...]) -> list[str]:
    text = str(raw_text or "").strip()
    source = text if text else ",".join(default_values)
    supported = {"standard", "speculative"}
    modes: list[str] = []
    seen: set[str] = set()
    for chunk in source.split(","):
        mode = chunk.strip().lower()
        if not mode:
            continue
        if mode not in supported:
            raise ValueError(f"不支持的 decoding_mode={mode!r}，仅支持 {sorted(supported)}。")
        if mode in seen:
            continue
        seen.add(mode)
        modes.append(mode)
    if not modes:
        raise ValueError("至少需要一个 decoding_mode。")
    return modes


def _mode_label(decoding_mode: str) -> str:
    return "无投机解码" if decoding_mode == "standard" else "投机解码"


def _plot_mode_label(decoding_mode: str) -> str:
    return "Standard" if decoding_mode == "standard" else "Speculative"


def _scenario_label(backend: str, decoding_mode: str, speculative_method: str) -> str:
    backend_label = "Transformers" if backend == "transformers" else "vLLM"
    return f"{backend_label} {_mode_label(decoding_mode)}"


def _plot_scenario_label(backend: str, decoding_mode: str, speculative_method: str) -> str:
    backend_label = "Transformers" if backend == "transformers" else "vLLM"
    return f"{backend_label} {_plot_mode_label(decoding_mode)}"


def build_case_matrix(
    *,
    backends: list[str],
    decoding_modes: list[str],
    experiment_profile: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for backend in backends:
        for decoding_mode in decoding_modes:
            if backend == "vllm" and decoding_mode != "standard":
                continue
            speculative_method = "none"
            if decoding_mode == "speculative":
                speculative_method = "assistant_model"
            if backend == "vllm":
                execution_note = "vLLM 仅运行基线解码，主模型以 float16 精度加载，用于和 Transformers 基线与投机解码结果对比。"
            else:
                execution_note = (
                    "与 exp8 保持一致：BF16/FP16 自动选择、主模型与助手模型优先同卡加载、"
                    "assistant_confidence_threshold 与 constant schedule 按 exp8 配置执行。"
                    if decoding_mode == "speculative"
                    else "与 exp8 保持一致：BF16/FP16 自动选择、单模型生成，不启用投机解码。"
                )

            cases.append(
                {
                    "name": f"{backend}_{decoding_mode}",
                    "backend": backend,
                    "decoding_mode": decoding_mode,
                    "decoding_mode_label": _mode_label(decoding_mode),
                    "scenario_label": _scenario_label(backend, decoding_mode, speculative_method),
                    "plot_scenario_label": _plot_scenario_label(backend, decoding_mode, speculative_method),
                    "experiment_profile": experiment_profile,
                    "speculative_method": speculative_method,
                    "execution_note": execution_note,
                }
            )
    return cases


def infer_gpu_id(explicit_gpu_id: int | None) -> int:
    if explicit_gpu_id is not None:
        return int(explicit_gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


def query_total_gpu_memory_mb(gpu_id: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if values and values[0].replace(".", "", 1).isdigit():
            return int(float(values[0]))
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory // (1024 * 1024))
    except Exception:
        pass
    return None


def query_free_gpu_memory_mb(gpu_id: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if values and values[0].replace(".", "", 1).isdigit():
            return int(float(values[0]))
    except Exception:
        pass
    return None


def resolve_vllm_gpu_memory_utilization(
    *,
    config: dict[str, Any],
    gpu_id: int,
    total_gpu_memory_mb: int | None,
) -> float:
    requested_ratio = float(config.get("vllm_gpu_memory_utilization", 0.7))
    if not total_gpu_memory_mb or total_gpu_memory_mb <= 0:
        return requested_ratio

    free_gpu_memory_mb = query_free_gpu_memory_mb(gpu_id)
    if not free_gpu_memory_mb or free_gpu_memory_mb <= 0:
        return requested_ratio

    safe_ratio = max(
        GPU_MEMORY_UTILIZATION_FLOOR,
        min(
            requested_ratio,
            round(float(max(0, free_gpu_memory_mb - 256)) / float(total_gpu_memory_mb), 4),
        ),
    )
    return safe_ratio


def monitor_vram(stop_event: threading.Event) -> None:
    peak_vram_mb = 0.0
    gpu_id = int(getattr(stop_event, "gpu_id", 0))
    poll_interval_sec = float(getattr(stop_event, "poll_interval_sec", VRAM_POLL_INTERVAL_SEC))
    query_cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]

    while not stop_event.is_set():
        try:
            result = subprocess.run(query_cmd, check=True, capture_output=True, text=True)
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if values:
                peak_vram_mb = max(peak_vram_mb, max(values))
        except Exception as exc:
            stop_event.monitor_error = str(exc)
            break
        stop_event.wait(poll_interval_sec)

    stop_event.peak_vram_mb = peak_vram_mb


def _resolve_local_or_hf_ref(raw_value: str | Path) -> ResolvedModelRef:
    raw = str(raw_value).strip()
    if not raw:
        raise ValueError("模型路径不能为空。")

    candidate = Path(raw).expanduser()
    local_candidates: list[Path] = []
    if candidate.is_absolute():
        local_candidates.append(candidate)
    else:
        local_candidates.append((REPO_ROOT / candidate).resolve())
        local_candidates.append((EXPERIMENT_DIR / candidate).resolve())
        local_candidates.append((Path.cwd() / candidate).resolve())

    for local_path in local_candidates:
        if local_path.exists():
            return ResolvedModelRef(raw=raw, resolved=str(local_path), is_local=True)
    return ResolvedModelRef(raw=raw, resolved=raw, is_local=False)


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _maybe_load_adapter(model_ref: ResolvedModelRef) -> tuple[str, str | None]:
    if not model_ref.is_local:
        return model_ref.resolved, None

    adapter_config_path = Path(model_ref.resolved) / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_ref.resolved, None

    adapter_config = _load_json_dict(adapter_config_path)
    if not adapter_config:
        raise ValueError(f"无法解析 adapter 配置: {adapter_config_path}")

    base_model_name_or_path = str(adapter_config.get("base_model_name_or_path", "")).strip()
    if not base_model_name_or_path:
        raise ValueError(f"adapter_config.json 缺少 base_model_name_or_path: {adapter_config_path}")

    base_ref = _resolve_local_or_hf_ref(base_model_name_or_path)
    return base_ref.resolved, model_ref.resolved


def _resolve_dataset_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (EXPERIMENT_DIR / candidate).resolve()


def _resolve_report_path(raw_path: str | Path | None) -> Path:
    if raw_path is None:
        return (REPORTS_DIR / "speculative_engine_compare_report.json").resolve()
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (EXPERIMENT_DIR / candidate).resolve()


def _load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    file_config = load_config(args.config)
    runtime_config = dict(file_config)

    cli_overrides = {
        "target_model_path": args.target_model_path,
        "assistant_model_path": args.assistant_model_path,
        "dataset_path": args.dataset_path,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "warmup_samples": args.warmup_samples,
        "backends": args.backends,
        "decoding_modes": args.decoding_modes,
        "assistant_num_tokens": args.assistant_num_tokens,
        "assistant_confidence_threshold": args.assistant_confidence_threshold,
        "assistant_num_tokens_schedule": args.assistant_num_tokens_schedule,
        "preferred_cuda_device": args.preferred_cuda_device,
        "prefer_same_gpu": (False if args.disable_same_gpu_placement else None),
        "allow_auto_device_map_fallback": (False if args.disable_auto_device_map_fallback else None),
        "report_path": args.report_path,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            runtime_config[key] = value

    required_keys = [
        "target_model_path",
        "assistant_model_path",
        "dataset_path",
        "num_samples",
        "batch_size",
        "max_new_tokens",
        "temperature",
    ]
    missing = [key for key in required_keys if key not in runtime_config]
    if missing:
        raise ValueError(f"配置缺少必要字段: {missing}")
    runtime_config.setdefault("assistant_num_tokens", 8)
    runtime_config.setdefault("assistant_confidence_threshold", 0.55)
    runtime_config.setdefault("assistant_num_tokens_schedule", "constant")
    runtime_config.setdefault("prefer_same_gpu", True)
    runtime_config.setdefault("allow_auto_device_map_fallback", True)
    runtime_config.setdefault("experiment_profile", DEFAULT_EXPERIMENT_PROFILE)
    runtime_config.setdefault("vllm_dtype", DEFAULT_VLLM_DTYPE)
    runtime_config.setdefault("vllm_gpu_memory_utilization", 0.7)
    runtime_config.setdefault("vllm_enforce_eager", False)
    runtime_config.setdefault("vllm_enable_prefix_caching", False)
    return runtime_config


def _load_system_prompt(base_config_path: Path) -> str:
    merged_config = load_merged_config(base_config_path=base_config_path, override_config_path=None)
    section = (
        merged_config.get("test", {}).get("accuracy_eval", {})
        if isinstance(merged_config.get("test"), dict)
        else {}
    )
    prompt = str(section.get("system_prompt", "")).strip()
    return prompt or DEFAULT_SYSTEM_PROMPT


def _load_dataset(dataset_path: Path, num_samples: int) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"测试集必须是 JSON 数组: {dataset_path}")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        expected = str(item.get("output", "")).strip()
        if not instruction:
            continue
        normalized.append(
            {
                "index": idx,
                "instruction": instruction,
                "input": input_text,
                "expected_output": expected,
            }
        )
    if not normalized:
        raise ValueError(f"测试集中没有可用样本: {dataset_path}")
    return normalized[: max(1, int(num_samples))]


def _compose_user_prompt(instruction: str, input_text: str) -> str:
    if not input_text:
        return instruction
    return f"{instruction}\n\n补充输入：\n{input_text}"


def _render_chat_prompt(tokenizer: Any, system_prompt: str, instruction: str, input_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _compose_user_prompt(instruction, input_text)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(
        [
            f"<|system|>\n{system_prompt}",
            f"<|user|>\n{_compose_user_prompt(instruction, input_text)}",
            "<|assistant|>\n",
        ]
    )


def _try_parse_json(text: str) -> tuple[bool, str | None]:
    try:
        json.loads(text)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _cleanup_runtime_state() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def _maybe_shutdown_vllm_engine(llm: Any) -> None:
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    for candidate in [llm, engine, getattr(engine, "engine_core", None)]:
        if candidate is None:
            continue
        shutdown = getattr(candidate, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                continue


def _looks_like_oom_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "less than desired gpu memory utilization",
            "insufficient memory",
        ]
    )


def _classify_error_status(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return "oom" if _looks_like_oom_text(text) else "failed"


def _detect_model_primary_device(model: Any) -> str:
    if hasattr(model, "device") and model.device is not None:
        return str(model.device)
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


def _extract_hf_device_map(model: Any) -> dict[str, str] | None:
    raw_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_map, dict):
        return None
    return {str(key): str(value) for key, value in raw_map.items()}


def _has_cpu_offload(device_map: dict[str, str] | None) -> bool:
    if not device_map:
        return False
    for value in device_map.values():
        lowered = str(value).lower()
        if "cpu" in lowered or "disk" in lowered or "meta" in lowered:
            return True
    return False


def _is_flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _select_torch_dtype(torch_module: Any) -> tuple[Any, str]:
    if torch_module.cuda.is_available():
        if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16, "bfloat16"
        return torch_module.float16, "float16"
    return torch_module.float32, "float32"


def _build_model_kwargs(
    *,
    torch_module: Any,
    dtype: Any,
    trust_remote_code: bool,
    prefer_flash_attention_2: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    if torch_module.cuda.is_available() and prefer_flash_attention_2 and _is_flash_attn_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def _load_tokenizer(tokenizer_source: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def _sanitize_model_generation_config(
    model: Any,
    tokenizer: Any,
    *,
    temperature: float,
) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.do_sample = bool(temperature > 0.0)
    generation_config.num_beams = 1
    generation_config.use_cache = True
    generation_config.repetition_penalty = 1.0
    for key, value in [("logits_processor", None), ("logits_warper", None)]:
        try:
            setattr(generation_config, key, value)
        except Exception:
            continue
    if temperature > 0.0:
        generation_config.temperature = float(temperature)
    else:
        for key in ["temperature", "top_p", "top_k", "min_p", "typical_p"]:
            try:
                setattr(generation_config, key, None)
            except Exception:
                continue


def _validate_transformers_generate_support(target_model: Any) -> None:
    import inspect

    candidate_models: list[Any] = [target_model]
    get_base_model = getattr(target_model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
            if base_model is not None and base_model is not target_model:
                candidate_models.append(base_model)
        except Exception:
            pass

    for model in candidate_models:
        try:
            signature = inspect.signature(model.generate)
        except Exception:
            continue
        if "assistant_model" in signature.parameters:
            return
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return
    raise RuntimeError("当前 transformers generate() 不支持 assistant_model 参数。")


def _load_model_bundle(
    *,
    model_path: str,
    dtype: Any,
    dtype_label: str,
    trust_remote_code: bool,
    prefer_flash_attention_2: bool,
    torch_module: Any,
    prefer_same_gpu: bool,
    preferred_cuda_device: int,
    allow_auto_device_map_fallback: bool,
) -> LoadedModelBundle:
    from transformers import AutoModelForCausalLM

    resolved_ref = _resolve_local_or_hf_ref(model_path)
    base_model_ref, adapter_path = _maybe_load_adapter(resolved_ref)
    base_model_kwargs = _build_model_kwargs(
        torch_module=torch_module,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    model_kwargs = dict(base_model_kwargs)

    def load_with_kwargs(load_kwargs: dict[str, Any]) -> Any:
        nonlocal attn_implementation
        try:
            return AutoModelForCausalLM.from_pretrained(base_model_ref, **load_kwargs)
        except Exception as exc:
            if load_kwargs.get("attn_implementation") != "flash_attention_2":
                raise
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            attn_implementation = "sdpa_or_default"
            return AutoModelForCausalLM.from_pretrained(base_model_ref, **fallback_kwargs)

    attn_implementation = str(model_kwargs.get("attn_implementation", "sdpa_or_default"))
    model = None
    load_strategy = "cpu"
    if torch_module.cuda.is_available() and prefer_same_gpu:
        target_device = torch_module.device(f"cuda:{preferred_cuda_device}")
        try:
            model = load_with_kwargs(model_kwargs)
            load_strategy = f"single_gpu:{target_device}"
        except Exception as exc:
            oom_like = isinstance(exc, torch_module.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not oom_like:
                raise
            _cleanup_runtime_state()
            if not allow_auto_device_map_fallback:
                raise RuntimeError("单 GPU 加载模型失败，并且已禁用 device_map=auto 回退。") from exc
            model_kwargs = dict(base_model_kwargs)
            model_kwargs["device_map"] = "auto"
            model = load_with_kwargs(model_kwargs)
            load_strategy = "auto_device_map_fallback"
    else:
        if torch_module.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            load_strategy = "auto_device_map"
        model = load_with_kwargs(model_kwargs)

    if adapter_path:
        from peft import PeftModel

        offload_dir = Path("/tmp/peft_offload") / Path(adapter_path).name
        offload_dir.mkdir(parents=True, exist_ok=True)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, offload_dir=str(offload_dir))

    if torch_module.cuda.is_available() and prefer_same_gpu and load_strategy.startswith("single_gpu:"):
        target_device = torch_module.device(f"cuda:{preferred_cuda_device}")
        model = model.to(target_device)

    model.eval()
    device_map = _extract_hf_device_map(model)
    return LoadedModelBundle(
        model=model,
        base_model_ref=base_model_ref,
        adapter_path=adapter_path,
        primary_device=_detect_model_primary_device(model),
        device_map=device_map,
        cpu_offload_detected=_has_cpu_offload(device_map),
        load_strategy=load_strategy,
        dtype_label=dtype_label,
        attn_implementation=attn_implementation,
    )


def _build_transformers_generation_kwargs(
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    assistant_model: Any | None,
    assistant_num_tokens: int | None,
    assistant_confidence_threshold: float,
    assistant_num_tokens_schedule: str,
) -> dict[str, Any]:
    from transformers import GenerationConfig

    base_generation_config = getattr(model, "generation_config", None)
    if base_generation_config is not None:
        generation_config = copy.deepcopy(base_generation_config)
    else:
        generation_config = GenerationConfig()
    if generation_config is None:
        generation_config = GenerationConfig()

    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.do_sample = bool(temperature > 0.0)
    generation_config.num_beams = 1
    generation_config.use_cache = True
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.repetition_penalty = 1.0
    for key, value in [("logits_processor", None), ("logits_warper", None)]:
        try:
            setattr(generation_config, key, value)
        except Exception:
            continue

    if temperature > 0.0:
        generation_config.temperature = float(temperature)
    else:
        for key in ["temperature", "top_p", "top_k", "min_p", "typical_p"]:
            try:
                setattr(generation_config, key, None)
            except Exception:
                continue

    kwargs = {
        "generation_config": generation_config,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
        kwargs["num_assistant_tokens"] = int(assistant_num_tokens or 0)
        kwargs["assistant_confidence_threshold"] = float(assistant_confidence_threshold)
        kwargs["num_assistant_tokens_schedule"] = str(assistant_num_tokens_schedule)
    return kwargs


def _decode_generated_batch(tokenizer: Any, output_ids: Any, attention_mask: Any) -> tuple[list[str], list[int]]:
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    predictions: list[str] = []
    output_token_counts: list[int] = []
    for row_idx, prompt_len in enumerate(prompt_lengths):
        generated_ids = output_ids[row_idx][int(prompt_len) :]
        output_token_counts.append(int(generated_ids.shape[0]))
        predictions.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return predictions, output_token_counts


def _build_sampling_params_for_vllm(
    *,
    max_new_tokens: int,
    temperature: float,
) -> Any:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
    if float(temperature) <= 0.0:
        kwargs["top_p"] = 1.0
    return SamplingParams(**kwargs)


def _resolve_vllm_runtime_model(raw_model_path: str) -> tuple[str, str | None]:
    resolved_ref = _resolve_local_or_hf_ref(raw_model_path)
    base_model_ref, adapter_path = _maybe_load_adapter(resolved_ref)
    if adapter_path is not None:
        raise RuntimeError(
            f"vLLM 对本实验仅支持 merged/base 模型目录，当前检测到 adapter 目录：{adapter_path}。"
        )
    return base_model_ref, None


def _load_vllm_engine(
    *,
    target_model_path: str,
    tokenizer_path: str,
    vllm_dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    trust_remote_code: bool,
    enforce_eager: bool,
    enable_prefix_caching: bool,
) -> Any:
    from vllm import LLM

    target_model_ref, _ = _resolve_vllm_runtime_model(target_model_path)
    llm_kwargs: dict[str, Any] = {
        "model": target_model_ref,
        "tokenizer": tokenizer_path,
        "trust_remote_code": trust_remote_code,
        "dtype": str(vllm_dtype),
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enforce_eager": bool(enforce_eager),
        "disable_log_stats": True,
        "enable_prefix_caching": bool(enable_prefix_caching),
    }
    if max_model_len is not None and int(max_model_len) > 0:
        llm_kwargs["max_model_len"] = int(max_model_len)
    return LLM(**llm_kwargs)


def _extract_vllm_prediction(output: Any) -> tuple[str, int]:
    candidates = getattr(output, "outputs", None) or []
    if not candidates:
        return "", 0
    first = candidates[0]
    text = str(getattr(first, "text", "") or "").strip()
    token_ids = getattr(first, "token_ids", None) or []
    return text, len(token_ids)


def _count_prompt_tokens_from_vllm_output(output: Any, tokenizer: Any, prompt: str) -> int:
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    if isinstance(prompt_token_ids, list):
        return len(prompt_token_ids)
    return int(len(tokenizer.encode(prompt, add_special_tokens=False)))


def _start_vram_monitor(gpu_id: int) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    stop_event.gpu_id = gpu_id
    stop_event.poll_interval_sec = VRAM_POLL_INTERVAL_SEC
    thread = threading.Thread(target=monitor_vram, args=(stop_event,), daemon=True)
    thread.start()
    return stop_event, thread


def _stop_vram_monitor(stop_event: threading.Event, thread: threading.Thread) -> tuple[float, str | None]:
    stop_event.set()
    thread.join(timeout=5)
    peak_vram_mb = float(getattr(stop_event, "peak_vram_mb", 0.0) or 0.0)
    monitor_error = getattr(stop_event, "monitor_error", None)
    return peak_vram_mb, monitor_error


def run_transformers_case(
    *,
    case: dict[str, Any],
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    gpu_id: int,
) -> dict[str, Any]:
    import torch

    if int(config["batch_size"]) != 1:
        raise ValueError("当前实验为保证公平性与兼容性，batch_size 仅支持 1。")

    dtype, dtype_label = _select_torch_dtype(torch)
    preferred_cuda_device = int(config.get("preferred_cuda_device", 0))
    decoding_mode = str(case["decoding_mode"])
    use_speculative = decoding_mode == "speculative"
    prefer_same_gpu = bool(config.get("prefer_same_gpu", True))
    allow_auto_device_map_fallback = bool(config.get("allow_auto_device_map_fallback", True))
    assistant_confidence_threshold = float(config.get("assistant_confidence_threshold", 0.55))
    assistant_num_tokens_schedule = str(config.get("assistant_num_tokens_schedule", "constant"))
    started_at = time.perf_counter()
    stop_event, thread = _start_vram_monitor(gpu_id)
    target_bundle = None
    assistant_bundle = None
    try:
        target_bundle = _load_model_bundle(
            model_path=str(config["target_model_path"]),
            dtype=dtype,
            dtype_label=dtype_label,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            prefer_flash_attention_2=bool(config.get("prefer_flash_attention_2", True)),
            torch_module=torch,
            prefer_same_gpu=prefer_same_gpu,
            preferred_cuda_device=preferred_cuda_device,
            allow_auto_device_map_fallback=allow_auto_device_map_fallback,
        )
        _sanitize_model_generation_config(target_bundle.model, tokenizer, temperature=float(config["temperature"]))
        if use_speculative:
            assistant_bundle = _load_model_bundle(
                model_path=str(config["assistant_model_path"]),
                dtype=dtype,
                dtype_label=dtype_label,
                trust_remote_code=bool(config.get("trust_remote_code", True)),
                prefer_flash_attention_2=bool(config.get("prefer_flash_attention_2", True)),
                torch_module=torch,
                prefer_same_gpu=prefer_same_gpu,
                preferred_cuda_device=preferred_cuda_device,
                allow_auto_device_map_fallback=allow_auto_device_map_fallback,
            )
            _sanitize_model_generation_config(assistant_bundle.model, tokenizer, temperature=float(config["temperature"]))
            _validate_transformers_generate_support(target_bundle.model)
        generation_kwargs = _build_transformers_generation_kwargs(
            model=target_bundle.model,
            tokenizer=tokenizer,
            max_new_tokens=int(config["max_new_tokens"]),
            temperature=float(config["temperature"]),
            assistant_model=assistant_bundle.model if assistant_bundle is not None else None,
            assistant_num_tokens=int(config["assistant_num_tokens"]) if use_speculative else None,
            assistant_confidence_threshold=assistant_confidence_threshold,
            assistant_num_tokens_schedule=assistant_num_tokens_schedule,
        )

        warmup_samples = min(int(config.get("warmup_samples", 0)), len(dataset))
        for item in dataset[:warmup_samples]:
            prompt = _render_chat_prompt(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                instruction=str(item["instruction"]),
                input_text=str(item["input"]),
            )
            inputs = tokenizer([prompt], return_tensors="pt", padding=True)
            target_device = (
                target_bundle.model.device
                if hasattr(target_bundle.model, "device") and target_bundle.model.device is not None
                else next(target_bundle.model.parameters()).device
            )
            input_ids = inputs["input_ids"].to(target_device)
            attention_mask = inputs["attention_mask"].to(target_device)
            with torch.no_grad():
                _ = target_bundle.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sample_rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        parse_ok_count = 0

        for batch in _chunked(dataset, 1):
            item = batch[0]
            prompt = _render_chat_prompt(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                instruction=str(item["instruction"]),
                input_text=str(item["input"]),
            )
            inputs = tokenizer([prompt], return_tensors="pt", padding=True)
            input_token_count = int(inputs["attention_mask"].sum().item())
            target_device = (
                target_bundle.model.device
                if hasattr(target_bundle.model, "device") and target_bundle.model.device is not None
                else next(target_bundle.model.parameters()).device
            )
            input_ids = inputs["input_ids"].to(target_device)
            attention_mask = inputs["attention_mask"].to(target_device)

            batch_started_at = time.perf_counter()
            with torch.no_grad():
                output_ids = target_bundle.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_latency = time.perf_counter() - batch_started_at
            predictions, output_token_counts = _decode_generated_batch(tokenizer, output_ids, attention_mask)
            prediction = predictions[0]
            output_token_count = int(output_token_counts[0])
            parse_ok, parse_error = _try_parse_json(prediction)

            total_input_tokens += input_token_count
            total_output_tokens += output_token_count
            if parse_ok:
                parse_ok_count += 1
            latencies.append(batch_latency)
            sample_rows.append(
                {
                    "sample_index": int(item["index"]),
                    "instruction": str(item["instruction"]),
                    "input": str(item["input"]),
                    "prediction": prediction,
                    "parse_ok": bool(parse_ok),
                    "parse_error": parse_error,
                    "input_tokens": int(input_token_count),
                    "output_tokens": int(output_token_count),
                    "latency_sec": _round_float(batch_latency),
                }
            )

        total_runtime = float(sum(latencies))
        peak_vram_mb, monitor_error = _stop_vram_monitor(stop_event, thread)
        return {
            "name": str(case["name"]),
            "backend": "transformers",
            "decoding_mode": decoding_mode,
            "decoding_mode_label": str(case["decoding_mode_label"]),
            "scenario_label": str(case["scenario_label"]),
            "plot_scenario_label": str(case["plot_scenario_label"]),
            "experiment_profile": str(case["experiment_profile"]),
            "speculative_method": str(case["speculative_method"]),
            "execution_note": str(case["execution_note"]),
            "precision": dtype_label,
            "assistant_num_tokens": int(config["assistant_num_tokens"]) if use_speculative else 0,
            "assistant_confidence_threshold": (assistant_confidence_threshold if use_speculative else None),
            "assistant_num_tokens_schedule": (assistant_num_tokens_schedule if use_speculative else None),
            "num_samples": int(len(sample_rows)),
            "batch_size": 1,
            "max_new_tokens": int(config["max_new_tokens"]),
            "temperature": float(config["temperature"]),
            "status": "success",
            "error_message": None,
            "monitor_error": monitor_error,
            "target_primary_device": target_bundle.primary_device,
            "assistant_primary_device": assistant_bundle.primary_device if assistant_bundle is not None else None,
            "target_cpu_offload_detected": target_bundle.cpu_offload_detected,
            "assistant_cpu_offload_detected": assistant_bundle.cpu_offload_detected if assistant_bundle is not None else None,
            "target_load_strategy": target_bundle.load_strategy,
            "assistant_load_strategy": assistant_bundle.load_strategy if assistant_bundle is not None else "disabled",
            "avg_latency_sec_per_sample": _round_float(_safe_mean(latencies)),
            "token_throughput_tps": _round_float(_safe_div(total_output_tokens, total_runtime), 3),
            "peak_vram_mb": _round_float(peak_vram_mb, 3),
            "parse_ok_rate": _round_float(_safe_div(parse_ok_count, len(sample_rows)), 4),
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "avg_input_tokens": _round_float(_safe_div(total_input_tokens, len(sample_rows)), 3),
            "avg_output_tokens": _round_float(_safe_div(total_output_tokens, len(sample_rows)), 3),
            "wall_clock_sec_including_load": _round_float(time.perf_counter() - started_at, 3),
            "samples": sample_rows,
        }
    finally:
        if thread.is_alive():
            _stop_vram_monitor(stop_event, thread)
        _cleanup_runtime_state()
        if assistant_bundle is not None:
            try:
                del assistant_bundle.model
            except Exception:
                pass
        if target_bundle is not None:
            try:
                del target_bundle.model
            except Exception:
                pass
        _cleanup_runtime_state()
        time.sleep(float(config.get("post_case_sleep_seconds", POST_CASE_SLEEP_SECONDS)))


def run_vllm_case(
    *,
    case: dict[str, Any],
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    gpu_id: int,
) -> dict[str, Any]:
    decoding_mode = str(case["decoding_mode"])
    if decoding_mode != "standard":
        raise ValueError("当前 exp11 不再支持 vLLM speculative decoding。")
    vllm_dtype = str(config.get("vllm_dtype", DEFAULT_VLLM_DTYPE))
    resolved_gpu_memory_utilization = resolve_vllm_gpu_memory_utilization(
        config=config,
        gpu_id=gpu_id,
        total_gpu_memory_mb=query_total_gpu_memory_mb(gpu_id),
    )
    sampling_params = _build_sampling_params_for_vllm(
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=float(config["temperature"]),
    )
    stop_event, thread = _start_vram_monitor(gpu_id)
    llm = None
    started_at = time.perf_counter()
    try:
        tokenizer_source = _maybe_load_adapter(_resolve_local_or_hf_ref(str(config["target_model_path"])))[0]
        llm = _load_vllm_engine(
            target_model_path=str(config["target_model_path"]),
            tokenizer_path=tokenizer_source,
            vllm_dtype=vllm_dtype,
            gpu_memory_utilization=resolved_gpu_memory_utilization,
            max_model_len=(int(config["max_model_len"]) if config.get("max_model_len") is not None else None),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            enforce_eager=bool(config.get("vllm_enforce_eager", False)),
            enable_prefix_caching=bool(config.get("vllm_enable_prefix_caching", False)),
        )

        warmup_samples = min(int(config.get("warmup_samples", 0)), len(dataset))
        for item in dataset[:warmup_samples]:
            prompt = _render_chat_prompt(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                instruction=str(item["instruction"]),
                input_text=str(item["input"]),
            )
            _ = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)

        sample_rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        parse_ok_count = 0

        for batch in _chunked(dataset, int(config["batch_size"])):
            prompts = [
                _render_chat_prompt(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    instruction=str(item["instruction"]),
                    input_text=str(item["input"]),
                )
                for item in batch
            ]
            batch_started_at = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            batch_latency = time.perf_counter() - batch_started_at
            batch_latency_per_sample = _safe_div(batch_latency, len(batch))
            latencies.append(batch_latency_per_sample)

            for item, prompt, output in zip(batch, prompts, outputs):
                prediction, output_token_count = _extract_vllm_prediction(output)
                input_token_count = _count_prompt_tokens_from_vllm_output(output, tokenizer, prompt)
                parse_ok, parse_error = _try_parse_json(prediction)
                total_input_tokens += int(input_token_count)
                total_output_tokens += int(output_token_count)
                if parse_ok:
                    parse_ok_count += 1
                sample_rows.append(
                    {
                        "sample_index": int(item["index"]),
                        "instruction": str(item["instruction"]),
                        "input": str(item["input"]),
                        "prediction": prediction,
                        "parse_ok": bool(parse_ok),
                        "parse_error": parse_error,
                        "input_tokens": int(input_token_count),
                        "output_tokens": int(output_token_count),
                        "latency_sec": _round_float(batch_latency_per_sample),
                    }
                )

        total_runtime = float(sum(float(row["latency_sec"]) for row in sample_rows))
        peak_vram_mb, monitor_error = _stop_vram_monitor(stop_event, thread)
        return {
            "name": str(case["name"]),
            "backend": "vllm",
            "decoding_mode": decoding_mode,
            "decoding_mode_label": str(case["decoding_mode_label"]),
            "scenario_label": str(case["scenario_label"]),
            "plot_scenario_label": str(case["plot_scenario_label"]),
            "experiment_profile": str(case["experiment_profile"]),
            "speculative_method": str(case["speculative_method"]),
            "execution_note": str(case["execution_note"]),
            "precision": vllm_dtype,
            "assistant_num_tokens": 0,
            "assistant_confidence_threshold": None,
            "assistant_num_tokens_schedule": None,
            "vllm_gpu_memory_utilization": _round_float(resolved_gpu_memory_utilization, 4),
            "num_samples": int(len(sample_rows)),
            "batch_size": int(config["batch_size"]),
            "max_new_tokens": int(config["max_new_tokens"]),
            "temperature": float(config["temperature"]),
            "status": "success",
            "error_message": None,
            "monitor_error": monitor_error,
            "target_primary_device": f"cuda:{int(config.get('preferred_cuda_device', 0))}",
            "assistant_primary_device": None,
            "target_cpu_offload_detected": False,
            "assistant_cpu_offload_detected": None,
            "target_load_strategy": f"vllm_standard@dtype={vllm_dtype}@util={resolved_gpu_memory_utilization:.4f}",
            "assistant_load_strategy": "disabled",
            "avg_latency_sec_per_sample": _round_float(_safe_mean([float(row["latency_sec"]) for row in sample_rows])),
            "token_throughput_tps": _round_float(_safe_div(total_output_tokens, total_runtime), 3),
            "peak_vram_mb": _round_float(peak_vram_mb, 3),
            "parse_ok_rate": _round_float(_safe_div(parse_ok_count, len(sample_rows)), 4),
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "avg_input_tokens": _round_float(_safe_div(total_input_tokens, len(sample_rows)), 3),
            "avg_output_tokens": _round_float(_safe_div(total_output_tokens, len(sample_rows)), 3),
            "wall_clock_sec_including_load": _round_float(time.perf_counter() - started_at, 3),
            "samples": sample_rows,
        }
    finally:
        if thread.is_alive():
            _stop_vram_monitor(stop_event, thread)
        _maybe_shutdown_vllm_engine(llm)
        try:
            del llm
        except Exception:
            pass
        _cleanup_runtime_state()
        time.sleep(float(config.get("post_case_sleep_seconds", POST_CASE_SLEEP_SECONDS)))


def execute_case(
    *,
    case: dict[str, Any],
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    gpu_id: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        if str(case["backend"]) == "vllm":
            return run_vllm_case(
                case=case,
                config=config,
                dataset=dataset,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                gpu_id=gpu_id,
            )
        if str(case["backend"]) == "transformers":
            return run_transformers_case(
                case=case,
                config=config,
                dataset=dataset,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                gpu_id=gpu_id,
            )
        raise ValueError(f"未知 backend={case['backend']!r}")
    except Exception as exc:
        error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return {
            "name": str(case["name"]),
            "backend": str(case["backend"]),
            "decoding_mode": str(case["decoding_mode"]),
            "decoding_mode_label": str(case["decoding_mode_label"]),
            "scenario_label": str(case["scenario_label"]),
            "plot_scenario_label": str(case["plot_scenario_label"]),
            "experiment_profile": str(case["experiment_profile"]),
            "speculative_method": str(case["speculative_method"]),
            "execution_note": str(case["execution_note"]),
            "precision": None,
            "assistant_num_tokens": int(config["assistant_num_tokens"]) if case["decoding_mode"] == "speculative" else 0,
            "assistant_confidence_threshold": (
                float(config.get("assistant_confidence_threshold", 0.55))
                if case["backend"] == "transformers" and case["decoding_mode"] == "speculative"
                else None
            ),
            "assistant_num_tokens_schedule": (
                str(config.get("assistant_num_tokens_schedule", "constant"))
                if case["backend"] == "transformers" and case["decoding_mode"] == "speculative"
                else None
            ),
            "vllm_gpu_memory_utilization": (
                _round_float(resolve_vllm_gpu_memory_utilization(
                    config=config,
                    gpu_id=gpu_id,
                    total_gpu_memory_mb=query_total_gpu_memory_mb(gpu_id),
                ), 4)
                if case["backend"] == "vllm"
                else None
            ),
            "num_samples": 0,
            "batch_size": int(config["batch_size"]),
            "max_new_tokens": int(config["max_new_tokens"]),
            "temperature": float(config["temperature"]),
            "status": _classify_error_status(exc),
            "error_message": error_message,
            "monitor_error": None,
            "target_primary_device": None,
            "assistant_primary_device": None,
            "target_cpu_offload_detected": None,
            "assistant_cpu_offload_detected": None,
            "target_load_strategy": None,
            "assistant_load_strategy": None,
            "avg_latency_sec_per_sample": 0.0,
            "token_throughput_tps": 0.0,
            "peak_vram_mb": 0.0,
            "parse_ok_rate": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "wall_clock_sec_including_load": _round_float(time.perf_counter() - started_at, 3),
            "samples": [],
        }


def build_backend_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    by_backend: dict[str, dict[str, Any]] = {}
    for item in results:
        if str(item.get("status")) != "success":
            continue
        by_backend.setdefault(str(item["backend"]), {})[str(item["decoding_mode"])] = item

    for backend, mode_group in by_backend.items():
        standard_item = mode_group.get("standard")
        speculative_item = mode_group.get("speculative")
        if not standard_item or not speculative_item:
            continue
        comparisons[backend] = {
            "speculative_latency_speedup_vs_standard": _round_float(
                _safe_div(
                    float(standard_item["avg_latency_sec_per_sample"]),
                    float(speculative_item["avg_latency_sec_per_sample"]),
                ),
                4,
            ),
            "speculative_throughput_gain_pct_vs_standard": _round_float(
                _safe_div(
                    float(speculative_item["token_throughput_tps"]) - float(standard_item["token_throughput_tps"]),
                    float(standard_item["token_throughput_tps"]),
                )
                * 100.0,
                4,
            ),
            "speculative_peak_vram_delta_mb_vs_standard": _round_float(
                float(speculative_item["peak_vram_mb"]) - float(standard_item["peak_vram_mb"]),
                3,
            ),
            "speculative_parse_ok_rate_delta_vs_standard": _round_float(
                float(speculative_item["parse_ok_rate"]) - float(standard_item["parse_ok_rate"]),
                4,
            ),
        }
    return comparisons


def build_summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "Name": item["name"],
                "Backend": item["backend"],
                "Decoding Mode": item["decoding_mode"],
                "Decoding Mode Label": item["decoding_mode_label"],
                "Speculative Method": item["speculative_method"],
                "Scenario Label": item["scenario_label"],
                "Plot Scenario Label": item["plot_scenario_label"],
                "Experiment Profile": item["experiment_profile"],
                "Precision": item["precision"],
                "Avg Latency (s)": item["avg_latency_sec_per_sample"],
                "Token Throughput (tokens/s)": item["token_throughput_tps"],
                "Peak VRAM (MB)": item["peak_vram_mb"],
                "Parse OK Rate": item["parse_ok_rate"],
                "Status": item["status"],
                "Assistant Draft Tokens": item["assistant_num_tokens"],
                "Execution Note": item["execution_note"],
            }
        )
    return rows


def write_markdown_report(
    *,
    results: list[dict[str, Any]],
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    scenario_labels = [str(item["scenario_label"]) for item in results]
    lines: list[str] = [
        "# Exp11 投机解码引擎对比报告",
        "",
        "## 实验目标",
        "",
        "- 将 `exp11` 的主模型、助手模型、数据集、样本数、温度、最大生成长度与预热设置对齐到 `exp8`。",
        "- `Transformers` 侧继续使用 `assistant_model` assisted generation；`vLLM` 侧仅保留基线解码，并固定使用 `float16` 精度加载 3B 模型。",
        "- 当前场景包括：" + "、".join(f"`{label}`" for label in scenario_labels) + "。",
        "",
        "## 结果总表",
        "",
        "| Case | Backend | Mode | Method | Precision | Avg Latency (s) | Tokens/s | Peak VRAM (MB) | Parse OK Rate | Status |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            "| {name} | {backend} | {mode} | {method} | {precision} | {latency:.4f} | {tps:.3f} | {vram:.1f} | {parse:.4f} | {status} |".format(
                name=str(item["name"]),
                backend=str(item["backend"]),
                mode=str(item["decoding_mode_label"]),
                method=str(item["speculative_method"]),
                precision=str(item.get("precision") or "-"),
                latency=float(item["avg_latency_sec_per_sample"]),
                tps=float(item["token_throughput_tps"]),
                vram=float(item["peak_vram_mb"]),
                parse=float(item["parse_ok_rate"]),
                status=str(item["status"]),
            )
        )

    lines.extend(["", "## 口径说明", ""])
    appended_notes: set[str] = set()
    for item in results:
        note = str(item["execution_note"])
        if note in appended_notes:
            continue
        appended_notes.add(note)
        lines.append(f"- {note}")

    lines.extend(["", "## 后端结论", ""])
    if comparison:
        for backend, metrics in comparison.items():
            lines.extend(
                [
                    f"### {backend}",
                    "",
                    (
                        f"- 投机解码相对无投机的延迟加速比为 "
                        f"`{float(metrics['speculative_latency_speedup_vs_standard']):.4f}x`。"
                    ),
                    (
                        f"- 投机解码相对无投机的 token 吞吐变化为 "
                        f"`{float(metrics['speculative_throughput_gain_pct_vs_standard']):.2f}%`。"
                    ),
                    (
                        f"- 投机解码相对无投机的峰值显存变化为 "
                        f"`{float(metrics['speculative_peak_vram_delta_mb_vs_standard']):.1f} MB`。"
                    ),
                ]
            )
    else:
        lines.append("- 当前没有形成可比的成功结果对。")

    failed_rows = [item for item in results if str(item.get("status")) != "success"]
    if failed_rows:
        lines.extend(["", "## 失败情况", ""])
        for item in failed_rows:
            lines.append(f"- `{item['name']}` 失败，状态为 `{item['status']}`，原因：`{item.get('error_message') or '未知错误'}`。")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def draw_summary_figures(df: pd.DataFrame, *, output_dir: Path) -> None:
    if df.empty:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((EXPERIMENT_DIR / ".cache" / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_report_matplotlib(matplotlib)
    ensure_dir(output_dir)

    success_df = df[df["Status"] == "success"].copy()
    if success_df.empty:
        return
    success_df["Group Label"] = success_df["Plot Scenario Label"]
    color_map = {
        ("transformers", "standard"): "#4c78a8",
        ("transformers", "speculative"): "#f58518",
        ("vllm", "standard"): "#54a24b",
        ("vllm", "speculative"): "#e45756",
    }
    colors = [
        color_map.get((str(row["Backend"]), str(row["Decoding Mode"])), "#72b7b2")
        for _, row in success_df.iterrows()
    ]

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.bar(success_df["Group Label"], success_df["Avg Latency (s)"], color=colors)
    ax1.set_title(pick_plot_text("图1：无投机与投机解码平均延迟对比", "Figure 1: Standard vs Speculative Latency"))
    ax1.set_xlabel(pick_plot_text("方案", "Scenario"))
    ax1.set_ylabel("Seconds")
    ax1.tick_params(axis="x", rotation=15)
    fig1.tight_layout()
    fig1.savefig(output_dir / "exp11_speculative_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.bar(success_df["Group Label"], success_df["Token Throughput (tokens/s)"], color=colors)
    ax2.set_title(pick_plot_text("图2：无投机与投机解码吞吐对比", "Figure 2: Standard vs Speculative Throughput"))
    ax2.set_xlabel(pick_plot_text("方案", "Scenario"))
    ax2.set_ylabel("Tokens / Second")
    ax2.tick_params(axis="x", rotation=15)
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp11_speculative_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.bar(success_df["Group Label"], success_df["Peak VRAM (MB)"], color=colors)
    ax3.set_title(pick_plot_text("图3：无投机与投机解码峰值显存对比", "Figure 3: Standard vs Speculative Peak VRAM"))
    ax3.set_xlabel(pick_plot_text("方案", "Scenario"))
    ax3.set_ylabel("MB")
    ax3.tick_params(axis="x", rotation=15)
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp11_speculative_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_config = _load_runtime_config(args)
    ensure_dir(REPORTS_DIR)
    ensure_dir(LOGS_DIR)

    gpu_id = infer_gpu_id(args.gpu_id)
    total_gpu_memory_mb = int(args.total_gpu_memory_mb) if args.total_gpu_memory_mb else query_total_gpu_memory_mb(gpu_id)

    report_path = _resolve_report_path(runtime_config.get("report_path"))
    ensure_dir(report_path.parent)

    backends = _parse_backend_list(runtime_config.get("backends"), DEFAULT_BACKENDS)
    decoding_modes = _parse_decoding_mode_list(runtime_config.get("decoding_modes"), DEFAULT_DECODING_MODES)
    runtime_config["backends"] = ",".join(backends)
    runtime_config["decoding_modes"] = ",".join(decoding_modes)
    runtime_config["assistant_num_tokens"] = int(runtime_config.get("assistant_num_tokens", 8))
    runtime_config["assistant_confidence_threshold"] = float(runtime_config.get("assistant_confidence_threshold", 0.55))
    runtime_config["assistant_num_tokens_schedule"] = str(
        runtime_config.get("assistant_num_tokens_schedule", "constant")
    )
    runtime_config["post_case_sleep_seconds"] = int(runtime_config.get("post_case_sleep_seconds", POST_CASE_SLEEP_SECONDS))

    system_prompt = _load_system_prompt(args.base_config)
    dataset_path = _resolve_dataset_path(runtime_config["dataset_path"])
    dataset = _load_dataset(dataset_path, int(runtime_config["num_samples"]))
    tokenizer_source = _maybe_load_adapter(_resolve_local_or_hf_ref(str(runtime_config["target_model_path"])))[0]
    tokenizer = _load_tokenizer(
        tokenizer_source=tokenizer_source,
        trust_remote_code=bool(runtime_config.get("trust_remote_code", True)),
    )
    cases = build_case_matrix(
        backends=backends,
        decoding_modes=decoding_modes,
        experiment_profile=str(runtime_config.get("experiment_profile", DEFAULT_EXPERIMENT_PROFILE)),
    )

    print(
        "[INFO] 开始实验15：profile={profile}，后端={backends}，assistant_num_tokens={draft_tokens}，样本数={num_samples}".format(
            profile=str(runtime_config.get("experiment_profile", DEFAULT_EXPERIMENT_PROFILE)),
            backends=",".join(backends),
            draft_tokens=int(runtime_config["assistant_num_tokens"]),
            num_samples=len(dataset),
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        print(
            "[INFO] 运行 case={name} backend={backend} method={method}".format(
                name=case["name"],
                backend=case["backend"],
                method=case["speculative_method"],
            ),
            flush=True,
        )
        result = execute_case(
            case=case,
            config=runtime_config,
            dataset=dataset,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            gpu_id=gpu_id,
        )
        results.append(result)
        print(
            "[INFO] case={name} status={status} latency={latency:.4f}s tps={tps:.3f} peak_vram={vram:.1f}MB parse_ok={parse:.4f}".format(
                name=result["name"],
                status=result["status"],
                latency=float(result["avg_latency_sec_per_sample"]),
                tps=float(result["token_throughput_tps"]),
                vram=float(result["peak_vram_mb"]),
                parse=float(result["parse_ok_rate"]),
            ),
            flush=True,
        )

    comparison = build_backend_comparison(results)
    summary_rows = build_summary_rows(results)
    summary_df = pd.DataFrame(summary_rows)
    csv_path = report_path.parent / "exp11_speculative_engine_compare.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path = report_path.parent / "exp11_speculative_engine_compare.md"
    write_markdown_report(results=results, comparison=comparison, output_path=markdown_path)
    draw_summary_figures(summary_df, output_dir=report_path.parent)

    environment: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_id": gpu_id,
        "total_gpu_memory_mb": int(total_gpu_memory_mb) if total_gpu_memory_mb else None,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
        "vllm_dtype": str(runtime_config.get("vllm_dtype", DEFAULT_VLLM_DTYPE)),
        "vllm_gpu_memory_utilization": float(runtime_config.get("vllm_gpu_memory_utilization", 0.7)),
        "vllm_enforce_eager": bool(runtime_config.get("vllm_enforce_eager", False)),
        "vllm_enable_prefix_caching": bool(runtime_config.get("vllm_enable_prefix_caching", False)),
    }
    try:
        import torch
        import transformers

        environment["torch_version"] = getattr(torch, "__version__", None)
        environment["transformers_version"] = getattr(transformers, "__version__", None)
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["flash_attention_2_available"] = _is_flash_attn_available()
    except Exception:
        pass
    try:
        import vllm

        environment["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception:
        environment["vllm_version"] = None

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": {
            "name": "exp11_speculative_engine_compare",
            "entry": "experiments/15_exp11_speculative_engine_compare/run_exp11_speculative_engine_compare.py",
        },
        "config": {
            **runtime_config,
            "resolved_dataset_path": str(dataset_path),
            "tokenizer_source": tokenizer_source,
            "system_prompt": system_prompt,
        },
        "environment": environment,
        "results": results,
        "comparison": comparison,
        "artifacts": {
            "csv_path": str(csv_path.resolve()),
            "markdown_path": str(markdown_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = record_run_meta(
        report_path.parent,
        merged_config={"speculative_engine_compare": runtime_config},
        cli_args=vars(args),
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        data_paths=[dataset_path],
        extra_meta={
            "entry": "experiments/15_exp11_speculative_engine_compare/run_exp11_speculative_engine_compare.py",
            "stage": "speculative_engine_compare",
            "report_path": str(report_path.resolve()),
            "gpu_id": gpu_id,
            "total_gpu_memory_mb": total_gpu_memory_mb,
        },
    )

    print(f"[OK] JSON 报告已写入：{report_path}", flush=True)
    print(f"[OK] CSV 汇总已写入：{csv_path}", flush=True)
    print(f"[OK] Markdown 报告已写入：{markdown_path}", flush=True)
    print(f"[OK] 运行元数据已写入：{meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
