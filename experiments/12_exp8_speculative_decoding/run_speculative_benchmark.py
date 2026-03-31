#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "speculative.yaml"
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.performance_monitor import time_and_memory_tracker
from src.utils.config import load_config, load_merged_config
from src.utils.run_meta import record_run_meta


DEFAULT_SYSTEM_PROMPT = (
    "你是 Franka 机械臂控制指令生成器。请把用户自然语言转换为可执行的 JSON action。"
    "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
    "只输出 JSON，不要输出解释。"
)


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
    dtype_label: str
    attn_implementation: str
    primary_device: str
    device_map: dict[str, str] | None
    cpu_offload_detected: bool
    load_strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验12 Exp8：Speculative Decoding 推理加速基准。")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="实验配置文件路径。",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG_PATH,
        help="全局基础配置文件路径，用于复用系统提示词等通用配置。",
    )
    parser.add_argument("--target-model-path", type=str, default=None, help="主模型路径或 Hugging Face Repo ID。")
    parser.add_argument("--assistant-model-path", type=str, default=None, help="草稿模型路径或 Hugging Face Repo ID。")
    parser.add_argument("--dataset-path", type=str, default=None, help="测试集 JSON 路径。")
    parser.add_argument("--num-samples", type=int, default=None, help="实际评测样本数。")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小，默认读取配置。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数。")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度。投机解码建议使用 0.0。")
    parser.add_argument("--warmup-samples", type=int, default=None, help="正式计时前的预热样本数。")
    parser.add_argument("--assistant-num-tokens", type=int, default=None, help="每轮 speculative 草稿 token 上限。")
    parser.add_argument(
        "--assistant-confidence-threshold",
        type=float,
        default=None,
        help="assistant 置信度阈值，越高越保守。",
    )
    parser.add_argument(
        "--assistant-num-tokens-schedule",
        type=str,
        default=None,
        help="assistant token 步长调度策略，如 constant / heuristic / heuristic_transient。",
    )
    parser.add_argument(
        "--preferred-cuda-device",
        type=int,
        default=None,
        help="优先使用的 CUDA 设备编号，默认 0。",
    )
    parser.add_argument(
        "--disable-same-gpu-placement",
        action="store_true",
        help="关闭“主模型与草稿模型优先放在同一张 GPU”策略。",
    )
    parser.add_argument(
        "--disable-auto-device-map-fallback",
        action="store_true",
        help="关闭单卡加载失败后的 device_map=auto 回退。",
    )
    parser.add_argument(
        "--disable-assistant",
        action="store_true",
        help="关闭 assistant_model，仅运行基线 greedy decoding。",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="结果 JSON 输出路径，默认写入 experiments/12_exp8_speculative_decoding/reports/speculative_benchmark.json。",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


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


def _maybe_load_adapter(model_ref: ResolvedModelRef) -> tuple[str, str | None]:
    if not model_ref.is_local:
        return model_ref.resolved, None

    adapter_config_path = Path(model_ref.resolved) / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_ref.resolved, None

    try:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"无法解析 LoRA adapter 配置文件: {adapter_config_path}") from exc

    if not isinstance(adapter_config, dict):
        raise ValueError(f"LoRA adapter 配置格式非法: {adapter_config_path}")

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
        return (REPORTS_DIR / "speculative_benchmark.json").resolve()

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (EXPERIMENT_DIR / candidate).resolve()


def _is_flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _select_torch_dtype(torch_module: Any) -> tuple[Any, str]:
    if torch_module.cuda.is_available():
        if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16, "bfloat16"
        return torch_module.float16, "float16"
    return torch_module.float32, "float32"


def _build_model_kwargs(*, torch_module: Any, dtype: Any, trust_remote_code: bool, prefer_flash_attention_2: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if torch_module.cuda.is_available():
        kwargs["dtype"] = dtype
        if prefer_flash_attention_2 and _is_flash_attn_available():
            kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["dtype"] = dtype
    return kwargs


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
    return {str(k): str(v) for k, v in raw_map.items()}


def _has_cpu_offload(device_map: dict[str, str] | None) -> bool:
    if not device_map:
        return False
    for value in device_map.values():
        text = str(value).lower()
        if "cpu" in text or "disk" in text or "meta" in text:
            return True
    return False


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

    attn_implementation = str(model_kwargs.get("attn_implementation", "sdpa_or_default"))
    model = None
    load_strategy = "cpu"

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
            print(
                f"[WARN] 模型 {base_model_ref} 无法启用 flash_attention_2，已自动回退默认注意力实现: {exc}",
                flush=True,
            )
            return AutoModelForCausalLM.from_pretrained(base_model_ref, **fallback_kwargs)

    if torch_module.cuda.is_available() and prefer_same_gpu:
        target_device = torch_module.device(f"cuda:{preferred_cuda_device}")
        try:
            model = load_with_kwargs(model_kwargs)
            load_strategy = f"single_gpu:{target_device}"
        except Exception as exc:
            oom_like = isinstance(exc, torch_module.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not oom_like:
                raise
            _cleanup_models(model)
            if not allow_auto_device_map_fallback:
                raise RuntimeError(
                    "单 GPU 加载模型失败，并且已禁用 device_map=auto 回退。"
                ) from exc
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
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"检测到 LoRA adapter 目录 {adapter_path}，但当前环境缺少 peft 依赖。"
            ) from exc
        offload_dir = Path("/tmp/peft_offload") / Path(adapter_path).name
        offload_dir.mkdir(parents=True, exist_ok=True)
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            offload_dir=str(offload_dir),
        )

    if torch_module.cuda.is_available() and prefer_same_gpu and load_strategy.startswith("single_gpu:"):
        target_device = torch_module.device(f"cuda:{preferred_cuda_device}")
        model = model.to(target_device)

    model.eval()
    device_map = _extract_hf_device_map(model)
    primary_device = _detect_model_primary_device(model)
    cpu_offload_detected = _has_cpu_offload(device_map)
    return LoadedModelBundle(
        model=model,
        base_model_ref=base_model_ref,
        adapter_path=adapter_path,
        dtype_label=dtype_label,
        attn_implementation=attn_implementation,
        primary_device=primary_device,
        device_map=device_map,
        cpu_offload_detected=cpu_offload_detected,
        load_strategy=load_strategy,
    )


def _set_generation_tokens(model: Any, tokenizer: Any) -> None:
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        if pad_token_id is not None:
            model.generation_config.pad_token_id = pad_token_id
        if eos_token_id is not None:
            model.generation_config.eos_token_id = eos_token_id
    if getattr(model, "config", None) is not None and pad_token_id is not None:
        model.config.pad_token_id = pad_token_id


def _sanitize_model_generation_config(
    model: Any,
    tokenizer: Any,
    *,
    temperature: float,
    repetition_penalty: float,
) -> None:
    _set_generation_tokens(model, tokenizer)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.do_sample = bool(temperature > 0.0)
    generation_config.num_beams = 1
    generation_config.use_cache = True
    generation_config.repetition_penalty = float(repetition_penalty)
    for key, value in [("logits_processor", None), ("logits_warper", None)]:
        try:
            setattr(generation_config, key, value)
        except Exception:
            continue

    if temperature > 0.0:
        generation_config.temperature = float(temperature)
    else:
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "penalty_alpha",
        ]:
            try:
                setattr(generation_config, key, None)
            except Exception:
                continue


def _build_generation_kwargs(
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    assistant_enabled: bool,
    assistant_model: Any | None,
    assistant_num_tokens: int,
    assistant_confidence_threshold: float,
    assistant_num_tokens_schedule: str,
) -> dict[str, Any]:
    from transformers import GenerationConfig

    base_generation_config = getattr(model, "generation_config", None)
    if base_generation_config is None:
        generation_config = GenerationConfig()
    else:
        generation_config = copy.deepcopy(base_generation_config)

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
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "penalty_alpha",
        ]:
            try:
                setattr(generation_config, key, None)
            except Exception:
                continue

    kwargs: dict[str, Any] = {
        "generation_config": generation_config,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if assistant_enabled and assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
        kwargs["num_assistant_tokens"] = int(assistant_num_tokens)
        kwargs["assistant_confidence_threshold"] = float(assistant_confidence_threshold)
        kwargs["num_assistant_tokens_schedule"] = str(assistant_num_tokens_schedule)
    return kwargs


def _cleanup_models(*models: Any) -> None:
    for model in models:
        if model is None:
            continue
        try:
            del model
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


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
        "temperature": args.temperature,
        "warmup_samples": args.warmup_samples,
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


def _load_tokenizer(tokenizer_source: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def _validate_generate_support(target_model: Any) -> None:
    candidate_models: list[Any] = [target_model]

    get_base_model = getattr(target_model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
            if base_model is not None and base_model is not target_model:
                candidate_models.append(base_model)
        except Exception:
            pass

    base_model_attr = getattr(target_model, "base_model", None)
    if base_model_attr is not None and base_model_attr is not target_model:
        candidate_models.append(base_model_attr)

    for model in candidate_models:
        try:
            signature = inspect.signature(model.generate)
        except Exception:
            continue

        if "assistant_model" in signature.parameters:
            return

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return

    raise RuntimeError(
        "当前模型的 generate() 接口未暴露 assistant_model 或通用 **kwargs，"
        "无法安全启用 speculative decoding。请检查 transformers / peft 版本兼容性。"
    )


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

    prompt_lines = [
        f"<|system|>\n{system_prompt}",
        f"<|user|>\n{_compose_user_prompt(instruction, input_text)}",
        "<|assistant|>\n",
    ]
    return "\n".join(prompt_lines)


def _chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _decode_generated_batch(tokenizer: Any, output_ids: Any, attention_mask: Any) -> tuple[list[str], list[int]]:
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    predictions: list[str] = []
    output_token_counts: list[int] = []
    for row_idx, prompt_len in enumerate(prompt_lengths):
        generated_ids = output_ids[row_idx][int(prompt_len) :]
        output_token_counts.append(int(generated_ids.shape[0]))
        predictions.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return predictions, output_token_counts


def _try_parse_json(text: str) -> tuple[bool, str | None]:
    try:
        json.loads(text)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _build_case_name(assistant_enabled: bool) -> str:
    return "speculative_on" if assistant_enabled else "baseline_off"


def _warmup_case(
    *,
    target_bundle: LoadedModelBundle,
    assistant_bundle: LoadedModelBundle | None,
    tokenizer: Any,
    system_prompt: str,
    dataset: list[dict[str, Any]],
    warmup_samples: int,
    generation_kwargs: dict[str, Any],
) -> None:
    if warmup_samples <= 0 or not dataset:
        return

    import torch

    target_device = (
        target_bundle.model.device
        if hasattr(target_bundle.model, "device") and target_bundle.model.device is not None
        else next(target_bundle.model.parameters()).device
    )
    warmup_batch = dataset[:warmup_samples]
    with torch.no_grad():
        # speculative decoding 的 assisted generate 仅支持 batch_size=1，
        # 因此预热也必须逐条执行，避免把多条样本拼成一个 batch。
        for item in warmup_batch:
            prompt = _render_chat_prompt(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                instruction=str(item["instruction"]),
                input_text=str(item["input"]),
            )
            inputs = tokenizer([prompt], return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(target_device)
            attention_mask = inputs["attention_mask"].to(target_device)
            _ = target_bundle.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_case(
    *,
    case_name: str,
    assistant_enabled: bool,
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    dtype: Any,
    dtype_label: str,
) -> dict[str, Any]:
    import torch

    trust_remote_code = bool(config.get("trust_remote_code", True))
    prefer_flash_attention_2 = bool(config.get("prefer_flash_attention_2", True))
    prefer_same_gpu = bool(config.get("prefer_same_gpu", True))
    preferred_cuda_device = int(config.get("preferred_cuda_device", 0))
    allow_auto_device_map_fallback = bool(config.get("allow_auto_device_map_fallback", True))
    warmup_samples = int(config.get("warmup_samples", 1))
    assistant_num_tokens = int(config.get("assistant_num_tokens", 8))
    assistant_confidence_threshold = float(config.get("assistant_confidence_threshold", 0.55))
    assistant_num_tokens_schedule = str(config.get("assistant_num_tokens_schedule", "heuristic_transient"))
    batch_size = int(config["batch_size"])
    max_new_tokens = int(config["max_new_tokens"])
    temperature = float(config["temperature"])

    target_bundle = _load_model_bundle(
        model_path=str(config["target_model_path"]),
        dtype=dtype,
        dtype_label=dtype_label,
        trust_remote_code=trust_remote_code,
        prefer_flash_attention_2=prefer_flash_attention_2,
        torch_module=torch,
        prefer_same_gpu=prefer_same_gpu,
        preferred_cuda_device=preferred_cuda_device,
        allow_auto_device_map_fallback=allow_auto_device_map_fallback,
    )
    assistant_bundle: LoadedModelBundle | None = None
    if assistant_enabled:
        assistant_bundle = _load_model_bundle(
            model_path=str(config["assistant_model_path"]),
            dtype=dtype,
            dtype_label=dtype_label,
            trust_remote_code=trust_remote_code,
            prefer_flash_attention_2=prefer_flash_attention_2,
            torch_module=torch,
            prefer_same_gpu=prefer_same_gpu,
            preferred_cuda_device=preferred_cuda_device,
            allow_auto_device_map_fallback=allow_auto_device_map_fallback,
        )

    try:
        _sanitize_model_generation_config(
            target_bundle.model,
            tokenizer,
            temperature=temperature,
            repetition_penalty=1.0,
        )
        if assistant_bundle is not None:
            _sanitize_model_generation_config(
                assistant_bundle.model,
                tokenizer,
                temperature=temperature,
                repetition_penalty=1.0,
            )
            _validate_generate_support(target_bundle.model)

        generation_kwargs = _build_generation_kwargs(
            model=target_bundle.model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            assistant_enabled=assistant_enabled,
            assistant_model=(assistant_bundle.model if assistant_bundle is not None else None),
            assistant_num_tokens=assistant_num_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
            assistant_num_tokens_schedule=assistant_num_tokens_schedule,
        )

        print(
            "[INFO] {case} 设备: target={target_device} strategy={target_strategy} cpu_offload={target_offload}; "
            "assistant={assistant_device} strategy={assistant_strategy} cpu_offload={assistant_offload}".format(
                case=case_name,
                target_device=target_bundle.primary_device,
                target_strategy=target_bundle.load_strategy,
                target_offload=target_bundle.cpu_offload_detected,
                assistant_device=(assistant_bundle.primary_device if assistant_bundle else "-"),
                assistant_strategy=(assistant_bundle.load_strategy if assistant_bundle else "-"),
                assistant_offload=(assistant_bundle.cpu_offload_detected if assistant_bundle else "-"),
            ),
            flush=True,
        )

        if assistant_enabled:
            print(
                "[INFO] {case} speculative 参数: num_assistant_tokens={num_tokens}, schedule={schedule}, "
                "assistant_confidence_threshold={threshold}".format(
                    case=case_name,
                    num_tokens=assistant_num_tokens,
                    schedule=assistant_num_tokens_schedule,
                    threshold=assistant_confidence_threshold,
                ),
                flush=True,
            )

        _warmup_case(
            target_bundle=target_bundle,
            assistant_bundle=assistant_bundle,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            dataset=dataset,
            warmup_samples=warmup_samples,
            generation_kwargs=generation_kwargs,
        )

        sample_rows: list[dict[str, Any]] = []
        batch_latencies: list[float] = []
        batch_throughputs: list[float] = []
        batch_peak_vram: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        parse_ok_count = 0

        for batch in _chunked(dataset, batch_size):
            prompts = [
                _render_chat_prompt(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    instruction=str(item["instruction"]),
                    input_text=str(item["input"]),
                )
                for item in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            input_token_count = int(attention_mask.sum().item())

            target_device = (
                target_bundle.model.device
                if hasattr(target_bundle.model, "device") and target_bundle.model.device is not None
                else next(target_bundle.model.parameters()).device
            )
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)

            with time_and_memory_tracker(
                input_text="\n\n".join(prompts),
                input_tokens=input_token_count,
                output_tokens=0,
                model_config={
                    "assistant_enabled": assistant_enabled,
                    "target_model_path": str(config["target_model_path"]),
                    "assistant_model_path": str(config["assistant_model_path"]) if assistant_enabled else None,
                    "dtype": dtype_label,
                    "attn_implementation": target_bundle.attn_implementation,
                    "case_name": case_name,
                    "target_primary_device": target_bundle.primary_device,
                    "target_cpu_offload": target_bundle.cpu_offload_detected,
                    "assistant_primary_device": (assistant_bundle.primary_device if assistant_bundle else None),
                    "assistant_cpu_offload": (assistant_bundle.cpu_offload_detected if assistant_bundle else None),
                },
            ) as tracker:
                with torch.no_grad():
                    output_ids = target_bundle.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )
                predictions, output_token_counts = _decode_generated_batch(tokenizer, output_ids, attention_mask)
                batch_output_tokens = int(sum(output_token_counts))
                tracker.set_output_tokens(batch_output_tokens)
                tracker.set_output_text("\n\n".join(predictions))

            metrics = tracker.metrics
            batch_latency = float(metrics["latency_sec"])
            batch_latency_per_sample = _safe_div(batch_latency, len(batch))
            batch_peak = float(metrics["peak_vram_mb"])
            batch_tps = float(metrics["throughput_tps"])
            batch_latencies.append(batch_latency)
            batch_throughputs.append(batch_tps)
            batch_peak_vram.append(batch_peak)

            total_input_tokens += input_token_count
            total_output_tokens += int(sum(output_token_counts))

            for item, prompt, prediction, output_token_count in zip(batch, prompts, predictions, output_token_counts):
                parse_ok, parse_error = _try_parse_json(prediction)
                if parse_ok:
                    parse_ok_count += 1
                sample_rows.append(
                    {
                        "sample_index": int(item["index"]),
                        "instruction": str(item["instruction"]),
                        "input": str(item["input"]),
                        "prompt": prompt,
                        "expected_output": str(item["expected_output"]),
                        "prediction": prediction,
                        "parse_ok": bool(parse_ok),
                        "parse_error": parse_error,
                        "latency_sec": _round_float(batch_latency_per_sample),
                        "batch_latency_sec": _round_float(batch_latency),
                        "output_tokens": int(output_token_count),
                        "batch_peak_vram_mb": _round_float(batch_peak, 3),
                        "batch_throughput_tps": _round_float(batch_tps, 3),
                    }
                )

        num_samples = len(sample_rows)
        avg_latency = _safe_mean([float(row["latency_sec"]) for row in sample_rows])
        avg_output_tokens = _safe_div(total_output_tokens, num_samples)
        avg_input_tokens = _safe_div(total_input_tokens, num_samples)
        total_runtime = float(sum(batch_latencies))
        token_tps = _safe_div(total_output_tokens, total_runtime)
        parse_ok_rate = _safe_div(parse_ok_count, num_samples)

        return {
            "case_name": case_name,
            "assistant_enabled": assistant_enabled,
            "target_model_path": str(config["target_model_path"]),
            "assistant_model_path": str(config["assistant_model_path"]) if assistant_enabled else None,
            "target_base_model_ref": target_bundle.base_model_ref,
            "target_adapter_path": target_bundle.adapter_path,
            "assistant_base_model_ref": assistant_bundle.base_model_ref if assistant_bundle else None,
            "assistant_adapter_path": assistant_bundle.adapter_path if assistant_bundle else None,
            "dtype": dtype_label,
            "attn_implementation": target_bundle.attn_implementation,
            "target_primary_device": target_bundle.primary_device,
            "target_device_map": target_bundle.device_map,
            "target_cpu_offload_detected": target_bundle.cpu_offload_detected,
            "target_load_strategy": target_bundle.load_strategy,
            "assistant_primary_device": assistant_bundle.primary_device if assistant_bundle else None,
            "assistant_device_map": assistant_bundle.device_map if assistant_bundle else None,
            "assistant_cpu_offload_detected": assistant_bundle.cpu_offload_detected if assistant_bundle else None,
            "assistant_load_strategy": assistant_bundle.load_strategy if assistant_bundle else None,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "warmup_samples": warmup_samples,
            "assistant_num_tokens": (assistant_num_tokens if assistant_enabled else None),
            "assistant_confidence_threshold": (assistant_confidence_threshold if assistant_enabled else None),
            "assistant_num_tokens_schedule": (assistant_num_tokens_schedule if assistant_enabled else None),
            "avg_latency_sec_per_sample": _round_float(avg_latency),
            "token_throughput_tps": _round_float(token_tps, 3),
            "peak_vram_mb": _round_float(max(batch_peak_vram) if batch_peak_vram else 0.0, 3),
            "parse_ok_count": int(parse_ok_count),
            "parse_ok_rate": _round_float(parse_ok_rate, 4),
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "avg_input_tokens": _round_float(avg_input_tokens, 3),
            "avg_output_tokens": _round_float(avg_output_tokens, 3),
            "avg_batch_latency_sec": _round_float(_safe_mean(batch_latencies)),
            "avg_batch_throughput_tps": _round_float(_safe_mean(batch_throughputs), 3),
            "samples": sample_rows,
        }
    finally:
        _cleanup_models(
            assistant_bundle.model if assistant_bundle is not None else None,
            target_bundle.model,
        )


def _build_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((item for item in results if not bool(item.get("assistant_enabled"))), None)
    speculative = next((item for item in results if bool(item.get("assistant_enabled"))), None)
    if baseline is None or speculative is None:
        return {}

    baseline_latency = float(baseline.get("avg_latency_sec_per_sample", 0.0))
    speculative_latency = float(speculative.get("avg_latency_sec_per_sample", 0.0))
    baseline_tps = float(baseline.get("token_throughput_tps", 0.0))
    speculative_tps = float(speculative.get("token_throughput_tps", 0.0))
    baseline_parse = float(baseline.get("parse_ok_rate", 0.0))
    speculative_parse = float(speculative.get("parse_ok_rate", 0.0))
    baseline_vram = float(baseline.get("peak_vram_mb", 0.0))
    speculative_vram = float(speculative.get("peak_vram_mb", 0.0))

    return {
        "latency_speedup_vs_baseline": _round_float(
            _safe_div(baseline_latency, speculative_latency),
            4,
        ),
        "avg_latency_delta_sec": _round_float(speculative_latency - baseline_latency, 6),
        "throughput_gain_pct_vs_baseline": _round_float(
            _safe_div(speculative_tps - baseline_tps, baseline_tps) * 100.0,
            4,
        ),
        "peak_vram_delta_mb": _round_float(speculative_vram - baseline_vram, 3),
        "parse_ok_rate_delta": _round_float(speculative_parse - baseline_parse, 4),
    }


def main() -> None:
    args = parse_args()
    runtime_config = _load_runtime_config(args)
    system_prompt = _load_system_prompt(args.base_config)
    dataset_path = _resolve_dataset_path(runtime_config["dataset_path"])
    report_path = _resolve_report_path(runtime_config.get("report_path"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(dataset_path, int(runtime_config["num_samples"]))

    batch_size = int(runtime_config["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    if float(runtime_config["temperature"]) != 0.0:
        print("[WARN] 当前实验建议使用 greedy decoding，temperature 推荐保持 0.0。", flush=True)

    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise RuntimeError("运行该实验需要安装 torch 与 transformers。") from exc

    dtype, dtype_label = _select_torch_dtype(torch)
    tokenizer_source = _maybe_load_adapter(_resolve_local_or_hf_ref(str(runtime_config["target_model_path"])))[0]
    tokenizer = _load_tokenizer(
        tokenizer_source=tokenizer_source,
        trust_remote_code=bool(runtime_config.get("trust_remote_code", True)),
    )

    cases = [False] if args.disable_assistant else [False, True]
    results: list[dict[str, Any]] = []
    for assistant_enabled in cases:
        case_name = _build_case_name(assistant_enabled)
        print(f"[INFO] 开始运行 {case_name} ...", flush=True)
        result = run_case(
            case_name=case_name,
            assistant_enabled=assistant_enabled,
            config=runtime_config,
            dataset=dataset,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            dtype=dtype,
            dtype_label=dtype_label,
        )
        results.append(result)
        print(
            "[INFO] {name} 完成: avg_latency={latency:.4f}s/sample, "
            "tokens_per_second={tps:.3f}, peak_vram={vram:.1f}MB, parse_ok_rate={parse:.4f}".format(
                name=case_name,
                latency=float(result["avg_latency_sec_per_sample"]),
                tps=float(result["token_throughput_tps"]),
                vram=float(result["peak_vram_mb"]),
                parse=float(result["parse_ok_rate"]),
            ),
            flush=True,
        )

    comparison = _build_comparison(results)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": {
            "name": "exp8_speculative_decoding",
            "entry": "experiments/12_exp8_speculative_decoding/run_speculative_benchmark.py",
        },
        "config": {
            "target_model_path": str(runtime_config["target_model_path"]),
            "assistant_model_path": str(runtime_config["assistant_model_path"]),
            "dataset_path": str(dataset_path),
            "num_samples": int(len(dataset)),
            "batch_size": batch_size,
            "max_new_tokens": int(runtime_config["max_new_tokens"]),
            "temperature": float(runtime_config["temperature"]),
            "warmup_samples": int(runtime_config.get("warmup_samples", 1)),
            "assistant_num_tokens": int(runtime_config.get("assistant_num_tokens", 8)),
            "assistant_confidence_threshold": float(runtime_config.get("assistant_confidence_threshold", 0.55)),
            "assistant_num_tokens_schedule": str(
                runtime_config.get("assistant_num_tokens_schedule", "heuristic_transient")
            ),
            "prefer_same_gpu": bool(runtime_config.get("prefer_same_gpu", True)),
            "preferred_cuda_device": int(runtime_config.get("preferred_cuda_device", 0)),
            "allow_auto_device_map_fallback": bool(runtime_config.get("allow_auto_device_map_fallback", True)),
            "disable_assistant": bool(args.disable_assistant),
            "system_prompt": system_prompt,
            "tokenizer_source": tokenizer_source,
        },
        "environment": {
            "transformers_version": getattr(transformers, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "dtype": dtype_label,
            "flash_attention_2_available": _is_flash_attn_available(),
        },
        "results": results,
        "comparison": comparison,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = record_run_meta(
        report_path.parent,
        merged_config={"speculative_decoding": runtime_config},
        cli_args=vars(args),
        argv=sys.argv,
        seed=None,
        data_paths=[dataset_path],
        extra_meta={
            "entry": "experiments/12_exp8_speculative_decoding/run_speculative_benchmark.py",
            "stage": "speculative_benchmark",
            "report_path": str(report_path),
        },
    )

    print(f"[OK] report saved      : {report_path}", flush=True)
    print(f"[OK] run meta          : {meta_path}", flush=True)
    for result in results:
        print(
            "[OK] {name:<16} latency={latency:.4f}s/sample  tps={tps:.3f}  peak_vram={vram:.1f}MB  parse_ok={parse:.4f}".format(
                name=str(result["case_name"]),
                latency=float(result["avg_latency_sec_per_sample"]),
                tps=float(result["token_throughput_tps"]),
                vram=float(result["peak_vram_mb"]),
                parse=float(result["parse_ok_rate"]),
            ),
            flush=True,
        )
    if comparison:
        print(
            "[OK] speculative speedup vs baseline : {speedup:.4f}x".format(
                speedup=float(comparison["latency_speedup_vs_baseline"])
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
