#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "deepconf_speculative.yaml"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
EXP12_PATH = REPO_ROOT / "experiments" / "12_exp8_speculative_decoding" / "run_speculative_benchmark.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from deepconf_utils import (
    aggregate_token_confidence,
    select_weighted_confidence_candidate,
    summarize_candidate_pool,
)
from src.eval_core.evaluate_toolcall_accuracy import canonicalize_commands, payload_to_commands
from src.eval_core.performance_monitor import time_and_memory_tracker
from src.utils.config import load_config
from src.utils.run_meta import record_run_meta


EXP12_SPEC = importlib.util.spec_from_file_location("exp12_speculative", EXP12_PATH)
if EXP12_SPEC is None or EXP12_SPEC.loader is None:
    raise RuntimeError(f"无法加载 Exp12 模块: {EXP12_PATH}")
EXP12 = importlib.util.module_from_spec(EXP12_SPEC)
sys.modules[EXP12_SPEC.name] = EXP12
EXP12_SPEC.loader.exec_module(EXP12)


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    family_name: str
    assistant_enabled: bool
    deepconf_enabled: bool
    num_candidates: int
    do_sample: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验14 Exp10：DeepConf 与 Speculative Decoding 联合实验。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="实验配置 YAML。")
    parser.add_argument("--base-config", type=Path, default=EXP12.DEFAULT_BASE_CONFIG_PATH, help="基础配置 YAML。")
    parser.add_argument("--target-model-path", type=str, default=None)
    parser.add_argument("--assistant-model-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--warmup-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--deepconf-candidate-counts", type=str, default=None, help="逗号分隔，仅对 DeepConf 分支生效，例如 2,3,4")
    parser.add_argument("--cases", type=str, default=None, help="逗号分隔，可选 baseline_off,speculative_on,deepconf_target,deepconf_speculative")
    parser.add_argument("--assistant-num-tokens", type=int, default=None)
    parser.add_argument("--assistant-confidence-threshold", type=float, default=None)
    parser.add_argument("--assistant-num-tokens-schedule", type=str, default=None)
    parser.add_argument("--top-k-confidence", type=int, default=None)
    parser.add_argument("--bottom-fraction", type=float, default=None)
    parser.add_argument("--tail-fraction", type=float, default=None)
    parser.add_argument("--candidate-temperature-step", type=float, default=None)
    parser.add_argument("--candidate-top-p-step", type=float, default=None)
    parser.add_argument("--candidate-duplicate-retry-limit", type=int, default=None)
    parser.add_argument("--candidate-retry-temperature-bump", type=float, default=None)
    parser.add_argument("--candidate-retry-top-p-bump", type=float, default=None)
    parser.add_argument("--candidate-retry-seed-stride", type=int, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


ALL_CASES: tuple[ExperimentCase, ...] = (
    ExperimentCase(name="baseline_off", family_name="baseline_off", assistant_enabled=False, deepconf_enabled=False, num_candidates=1, do_sample=False),
    ExperimentCase(name="speculative_on", family_name="speculative_on", assistant_enabled=True, deepconf_enabled=False, num_candidates=1, do_sample=False),
    ExperimentCase(name="deepconf_target", family_name="deepconf_target", assistant_enabled=False, deepconf_enabled=True, num_candidates=4, do_sample=True),
    ExperimentCase(name="deepconf_speculative", family_name="deepconf_speculative", assistant_enabled=True, deepconf_enabled=True, num_candidates=4, do_sample=True),
)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _parse_candidate_counts(raw_counts: Any, fallback: int) -> list[int]:
    if raw_counts is None:
        return [max(2, int(fallback))]
    if isinstance(raw_counts, int):
        return [max(2, int(raw_counts))]
    if isinstance(raw_counts, (list, tuple)):
        values = [max(2, int(item)) for item in raw_counts]
    else:
        values = [max(2, int(part.strip())) for part in str(raw_counts).split(",") if part.strip()]
    deduped = sorted(set(values))
    if not deduped:
        return [max(2, int(fallback))]
    return deduped


def _resolve_cases(raw_cases: str | None, default_cases: str, default_num_candidates: int, deepconf_candidate_counts: list[int]) -> list[ExperimentCase]:
    raw = raw_cases or default_cases
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    mapping = {case.name: case for case in ALL_CASES}
    invalid = [name for name in names if name not in mapping]
    if invalid:
        raise ValueError(f"不支持的 cases: {invalid}")
    resolved: list[ExperimentCase] = []
    for name in names:
        case = mapping[name]
        if case.deepconf_enabled:
            candidate_counts = deepconf_candidate_counts or [max(2, int(default_num_candidates))]
            for count in candidate_counts:
                variant_name = case.name if len(candidate_counts) == 1 else f"{case.name}_k{count}"
                resolved.append(
                    ExperimentCase(
                        name=variant_name,
                        family_name=case.family_name,
                        assistant_enabled=case.assistant_enabled,
                        deepconf_enabled=True,
                        num_candidates=max(2, int(count)),
                        do_sample=True,
                    )
                )
        else:
            resolved.append(case)
    return resolved


def _load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    for key, value in {
        "target_model_path": args.target_model_path,
        "assistant_model_path": args.assistant_model_path,
        "dataset_path": args.dataset_path,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "warmup_samples": args.warmup_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_candidates": args.num_candidates,
        "deepconf_candidate_counts": args.deepconf_candidate_counts,
        "cases": args.cases,
        "assistant_num_tokens": args.assistant_num_tokens,
        "assistant_confidence_threshold": args.assistant_confidence_threshold,
        "assistant_num_tokens_schedule": args.assistant_num_tokens_schedule,
        "top_k_confidence": args.top_k_confidence,
        "bottom_fraction": args.bottom_fraction,
        "tail_fraction": args.tail_fraction,
        "candidate_temperature_step": args.candidate_temperature_step,
        "candidate_top_p_step": args.candidate_top_p_step,
        "candidate_duplicate_retry_limit": args.candidate_duplicate_retry_limit,
        "candidate_retry_temperature_bump": args.candidate_retry_temperature_bump,
        "candidate_retry_top_p_bump": args.candidate_retry_top_p_bump,
        "candidate_retry_seed_stride": args.candidate_retry_seed_stride,
        "report_path": args.report_path,
    }.items():
        if value is not None:
            cfg[key] = value
    required = [
        "target_model_path",
        "assistant_model_path",
        "dataset_path",
        "num_samples",
        "max_new_tokens",
        "temperature",
        "top_p",
        "num_candidates",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"配置缺少必要字段: {missing}")
    cfg.setdefault("cases", "baseline_off,speculative_on,deepconf_target,deepconf_speculative")
    cfg.setdefault("batch_size", 1)
    cfg.setdefault("warmup_samples", 2)
    cfg.setdefault("assistant_num_tokens", 8)
    cfg.setdefault("assistant_confidence_threshold", 0.55)
    cfg.setdefault("assistant_num_tokens_schedule", "constant")
    cfg.setdefault("top_k_confidence", 20)
    cfg.setdefault("bottom_fraction", 0.2)
    cfg.setdefault("tail_fraction", 0.2)
    cfg.setdefault("avg_weight", 0.35)
    cfg.setdefault("bottom_weight", 0.35)
    cfg.setdefault("tail_weight", 0.15)
    cfg.setdefault("actual_prob_weight", 0.15)
    cfg.setdefault("candidate_seed", 42)
    cfg.setdefault("deepconf_candidate_counts", [max(2, int(cfg["num_candidates"]))])
    cfg.setdefault("candidate_temperature_step", 0.08)
    cfg.setdefault("candidate_top_p_step", 0.02)
    cfg.setdefault("candidate_duplicate_retry_limit", 2)
    cfg.setdefault("candidate_retry_temperature_bump", 0.06)
    cfg.setdefault("candidate_retry_top_p_bump", 0.01)
    cfg.setdefault("candidate_retry_seed_stride", 97)
    cfg.setdefault("trust_remote_code", True)
    cfg.setdefault("prefer_flash_attention_2", True)
    cfg.setdefault("prefer_same_gpu", True)
    cfg.setdefault("preferred_cuda_device", 0)
    cfg.setdefault("allow_auto_device_map_fallback", True)
    cfg["deepconf_candidate_counts"] = _parse_candidate_counts(cfg.get("deepconf_candidate_counts"), int(cfg["num_candidates"]))
    return cfg


def _resolve_local_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (EXPERIMENT_DIR / candidate).resolve()


def _build_generation_kwargs(
    *,
    target_model: Any,
    tokenizer: Any,
    case: ExperimentCase,
    config: dict[str, Any],
    assistant_model: Any | None,
    temperature_override: float | None = None,
    top_p_override: float | None = None,
) -> dict[str, Any]:
    kwargs = EXP12._build_generation_kwargs(
        model=target_model,
        tokenizer=tokenizer,
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=float(temperature_override if temperature_override is not None else (config["temperature"] if case.do_sample else 0.0)),
        assistant_enabled=case.assistant_enabled,
        assistant_model=assistant_model,
        assistant_num_tokens=int(config["assistant_num_tokens"]),
        assistant_confidence_threshold=float(config["assistant_confidence_threshold"]),
        assistant_num_tokens_schedule=str(config["assistant_num_tokens_schedule"]),
    )
    generation_config = kwargs["generation_config"]
    generation_config.do_sample = bool(case.do_sample)
    generation_config.num_return_sequences = 1
    generation_config.use_cache = True
    generation_config.top_p = float(top_p_override if top_p_override is not None else config["top_p"]) if case.do_sample else None
    generation_config.top_k = None
    generation_config.temperature = float(temperature_override if temperature_override is not None else config["temperature"]) if case.do_sample else None
    return kwargs


def _prepare_model_bundles(config: dict[str, Any], dtype: Any, dtype_label: str, torch_module: Any) -> tuple[Any, Any | None]:
    target_bundle = EXP12._load_model_bundle(
        model_path=str(config["target_model_path"]),
        dtype=dtype,
        dtype_label=dtype_label,
        trust_remote_code=bool(config["trust_remote_code"]),
        prefer_flash_attention_2=bool(config["prefer_flash_attention_2"]),
        torch_module=torch_module,
        prefer_same_gpu=bool(config["prefer_same_gpu"]),
        preferred_cuda_device=int(config["preferred_cuda_device"]),
        allow_auto_device_map_fallback=bool(config["allow_auto_device_map_fallback"]),
    )
    assistant_bundle = None
    if str(config.get("assistant_model_path", "")).strip():
        assistant_bundle = EXP12._load_model_bundle(
            model_path=str(config["assistant_model_path"]),
            dtype=dtype,
            dtype_label=dtype_label,
            trust_remote_code=bool(config["trust_remote_code"]),
            prefer_flash_attention_2=bool(config["prefer_flash_attention_2"]),
            torch_module=torch_module,
            prefer_same_gpu=bool(config["prefer_same_gpu"]),
            preferred_cuda_device=int(config["preferred_cuda_device"]),
            allow_auto_device_map_fallback=bool(config["allow_auto_device_map_fallback"]),
        )
    return target_bundle, assistant_bundle


def _score_prediction(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    prediction: str,
    top_k_confidence: int,
    bottom_fraction: float,
    tail_fraction: float,
    avg_weight: float,
    bottom_weight: float,
    tail_weight: float,
    actual_prob_weight: float,
) -> dict[str, float]:
    import torch

    if not prediction.strip():
        return aggregate_token_confidence(
            token_confidences=[],
            actual_token_probs=[],
            bottom_fraction=bottom_fraction,
            tail_fraction=tail_fraction,
            avg_weight=avg_weight,
            bottom_weight=bottom_weight,
            tail_weight=tail_weight,
            actual_prob_weight=actual_prob_weight,
        )

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + prediction, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if full_len <= prompt_len:
        return aggregate_token_confidence(
            token_confidences=[],
            actual_token_probs=[],
            bottom_fraction=bottom_fraction,
            tail_fraction=tail_fraction,
            avg_weight=avg_weight,
            bottom_weight=bottom_weight,
            tail_weight=tail_weight,
            actual_prob_weight=actual_prob_weight,
        )

    device = model.device if hasattr(model, "device") and model.device is not None else next(model.parameters()).device
    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0]

    step_logits = logits[prompt_len - 1 : full_len - 1].float()
    actual_token_ids = input_ids[0, prompt_len:full_len]
    log_denom = torch.logsumexp(step_logits, dim=-1, keepdim=True)
    top_k = max(1, min(int(top_k_confidence), int(step_logits.shape[-1])))
    topk_logits = torch.topk(step_logits, k=top_k, dim=-1).values
    topk_probs = torch.exp(topk_logits - log_denom)
    actual_logits = step_logits.gather(dim=-1, index=actual_token_ids.unsqueeze(-1)).squeeze(-1)
    actual_log_probs = actual_logits - log_denom.squeeze(-1)
    token_confidences = topk_probs.sum(dim=-1).detach().cpu().tolist()
    actual_token_probs = torch.exp(actual_log_probs).detach().cpu().tolist()
    return aggregate_token_confidence(
        token_confidences=[float(v) for v in token_confidences],
        actual_token_probs=[float(v) for v in actual_token_probs],
        bottom_fraction=bottom_fraction,
        tail_fraction=tail_fraction,
        avg_weight=avg_weight,
        bottom_weight=bottom_weight,
        tail_weight=tail_weight,
        actual_prob_weight=actual_prob_weight,
    )


def _candidate_parse_payload(prediction: str) -> dict[str, Any]:
    try:
        commands = payload_to_commands(prediction)
        return {
            "parse_ok": True,
            "canonical_commands": canonicalize_commands(commands),
            "action_signature": [str(cmd.get("action", "")) for cmd in commands],
            "commands": commands,
        }
    except Exception as exc:
        return {
            "parse_ok": False,
            "canonical_commands": None,
            "action_signature": None,
            "commands": None,
            "parse_error": str(exc),
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _candidate_prediction_key(candidate: dict[str, Any]) -> str:
    if candidate.get("canonical_commands") is not None:
        return str(candidate["canonical_commands"])
    return " ".join(str(candidate.get("prediction", "")).split())


def _build_candidate_sampling_profile(
    *,
    case: ExperimentCase,
    config: dict[str, Any],
    candidate_index: int,
    retry_index: int,
) -> tuple[float | None, float | None]:
    if not case.do_sample:
        return None, None

    center = float(candidate_index) - ((float(case.num_candidates) - 1.0) / 2.0)
    base_temperature = float(config["temperature"])
    base_top_p = float(config["top_p"])

    temperature = base_temperature + center * float(config.get("candidate_temperature_step", 0.0))
    top_p = base_top_p + center * float(config.get("candidate_top_p_step", 0.0))

    if retry_index > 0:
        temperature += retry_index * float(config.get("candidate_retry_temperature_bump", 0.0))
        top_p += retry_index * float(config.get("candidate_retry_top_p_bump", 0.0))

    return _clamp(temperature, 0.05, 2.0), _clamp(top_p, 0.05, 1.0)


def _evaluate_selected_prediction(selected_candidate: dict[str, Any], expected_output: str) -> dict[str, Any]:
    try:
        gt_commands = payload_to_commands(expected_output)
    except Exception as exc:
        raise RuntimeError(f"Ground truth 非法: {exc}") from exc

    if not bool(selected_candidate.get("parse_ok", False)):
        return {"parse_ok": False, "exact_match": False, "action_match": False}

    pred_commands = selected_candidate["commands"]
    pred_sig = [str(cmd.get("action", "")) for cmd in pred_commands]
    gt_sig = [str(cmd.get("action", "")) for cmd in gt_commands]
    return {
        "parse_ok": True,
        "exact_match": canonicalize_commands(pred_commands) == canonicalize_commands(gt_commands),
        "action_match": pred_sig == gt_sig,
    }


def _run_warmup_case(
    *,
    case: ExperimentCase,
    target_model: Any,
    assistant_model: Any | None,
    tokenizer: Any,
    system_prompt: str,
    dataset: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    import torch

    if int(config["warmup_samples"]) <= 0:
        return
    generation_kwargs = _build_generation_kwargs(
        target_model=target_model,
        tokenizer=tokenizer,
        case=case,
        config=config,
        assistant_model=assistant_model,
    )
    warmup_data = dataset[: int(config["warmup_samples"])]
    for item in warmup_data:
        prompt = EXP12._render_chat_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction=str(item["instruction"]),
            input_text=str(item["input"]),
        )
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        device = target_model.device if hasattr(target_model, "device") and target_model.device is not None else next(target_model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            _ = target_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)


def _generate_candidate(
    *,
    target_model: Any,
    assistant_model: Any | None,
    tokenizer: Any,
    prompt: str,
    case: ExperimentCase,
    config: dict[str, Any],
    seed: int,
    candidate_index: int,
    retry_index: int = 0,
) -> dict[str, Any]:
    import torch

    sampling_temperature, sampling_top_p = _build_candidate_sampling_profile(
        case=case,
        config=config,
        candidate_index=candidate_index,
        retry_index=retry_index,
    )

    generation_kwargs = _build_generation_kwargs(
        target_model=target_model,
        tokenizer=tokenizer,
        case=case,
        config=config,
        assistant_model=assistant_model,
        temperature_override=sampling_temperature,
        top_p_override=sampling_top_p,
    )

    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    device = target_model.device if hasattr(target_model, "device") and target_model.device is not None else next(target_model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    start = time.perf_counter()
    fork_devices = []
    if getattr(device, "type", "cpu") == "cuda":
        try:
            fork_devices = [int(device.index)] if device.index is not None else list(range(torch.cuda.device_count()))
        except Exception:
            fork_devices = []

    with torch.random.fork_rng(devices=fork_devices):
        if case.do_sample:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            output_ids = target_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
    generation_time = time.perf_counter() - start
    predictions, output_token_counts = EXP12._decode_generated_batch(tokenizer, output_ids, attention_mask)
    prediction = predictions[0]
    generated_tokens = int(output_token_counts[0]) if output_token_counts else 0

    parse_meta = _candidate_parse_payload(prediction)
    return {
        "prediction": prediction,
        "generated_tokens": generated_tokens,
        "generation_time_sec": float(generation_time),
        "seed": int(seed),
        "sampling_temperature": float(sampling_temperature) if sampling_temperature is not None else None,
        "sampling_top_p": float(sampling_top_p) if sampling_top_p is not None else None,
        "resample_attempts": int(retry_index),
        **parse_meta,
    }


def run_case(
    *,
    case: ExperimentCase,
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    dtype: Any,
    dtype_label: str,
) -> dict[str, Any]:
    import torch

    target_bundle, assistant_bundle = _prepare_model_bundles(config, dtype, dtype_label, torch)
    try:
        EXP12._sanitize_model_generation_config(target_bundle.model, tokenizer, temperature=float(config["temperature"]), repetition_penalty=1.0)
        if assistant_bundle is not None:
            EXP12._sanitize_model_generation_config(assistant_bundle.model, tokenizer, temperature=float(config["temperature"]), repetition_penalty=1.0)
        if case.assistant_enabled:
            if assistant_bundle is None:
                raise RuntimeError("当前 case 需要 assistant_model，但配置未提供 assistant_model_path。")
            EXP12._validate_generate_support(target_bundle.model)

        _run_warmup_case(
            case=case,
            target_model=target_bundle.model,
            assistant_model=(assistant_bundle.model if assistant_bundle is not None else None),
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            dataset=dataset,
            config=config,
        )

        sample_rows: list[dict[str, Any]] = []
        parse_ok = 0
        exact_match = 0
        action_match = 0
        total_latency = 0.0
        total_generation_time = 0.0
        total_scoring_time = 0.0
        total_candidate_tokens = 0
        total_selected_output_tokens = 0
        total_unique_predictions = 0.0
        total_unique_action_signatures = 0.0
        total_duplicate_prediction_rate = 0.0
        total_duplicate_action_signature_rate = 0.0
        total_resample_attempts = 0
        selected_nonzero_count = 0
        peak_vrams: list[float] = []

        for sample_idx, item in enumerate(dataset):
            prompt = EXP12._render_chat_prompt(
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                instruction=str(item["instruction"]),
                input_text=str(item["input"]),
            )
            candidates: list[dict[str, Any]] = []
            scoring_time_sec = 0.0
            candidate_seed_base = int(config["candidate_seed"]) + sample_idx * 1000
            seen_prediction_keys: set[str] = set()

            with time_and_memory_tracker(
                input_text=prompt,
                input_tokens=len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                output_tokens=0,
                model_config={
                    "case_name": case.name,
                    "family_name": case.family_name,
                    "assistant_enabled": case.assistant_enabled,
                    "deepconf_enabled": case.deepconf_enabled,
                    "num_candidates": case.num_candidates,
                    "dtype": dtype_label,
                },
            ) as tracker:
                for candidate_index in range(case.num_candidates):
                    max_retry = max(0, int(config.get("candidate_duplicate_retry_limit", 0))) if case.deepconf_enabled else 0
                    candidate = None
                    for retry_index in range(max_retry + 1):
                        seed = candidate_seed_base + candidate_index + retry_index * int(config.get("candidate_retry_seed_stride", 97))
                        candidate = _generate_candidate(
                            target_model=target_bundle.model,
                            assistant_model=(assistant_bundle.model if assistant_bundle is not None else None),
                            tokenizer=tokenizer,
                            prompt=prompt,
                            case=case,
                            config=config,
                            seed=seed,
                            candidate_index=candidate_index,
                            retry_index=retry_index,
                        )
                        prediction_key = _candidate_prediction_key(candidate)
                        if prediction_key not in seen_prediction_keys or retry_index >= max_retry:
                            seen_prediction_keys.add(prediction_key)
                            break
                    if candidate is None:
                        raise RuntimeError("候选生成失败，未返回任何 candidate。")
                    candidate["candidate_index"] = candidate_index
                    total_candidate_tokens += int(candidate["generated_tokens"])
                    total_generation_time += float(candidate["generation_time_sec"])
                    total_resample_attempts += int(candidate.get("resample_attempts", 0))

                    score_start = time.perf_counter()
                    confidence_metrics = _score_prediction(
                        model=target_bundle.model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        prediction=str(candidate["prediction"]),
                        top_k_confidence=int(config["top_k_confidence"]),
                        bottom_fraction=float(config["bottom_fraction"]),
                        tail_fraction=float(config["tail_fraction"]),
                        avg_weight=float(config["avg_weight"]),
                        bottom_weight=float(config["bottom_weight"]),
                        tail_weight=float(config["tail_weight"]),
                        actual_prob_weight=float(config["actual_prob_weight"]),
                    )
                    score_time = time.perf_counter() - score_start
                    total_scoring_time += score_time
                    scoring_time_sec += score_time
                    candidate.update(confidence_metrics)
                    candidates.append(candidate)

                selected_candidate = (
                    select_weighted_confidence_candidate(candidates)
                    if case.deepconf_enabled
                    else candidates[0]
                )
                if int(selected_candidate["candidate_index"]) != 0:
                    selected_nonzero_count += 1
                tracker.set_output_text(str(selected_candidate["prediction"]))
                tracker.set_output_tokens(int(selected_candidate["generated_tokens"]))

            metrics = tracker.metrics
            peak_vrams.append(float(metrics.get("peak_vram_mb", 0.0)))
            total_latency += float(metrics.get("latency_sec", 0.0))
            total_selected_output_tokens += int(selected_candidate["generated_tokens"])

            eval_result = _evaluate_selected_prediction(selected_candidate, str(item["expected_output"]))
            if eval_result["parse_ok"]:
                parse_ok += 1
            if eval_result["exact_match"]:
                exact_match += 1
            if eval_result["action_match"]:
                action_match += 1

            pool_summary = summarize_candidate_pool(candidates)
            total_unique_predictions += float(pool_summary["candidate_unique_prediction_count"])
            total_unique_action_signatures += float(pool_summary["candidate_unique_action_signature_count"])
            total_duplicate_prediction_rate += float(pool_summary["candidate_duplicate_prediction_rate"])
            total_duplicate_action_signature_rate += float(pool_summary["candidate_duplicate_action_signature_rate"])
            sample_rows.append(
                {
                    "sample_index": int(item["index"]),
                    "instruction": str(item["instruction"]),
                    "input": str(item["input"]),
                    "selected_prediction": str(selected_candidate["prediction"]),
                    "selected_candidate_index": int(selected_candidate["candidate_index"]),
                    "selected_generated_tokens": int(selected_candidate["generated_tokens"]),
                    "selected_deepconf_score": _round_float(float(selected_candidate.get("deepconf_score", 0.0))),
                    "selected_avg_confidence": _round_float(float(selected_candidate.get("avg_confidence", 0.0))),
                    "parse_ok": bool(eval_result["parse_ok"]),
                    "exact_match": bool(eval_result["exact_match"]),
                    "action_match": bool(eval_result["action_match"]),
                    "sample_latency_sec": _round_float(float(metrics.get("latency_sec", 0.0))),
                    "sample_peak_vram_mb": _round_float(float(metrics.get("peak_vram_mb", 0.0)), 3),
                    "sample_generation_time_sec": _round_float(sum(float(c["generation_time_sec"]) for c in candidates)),
                    "sample_scoring_time_sec": _round_float(scoring_time_sec),
                    "candidate_count": int(pool_summary["candidate_count"]),
                    "candidate_parse_ok_rate": _round_float(pool_summary["candidate_parse_ok_rate"], 4),
                    "candidate_avg_deepconf_score": _round_float(pool_summary["candidate_avg_deepconf_score"]),
                    "candidate_best_deepconf_score": _round_float(pool_summary["candidate_best_deepconf_score"]),
                    "candidate_unique_prediction_count": int(pool_summary["candidate_unique_prediction_count"]),
                    "candidate_unique_action_signature_count": int(pool_summary["candidate_unique_action_signature_count"]),
                    "candidate_duplicate_prediction_rate": _round_float(pool_summary["candidate_duplicate_prediction_rate"], 4),
                    "candidate_duplicate_action_signature_rate": _round_float(pool_summary["candidate_duplicate_action_signature_rate"], 4),
                    "candidate_total_resample_attempts": int(pool_summary["candidate_total_resample_attempts"]),
                    "candidates": candidates,
                }
            )

        num_samples = len(sample_rows)
        return {
            "case_name": case.name,
            "family_name": case.family_name,
            "assistant_enabled": case.assistant_enabled,
            "deepconf_enabled": case.deepconf_enabled,
            "num_candidates": case.num_candidates,
            "temperature": float(config["temperature"]) if case.do_sample else 0.0,
            "top_p": float(config["top_p"]) if case.do_sample else None,
            "assistant_num_tokens": int(config["assistant_num_tokens"]) if case.assistant_enabled else None,
            "assistant_confidence_threshold": float(config["assistant_confidence_threshold"]) if case.assistant_enabled else None,
            "assistant_num_tokens_schedule": str(config["assistant_num_tokens_schedule"]) if case.assistant_enabled else None,
            "parse_ok_rate": _round_float(_safe_div(parse_ok, num_samples), 4),
            "exact_match_rate": _round_float(_safe_div(exact_match, num_samples), 4),
            "action_match_rate": _round_float(_safe_div(action_match, num_samples), 4),
            "avg_latency_sec_per_sample": _round_float(_safe_div(total_latency, num_samples)),
            "avg_generation_time_sec_per_sample": _round_float(_safe_div(total_generation_time, num_samples)),
            "avg_scoring_time_sec_per_sample": _round_float(_safe_div(total_scoring_time, num_samples)),
            "candidate_token_throughput_tps": _round_float(_safe_div(total_candidate_tokens, total_latency), 3),
            "selected_token_throughput_tps": _round_float(_safe_div(total_selected_output_tokens, total_latency), 3),
            "avg_candidate_tokens_per_sample": _round_float(_safe_div(total_candidate_tokens, num_samples), 3),
            "avg_selected_output_tokens": _round_float(_safe_div(total_selected_output_tokens, num_samples), 3),
            "avg_unique_predictions_per_sample": _round_float(_safe_div(total_unique_predictions, num_samples), 3),
            "avg_unique_action_signatures_per_sample": _round_float(_safe_div(total_unique_action_signatures, num_samples), 3),
            "avg_duplicate_prediction_rate": _round_float(_safe_div(total_duplicate_prediction_rate, num_samples), 4),
            "avg_duplicate_action_signature_rate": _round_float(_safe_div(total_duplicate_action_signature_rate, num_samples), 4),
            "avg_resample_attempts_per_sample": _round_float(_safe_div(total_resample_attempts, num_samples), 3),
            "selected_nonzero_candidate_rate": _round_float(_safe_div(selected_nonzero_count, num_samples), 4),
            "peak_vram_mb": _round_float(max(peak_vrams) if peak_vrams else 0.0, 3),
            "samples": sample_rows,
        }
    finally:
        EXP12._cleanup_models(
            assistant_bundle.model if assistant_bundle is not None else None,
            target_bundle.model,
        )


def _build_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {str(item["case_name"]): item for item in results}
    comparison: dict[str, Any] = {}

    def add_delta(name: str, base: str, target: str, metric: str) -> None:
        if base not in lookup or target not in lookup:
            return
        comparison[name] = _round_float(float(lookup[target][metric]) - float(lookup[base][metric]), 6)

    def add_speedup(name: str, base: str, target: str, metric: str) -> None:
        if base not in lookup or target not in lookup:
            return
        comparison[name] = _round_float(
            _safe_div(float(lookup[base][metric]), float(lookup[target][metric])),
            4,
        )

    add_delta("speculative_accuracy_gain_vs_baseline", "baseline_off", "speculative_on", "exact_match_rate")
    add_speedup("speculative_latency_speedup_vs_baseline", "baseline_off", "speculative_on", "avg_latency_sec_per_sample")

    deepconf_targets = [item for item in results if str(item.get("family_name")) == "deepconf_target"]
    deepconf_speculatives = [item for item in results if str(item.get("family_name")) == "deepconf_speculative"]
    target_by_k = {int(item["num_candidates"]): item for item in deepconf_targets}

    for speculative_item in deepconf_speculatives:
        case_name = str(speculative_item["case_name"])
        add_delta(f"{case_name}_accuracy_gain_vs_speculative", "speculative_on", case_name, "exact_match_rate")
        add_delta(f"{case_name}_action_gain_vs_speculative", "speculative_on", case_name, "action_match_rate")
        matching_target = target_by_k.get(int(speculative_item["num_candidates"]))
        if matching_target is not None:
            comparison[f"{case_name}_latency_speedup_vs_{matching_target['case_name']}"] = _round_float(
                _safe_div(float(matching_target["avg_latency_sec_per_sample"]), float(speculative_item["avg_latency_sec_per_sample"])),
                4,
            )

    if len(deepconf_speculatives) == 1:
        add_delta("deepconf_speculative_accuracy_gain_vs_speculative", "speculative_on", deepconf_speculatives[0]["case_name"], "exact_match_rate")
        add_delta("deepconf_speculative_action_gain_vs_speculative", "speculative_on", deepconf_speculatives[0]["case_name"], "action_match_rate")
    if len(deepconf_targets) == 1 and len(deepconf_speculatives) == 1:
        comparison["deepconf_speculative_latency_speedup_vs_deepconf_target"] = _round_float(
            _safe_div(float(deepconf_targets[0]["avg_latency_sec_per_sample"]), float(deepconf_speculatives[0]["avg_latency_sec_per_sample"])),
            4,
        )

    def best_snapshot(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        best = max(
            items,
            key=lambda item: (
                float(item["exact_match_rate"]),
                float(item["action_match_rate"]),
                -float(item["avg_latency_sec_per_sample"]),
            ),
        )
        return {
            "case_name": str(best["case_name"]),
            "num_candidates": int(best["num_candidates"]),
            "exact_match_rate": float(best["exact_match_rate"]),
            "action_match_rate": float(best["action_match_rate"]),
            "avg_latency_sec_per_sample": float(best["avg_latency_sec_per_sample"]),
        }

    best_target = best_snapshot(deepconf_targets)
    best_speculative = best_snapshot(deepconf_speculatives)
    if best_target is not None:
        comparison["best_deepconf_target_by_exact"] = best_target
    if best_speculative is not None:
        comparison["best_deepconf_speculative_by_exact"] = best_speculative
    return comparison


def main() -> None:
    args = parse_args()
    runtime_config = _load_runtime_config(args)
    cases = _resolve_cases(
        raw_cases=runtime_config.get("cases"),
        default_cases="baseline_off,speculative_on,deepconf_target,deepconf_speculative",
        default_num_candidates=int(runtime_config["num_candidates"]),
        deepconf_candidate_counts=list(runtime_config.get("deepconf_candidate_counts", [int(runtime_config["num_candidates"])])),
    )
    report_path = _resolve_local_path(runtime_config.get("report_path"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": runtime_config,
                    "cases": [case.__dict__ for case in cases],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise RuntimeError("运行该实验需要 torch 与 transformers。") from exc

    dtype, dtype_label = EXP12._select_torch_dtype(torch)
    system_prompt = EXP12._load_system_prompt(args.base_config)
    dataset_path = EXP12._resolve_dataset_path(runtime_config["dataset_path"])
    dataset = EXP12._load_dataset(dataset_path, int(runtime_config["num_samples"]))

    tokenizer_source = EXP12._maybe_load_adapter(EXP12._resolve_local_or_hf_ref(str(runtime_config["target_model_path"])))[0]
    tokenizer = EXP12._load_tokenizer(
        tokenizer_source=tokenizer_source,
        trust_remote_code=bool(runtime_config.get("trust_remote_code", True)),
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"[INFO] 开始运行 {case.name} ...", flush=True)
        result = run_case(
            case=case,
            config=runtime_config,
            dataset=dataset,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            dtype=dtype,
            dtype_label=dtype_label,
        )
        results.append(result)
        print(
            "[INFO] {name} 完成: exact={exact:.4f}, action={action:.4f}, "
            "latency={latency:.4f}s/sample, candidate_tps={tps:.3f}".format(
                name=case.name,
                exact=float(result["exact_match_rate"]),
                action=float(result["action_match_rate"]),
                latency=float(result["avg_latency_sec_per_sample"]),
                tps=float(result["candidate_token_throughput_tps"]),
            ),
            flush=True,
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": {
            "name": "exp10_deepconf_speculative",
            "entry": "experiments/14_exp10_deepconf_speculative/run_deepconf_speculative_benchmark.py",
        },
        "config": {
            **runtime_config,
            "dataset_path": str(dataset_path),
            "num_samples": len(dataset),
            "system_prompt": system_prompt,
            "tokenizer_source": tokenizer_source,
        },
        "environment": {
            "transformers_version": getattr(transformers, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "dtype": dtype_label,
            "flash_attention_2_available": EXP12._is_flash_attn_available(),
        },
        "results": results,
        "comparison": _build_comparison(results),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    record_run_meta(
        report_path.parent,
        merged_config={"deepconf_speculative": runtime_config},
        cli_args=vars(args),
        argv=sys.argv,
        seed=int(runtime_config.get("candidate_seed", 42)),
        data_paths=[str(dataset_path)],
        extra_meta={"report_path": str(report_path)},
    )
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
