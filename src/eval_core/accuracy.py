from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data_core.dataset_safety import enforce_train_eval_no_leakage
from src.eval_core.evaluate_toolcall_accuracy import (
    canonicalize_commands,
    normalize_text,
    evaluate_toolcall_accuracy,
    payload_to_commands,
)
from src.eval_core.parse_failure_diagnostics import diagnose_parse_failure, summarize_parse_failures
from src.eval_core.performance_monitor import time_and_memory_tracker
from src.eval_core.prompting import DEFAULT_EVAL_SYSTEM_PROMPT, build_eval_messages

logger = logging.getLogger(__name__)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _extract_generation_trace(engine: Any) -> dict[str, Any] | None:
    getter = getattr(engine, "get_last_generation_trace", None)
    if not callable(getter):
        return None
    trace = getter()
    return trace if isinstance(trace, dict) else None


def _summarize_early_exit_results(details: list[dict[str, Any]]) -> dict[str, Any] | None:
    sample_exit_layer_stats: list[dict[str, Any]] = []
    token_exit_layer_histogram: Counter[int] = Counter()
    sample_avg_exit_layers: list[float] = []
    token_exit_layers: list[float] = []
    total_layers = 0
    exit_triggered_tokens = 0
    forced_cap_tokens = 0
    fallback_used_count = 0
    string_guard_blocked_tokens = 0
    draft_only_candidate_tokens = 0
    draft_verified_matches = 0
    draft_verified_mismatches = 0
    candidate_probe_sums: dict[str, dict[str, float]] = {}

    for detail in details:
        trace = detail.get("early_exit_trace")
        if not isinstance(trace, dict):
            continue

        summary = trace.get("summary", {}) if isinstance(trace.get("summary"), dict) else {}
        trace_total_layers = int(trace.get("total_layers", 0) or 0)
        if trace_total_layers > 0:
            total_layers = trace_total_layers

        avg_exit_layer = float(summary.get("avg_exit_layer", float(trace_total_layers or 0)))
        tokens_generated = int(trace.get("tokens_generated", summary.get("token_count", 0) or 0))
        exit_triggered_token_count = int(summary.get("exit_triggered_tokens", 0) or 0)
        forced_cap_token_count = int(summary.get("forced_cap_tokens", 0) or 0)
        sample_exit_layer_stats.append(
            {
                "dataset_index": detail.get("dataset_index"),
                "avg_exit_layer": avg_exit_layer,
                "tokens_generated": tokens_generated,
                "final_layer_cap": int(trace.get("final_layer_cap", trace_total_layers - 1 if trace_total_layers else 0)),
                "exit_triggered_tokens": exit_triggered_token_count,
                "forced_cap_tokens": forced_cap_token_count,
                "fallback_used": bool(detail.get("early_exit_fallback_used", False)),
            }
        )
        sample_avg_exit_layers.append(avg_exit_layer)
        exit_triggered_tokens += exit_triggered_token_count
        forced_cap_tokens += forced_cap_token_count
        fallback_used_count += 1 if bool(detail.get("early_exit_fallback_used", False)) else 0
        string_guard_blocked_tokens += int(summary.get("string_guard_blocked_tokens", 0) or 0)
        draft_only_candidate_tokens += int(summary.get("draft_only_candidate_tokens", 0) or 0)
        draft_verified_matches += int(summary.get("draft_verified_matches", 0) or 0)
        draft_verified_mismatches += int(summary.get("draft_verified_mismatches", 0) or 0)
        candidate_probe_summary = summary.get("candidate_probe_summary", {})
        if isinstance(candidate_probe_summary, dict):
            for layer_key, layer_stats in candidate_probe_summary.items():
                if not isinstance(layer_stats, dict):
                    continue
                agg = candidate_probe_sums.setdefault(
                    str(layer_key),
                    {
                        "probe_count": 0.0,
                        "weighted_max_prob_sum": 0.0,
                        "meets_importance_sum": 0.0,
                        "meets_confidence_sum": 0.0,
                        "exit_sum": 0.0,
                    },
                )
                probe_count = float(layer_stats.get("probe_count", 0) or 0)
                agg["probe_count"] += probe_count
                agg["weighted_max_prob_sum"] += float(layer_stats.get("avg_max_prob", 0.0)) * probe_count
                agg["meets_importance_sum"] += float(layer_stats.get("meets_importance_rate", 0.0)) * probe_count
                agg["meets_confidence_sum"] += float(layer_stats.get("meets_confidence_rate", 0.0)) * probe_count
                agg["exit_sum"] += float(layer_stats.get("exit_rate", 0.0)) * probe_count

        token_traces = trace.get("token_traces", [])
        if isinstance(token_traces, list):
            for token_trace in token_traces:
                if not isinstance(token_trace, dict):
                    continue
                layer = int(token_trace.get("exit_layer", trace_total_layers - 1 if trace_total_layers else 0))
                token_exit_layer_histogram[layer] += 1
                token_exit_layers.append(float(layer))

    if not sample_exit_layer_stats:
        return None

    candidate_probe_summary: dict[str, Any] = {}
    for layer_key, agg in sorted(candidate_probe_sums.items(), key=lambda item: int(item[0])):
        probe_count = max(1.0, agg["probe_count"])
        candidate_probe_summary[str(layer_key)] = {
            "probe_count": int(agg["probe_count"]),
            "avg_max_prob": agg["weighted_max_prob_sum"] / probe_count,
            "meets_importance_rate": agg["meets_importance_sum"] / probe_count,
            "meets_confidence_rate": agg["meets_confidence_sum"] / probe_count,
            "exit_rate": agg["exit_sum"] / probe_count,
        }

    return {
        "enabled": True,
        "trace_count": len(sample_exit_layer_stats),
        "total_layers": total_layers,
        "avg_exit_layer_per_sample": _mean(sample_avg_exit_layers),
        "avg_exit_layer_per_token": _mean(token_exit_layers),
        "exit_triggered_tokens": exit_triggered_tokens,
        "forced_cap_tokens": forced_cap_tokens,
        "string_guard_blocked_tokens": string_guard_blocked_tokens,
        "draft_only_candidate_tokens": draft_only_candidate_tokens,
        "draft_verified_matches": draft_verified_matches,
        "draft_verified_mismatches": draft_verified_mismatches,
        "fallback_used_count": fallback_used_count,
        "token_exit_layer_histogram": {
            str(layer): int(count) for layer, count in sorted(token_exit_layer_histogram.items())
        },
        "candidate_probe_summary": candidate_probe_summary,
        "sample_exit_layer_stats": sample_exit_layer_stats,
    }


@dataclass(frozen=True)
class AccuracyEvalConfig:
    test_file: Path | None = None
    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json")
    predictions_file: Path | None = None
    report_file: Path = Path("experiments/03_eval_exp/reports/accuracy_report.json")
    num_samples: int = 200
    seed: int = 42

    # Evaluation mode: "api" | "local"
    mode: str = "api"

    # API settings (mode=api)
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-5"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout: int = 120
    max_retries: int = 3
    sleep_seconds: float = 0.0

    # Local model settings (mode=local)
    model_path: str = ""
    tokenizer_path: str | None = None
    backend: str = "transformers"  # transformers | vllm | llama.cpp | exllamav2
    quantization: str | None = None
    max_new_tokens: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    vllm_dtype: str | None = None
    trust_remote_code: bool = True
    use_flash_attention: bool = False
    early_exit_enabled: bool = False
    exit_layers: str | list[int] | None = None
    tau_importance: float = 0.6
    tau_confidence: float = 0.9
    importance_file: str | None = None
    early_exit_warmup_tokens: int = 16
    early_exit_min_streak: int = 4
    early_exit_protect_open_string: bool = False
    early_exit_draft_only_layers: str | list[int] | None = None
    early_exit_fallback_on_invalid_json: bool = False

    # System prompt
    system_prompt: str = DEFAULT_EVAL_SYSTEM_PROMPT


def run_accuracy_eval(cfg: AccuracyEvalConfig) -> dict[str, Any]:
    """Run accuracy evaluation — dispatches to API or local engine."""
    effective_dataset_file = cfg.test_file or cfg.dataset_file
    if cfg.mode == "local" and cfg.model_path:
        return _run_local_accuracy_eval(cfg, dataset_file=effective_dataset_file)
    return evaluate_toolcall_accuracy(
        dataset_file=effective_dataset_file,
        predictions_file=cfg.predictions_file,
        report_file=cfg.report_file,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        api_base=cfg.api_base,
        model=cfg.model,
        api_key=cfg.api_key,
        api_key_env=cfg.api_key_env,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        sleep_seconds=cfg.sleep_seconds,
        system_prompt=cfg.system_prompt,
    )


def _run_local_accuracy_eval(cfg: AccuracyEvalConfig, *, dataset_file: Path) -> dict[str, Any]:
    """Evaluate with a local model, collecting VRAM & latency metrics."""
    from src.eval_core.inference_engines import build_inference_engine

    # Load dataset
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    valid_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        instruction = row.get("instruction", "")
        output_text = row.get("output", "")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        if not isinstance(output_text, str) or not output_text.strip():
            continue
        try:
            gt_commands = payload_to_commands(output_text)
        except Exception:
            continue
        valid_rows.append({
            "dataset_index": i,
            "instruction": instruction,
            "system": row.get("system", "") if isinstance(row.get("system", ""), str) else "",
            "gt_output": output_text,
            "gt_commands": gt_commands,
        })

    rng = random.Random(cfg.seed)
    selected = rng.sample(valid_rows, min(cfg.num_samples, len(valid_rows)))

    # Build local inference engine
    engine_cfg: dict[str, Any] = {
        "backend": cfg.backend,
        "model_path": cfg.model_path,
        "tokenizer_path": cfg.tokenizer_path,
        "quantization": cfg.quantization,
        "max_new_tokens": cfg.max_new_tokens,
        "max_model_len": cfg.max_model_len,
        "temperature": cfg.temperature,
        "trust_remote_code": cfg.trust_remote_code,
        "use_flash_attention": cfg.use_flash_attention,
        "early_exit_enabled": cfg.early_exit_enabled,
        "exit_layers": cfg.exit_layers,
        "tau_importance": cfg.tau_importance,
        "tau_confidence": cfg.tau_confidence,
        "importance_file": cfg.importance_file,
        "early_exit_warmup_tokens": cfg.early_exit_warmup_tokens,
        "early_exit_min_streak": cfg.early_exit_min_streak,
        "early_exit_protect_open_string": cfg.early_exit_protect_open_string,
        "early_exit_draft_only_layers": cfg.early_exit_draft_only_layers,
        "early_exit_fallback_on_invalid_json": cfg.early_exit_fallback_on_invalid_json,
    }
    if cfg.backend == "vllm":
        engine_cfg["gpu_memory_utilization"] = cfg.gpu_memory_utilization
        engine_cfg["vllm_dtype"] = cfg.vllm_dtype
    engine = build_inference_engine(engine_cfg)
    tokenizer_getter = getattr(engine, "get_tokenizer", None)
    engine_tokenizer = tokenizer_getter() if callable(tokenizer_getter) else None
    tokenize_with_offsets = getattr(engine, "tokenize_with_offsets", None)
    if not callable(tokenize_with_offsets):
        tokenize_with_offsets = None

    total = len(selected)
    parse_ok = 0
    exact_match = 0
    action_match = 0
    latencies: list[float] = []
    throughputs: list[float] = []
    peak_vrams: list[float] = []
    details: list[dict[str, Any]] = []

    logger.info("Local eval: %d samples with backend=%s model=%s", total, cfg.backend, cfg.model_path)

    for i, sample in enumerate(selected):
        messages = build_eval_messages(
            instruction=sample["instruction"],
            cfg_system_prompt=cfg.system_prompt,
            sample_system_prompt=sample.get("system", ""),
        )
        pred_text = ""
        infer_error: str | None = None
        perf: dict[str, Any] = {}
        early_exit_trace: dict[str, Any] | None = None
        early_exit_fallback_used = False
        early_exit_fallback_reason: str | None = None
        early_exit_pre_fallback_parse_failure_diagnostic: dict[str, Any] | None = None
        early_exit_pre_fallback_preview: str | None = None

        try:
            with time_and_memory_tracker(input_text=sample["instruction"]) as tracker:
                pred_text = engine.generate_chat(messages)
                early_exit_trace = _extract_generation_trace(engine)
                if cfg.early_exit_enabled and cfg.early_exit_fallback_on_invalid_json:
                    try:
                        payload_to_commands(pred_text)
                    except Exception as exc:
                        fallback_generate = getattr(engine, "generate_chat_without_early_exit", None)
                        if callable(fallback_generate):
                            early_exit_pre_fallback_preview = normalize_text(pred_text)[:200]
                            early_exit_pre_fallback_parse_failure_diagnostic = diagnose_parse_failure(
                                pred_text,
                                error_message=str(exc),
                                trace=early_exit_trace,
                                tokenizer=engine_tokenizer,
                                tokenize_with_offsets=tokenize_with_offsets,
                            )
                            pred_text = fallback_generate(messages)
                            early_exit_fallback_used = True
                            early_exit_fallback_reason = "invalid_json"
                tracker.set_output_text(pred_text)
            perf = tracker.metrics
        except Exception as exc:
            infer_error = f"{type(exc).__name__}: {exc}"
            perf = {"latency_sec": 0.0, "throughput_tps": 0.0, "peak_vram_mb": 0.0}
            early_exit_trace = _extract_generation_trace(engine)

        lat = float(perf.get("latency_sec", 0.0))
        tps = float(perf.get("throughput_tps", 0.0))
        vram = float(perf.get("peak_vram_mb", 0.0))
        latencies.append(lat)
        throughputs.append(tps)
        peak_vrams.append(vram)

        result: dict[str, Any] = {
            "dataset_index": sample["dataset_index"],
            "instruction": sample["instruction"],
            "prediction_preview": normalize_text(pred_text)[:200],
            "parse_ok": False,
            "exact_match": False,
            "action_match": False,
            "error": infer_error,
            "latency_sec": lat,
            "throughput_tps": tps,
            "peak_vram_mb": vram,
        }
        if early_exit_trace is not None:
            result["early_exit_trace"] = early_exit_trace
            summary = early_exit_trace.get("summary", {}) if isinstance(early_exit_trace.get("summary"), dict) else {}
            result["avg_exit_layer"] = float(summary.get("avg_exit_layer", 0.0))
            result["generated_tokens"] = int(early_exit_trace.get("tokens_generated", 0) or 0)
            result["early_exit_fallback_used"] = early_exit_fallback_used
            result["early_exit_fallback_reason"] = early_exit_fallback_reason
            if early_exit_pre_fallback_parse_failure_diagnostic is not None:
                result["early_exit_pre_fallback_parse_failure_diagnostic"] = (
                    early_exit_pre_fallback_parse_failure_diagnostic
                )
                result["early_exit_pre_fallback_preview"] = early_exit_pre_fallback_preview

        if infer_error is None:
            try:
                pred_cmds = payload_to_commands(pred_text)
                gt_cmds = sample["gt_commands"]
                pred_sig = [str(c.get("action", "")) for c in pred_cmds]
                gt_sig = [str(c.get("action", "")) for c in gt_cmds]
                result["parse_ok"] = True
                result["exact_match"] = canonicalize_commands(pred_cmds) == canonicalize_commands(gt_cmds)
                result["action_match"] = pred_sig == gt_sig
                parse_ok += 1
                if result["exact_match"]:
                    exact_match += 1
                if result["action_match"]:
                    action_match += 1
            except Exception as exc:
                result["error"] = str(exc)
                if pred_text.strip():
                    result["parse_failure_diagnostic"] = diagnose_parse_failure(
                        pred_text,
                        error_message=str(exc),
                        trace=early_exit_trace,
                        tokenizer=engine_tokenizer,
                        tokenize_with_offsets=tokenize_with_offsets,
                    )

        details.append(result)
        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d  parse_ok=%.1f%%  exact_match=%.1f%%  action_match=%.1f%%",
                i + 1, total,
                parse_ok / (i + 1) * 100,
                exact_match / (i + 1) * 100,
                action_match / (i + 1) * 100,
            )

    report = {
        "mode": "local",
        "model_path": cfg.model_path,
        "backend": cfg.backend,
        "quantization": cfg.quantization,
        "early_exit_enabled": cfg.early_exit_enabled,
        "exit_layers": cfg.exit_layers,
        "tau_importance": cfg.tau_importance,
        "tau_confidence": cfg.tau_confidence,
        "importance_file": cfg.importance_file,
        "early_exit_warmup_tokens": cfg.early_exit_warmup_tokens,
        "early_exit_min_streak": cfg.early_exit_min_streak,
        "early_exit_protect_open_string": cfg.early_exit_protect_open_string,
        "early_exit_draft_only_layers": cfg.early_exit_draft_only_layers,
        "early_exit_fallback_on_invalid_json": cfg.early_exit_fallback_on_invalid_json,
        "dataset_file": str(dataset_file),
        "seed": cfg.seed,
        "num_samples_evaluated": total,
        "num_valid_rows_in_dataset": len(valid_rows),
        "parse_ok": parse_ok,
        "parse_ok_rate": (parse_ok / total) if total else 0.0,
        "exact_match": exact_match,
        "exact_match_rate": (exact_match / total) if total else 0.0,
        "action_match": action_match,
        "action_match_rate": (action_match / total) if total else 0.0,
        "avg_latency_sec": _mean(latencies),
        "avg_throughput_tps": _mean(throughputs),
        "avg_peak_vram_mb": _mean(peak_vrams),
        "max_peak_vram_mb": max(peak_vrams) if peak_vrams else 0.0,
        "details": details,
    }
    early_exit_summary = _summarize_early_exit_results(details)
    if early_exit_summary is not None:
        report["early_exit"] = early_exit_summary
    parse_failure_summary = summarize_parse_failures(details)
    if parse_failure_summary is not None:
        report["parse_failure_diagnostics"] = parse_failure_summary
    early_exit_break_summary = summarize_parse_failures(
        details,
        diagnostic_key="early_exit_pre_fallback_parse_failure_diagnostic",
    )
    if early_exit_break_summary is not None:
        report["early_exit_parse_break_diagnostics"] = early_exit_break_summary

    cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_accuracy_from_merged_config(config: dict[str, Any]) -> dict[str, Any]:
    section = (
        config.get("test", {}).get("accuracy_eval", {})
        if isinstance(config.get("test"), dict)
        else {}
    )
    test_file_raw = section.get("test_file")
    dataset_file_raw = section.get("dataset_file", AccuracyEvalConfig.dataset_file)
    test_file = Path(test_file_raw) if test_file_raw else None

    leak_cfg = section.get("leakage_check", {}) if isinstance(section.get("leakage_check"), dict) else {}
    leakage_enabled = bool(leak_cfg.get("enabled", True))
    leakage_strict = bool(leak_cfg.get("strict", True))
    train_section = config.get("finetune", {}).get("train", {}) if isinstance(config.get("finetune"), dict) else {}
    train_file = Path(train_section["train_file"]).expanduser().resolve() if train_section.get("train_file") else None
    val_file = Path(train_section["val_file"]).expanduser().resolve() if train_section.get("val_file") else None
    effective_test_file = (test_file or Path(dataset_file_raw)).expanduser().resolve()
    if leakage_enabled:
        enforce_train_eval_no_leakage(
            train_file=train_file,
            val_file=val_file,
            test_file=effective_test_file,
            strict=leakage_strict,
            check_content_overlap=True,
        )

    cfg = AccuracyEvalConfig(
        test_file=test_file,
        dataset_file=Path(dataset_file_raw),
        predictions_file=Path(section["predictions_file"]) if section.get("predictions_file") else None,
        report_file=Path(section.get("report_file", AccuracyEvalConfig.report_file)),
        num_samples=int(section.get("num_samples", AccuracyEvalConfig.num_samples)),
        seed=int(section.get("seed", AccuracyEvalConfig.seed)),
        mode=str(section.get("mode", AccuracyEvalConfig.mode)),
        api_base=str(section.get("api_base", AccuracyEvalConfig.api_base)),
        model=str(section.get("model", AccuracyEvalConfig.model)),
        api_key=str(section.get("api_key", AccuracyEvalConfig.api_key)),
        api_key_env=str(section.get("api_key_env", AccuracyEvalConfig.api_key_env)),
        temperature=float(section.get("temperature", AccuracyEvalConfig.temperature)),
        max_tokens=int(section.get("max_tokens", AccuracyEvalConfig.max_tokens)),
        timeout=int(section.get("timeout", AccuracyEvalConfig.timeout)),
        max_retries=int(section.get("max_retries", AccuracyEvalConfig.max_retries)),
        sleep_seconds=float(section.get("sleep_seconds", AccuracyEvalConfig.sleep_seconds)),
        model_path=str(section.get("model_path", AccuracyEvalConfig.model_path)),
        tokenizer_path=str(section.get("tokenizer_path")) if section.get("tokenizer_path") else None,
        backend=str(section.get("backend", AccuracyEvalConfig.backend)),
        quantization=section.get("quantization", AccuracyEvalConfig.quantization),
        max_new_tokens=int(section.get("max_new_tokens", AccuracyEvalConfig.max_new_tokens)),
        max_model_len=int(section.get("max_model_len", AccuracyEvalConfig.max_model_len)),
        gpu_memory_utilization=float(section.get("gpu_memory_utilization", AccuracyEvalConfig.gpu_memory_utilization)),
        vllm_dtype=str(section.get("vllm_dtype")) if section.get("vllm_dtype") else None,
        trust_remote_code=bool(section.get("trust_remote_code", AccuracyEvalConfig.trust_remote_code)),
        use_flash_attention=bool(section.get("use_flash_attention", AccuracyEvalConfig.use_flash_attention)),
        early_exit_enabled=bool(section.get("early_exit_enabled", AccuracyEvalConfig.early_exit_enabled)),
        exit_layers=section.get("exit_layers", AccuracyEvalConfig.exit_layers),
        tau_importance=float(section.get("tau_importance", AccuracyEvalConfig.tau_importance)),
        tau_confidence=float(section.get("tau_confidence", AccuracyEvalConfig.tau_confidence)),
        importance_file=str(section.get("importance_file")) if section.get("importance_file") else None,
        early_exit_warmup_tokens=int(section.get("early_exit_warmup_tokens", AccuracyEvalConfig.early_exit_warmup_tokens)),
        early_exit_min_streak=int(section.get("early_exit_min_streak", AccuracyEvalConfig.early_exit_min_streak)),
        early_exit_protect_open_string=bool(section.get("early_exit_protect_open_string", AccuracyEvalConfig.early_exit_protect_open_string)),
        early_exit_draft_only_layers=section.get("early_exit_draft_only_layers", AccuracyEvalConfig.early_exit_draft_only_layers),
        early_exit_fallback_on_invalid_json=bool(section.get("early_exit_fallback_on_invalid_json", AccuracyEvalConfig.early_exit_fallback_on_invalid_json)),
        system_prompt=str(section.get("system_prompt", AccuracyEvalConfig.system_prompt)),
    )
    return run_accuracy_eval(cfg)
