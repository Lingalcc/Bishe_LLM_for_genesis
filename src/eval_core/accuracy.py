from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval_core.evaluate_toolcall_accuracy import (
    canonicalize_commands,
    evaluate_toolcall_accuracy,
    payload_to_commands,
)
from src.eval_core.performance_monitor import time_and_memory_tracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccuracyEvalConfig:
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
    backend: str = "transformers"  # transformers | vllm
    quantization: str | None = None
    max_new_tokens: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True

    # System prompt
    system_prompt: str = (
        "你是 Franka 机械臂控制指令生成器。"
        "请把用户自然语言转换为可执行的 JSON action。"
        "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，"
        "你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
        "只输出 JSON，不要输出解释。"
    )


def run_accuracy_eval(cfg: AccuracyEvalConfig) -> dict[str, Any]:
    """Run accuracy evaluation — dispatches to API or local engine."""
    if cfg.mode == "local" and cfg.model_path:
        return _run_local_accuracy_eval(cfg)
    return evaluate_toolcall_accuracy(
        dataset_file=cfg.dataset_file,
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
    )


def _run_local_accuracy_eval(cfg: AccuracyEvalConfig) -> dict[str, Any]:
    """Evaluate with a local model, collecting VRAM & latency metrics."""
    from src.eval_core.inference_engines import build_inference_engine

    # Load dataset
    rows = json.loads(cfg.dataset_file.read_text(encoding="utf-8"))
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
            "gt_output": output_text,
            "gt_commands": gt_commands,
        })

    rng = random.Random(cfg.seed)
    selected = rng.sample(valid_rows, min(cfg.num_samples, len(valid_rows)))

    # Build local inference engine
    engine_cfg: dict[str, Any] = {
        "backend": cfg.backend,
        "model_path": cfg.model_path,
        "quantization": cfg.quantization,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.backend == "vllm":
        engine_cfg["max_model_len"] = cfg.max_model_len
        engine_cfg["gpu_memory_utilization"] = cfg.gpu_memory_utilization
    engine = build_inference_engine(engine_cfg)

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
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": sample["instruction"]},
        ]
        pred_text = ""
        infer_error: str | None = None
        perf: dict[str, Any] = {}

        try:
            with time_and_memory_tracker(input_text=sample["instruction"]) as tracker:
                pred_text = engine.generate_chat(messages)
                tracker.set_output_text(pred_text)
            perf = tracker.metrics
        except Exception as exc:
            infer_error = f"{type(exc).__name__}: {exc}"
            perf = {"latency_sec": 0.0, "throughput_tps": 0.0, "peak_vram_mb": 0.0}

        lat = float(perf.get("latency_sec", 0.0))
        tps = float(perf.get("throughput_tps", 0.0))
        vram = float(perf.get("peak_vram_mb", 0.0))
        latencies.append(lat)
        throughputs.append(tps)
        peak_vrams.append(vram)

        result: dict[str, Any] = {
            "dataset_index": sample["dataset_index"],
            "parse_ok": False,
            "exact_match": False,
            "action_match": False,
            "error": infer_error,
            "latency_sec": lat,
            "throughput_tps": tps,
            "peak_vram_mb": vram,
        }

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

        details.append(result)
        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d  parse_ok=%.1f%%  exact_match=%.1f%%  action_match=%.1f%%",
                i + 1, total,
                parse_ok / (i + 1) * 100,
                exact_match / (i + 1) * 100,
                action_match / (i + 1) * 100,
            )

    _mean = lambda vs: sum(vs) / len(vs) if vs else 0.0
    report = {
        "mode": "local",
        "model_path": cfg.model_path,
        "backend": cfg.backend,
        "quantization": cfg.quantization,
        "dataset_file": str(cfg.dataset_file),
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

    cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_accuracy_from_merged_config(config: dict[str, Any]) -> dict[str, Any]:
    section = (
        config.get("test", {}).get("accuracy_eval", {})
        if isinstance(config.get("test"), dict)
        else {}
    )
    cfg = AccuracyEvalConfig(
        dataset_file=Path(section.get("dataset_file", AccuracyEvalConfig.dataset_file)),
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
        backend=str(section.get("backend", AccuracyEvalConfig.backend)),
        quantization=section.get("quantization", AccuracyEvalConfig.quantization),
        max_new_tokens=int(section.get("max_new_tokens", AccuracyEvalConfig.max_new_tokens)),
        max_model_len=int(section.get("max_model_len", AccuracyEvalConfig.max_model_len)),
        gpu_memory_utilization=float(section.get("gpu_memory_utilization", AccuracyEvalConfig.gpu_memory_utilization)),
        trust_remote_code=bool(section.get("trust_remote_code", AccuracyEvalConfig.trust_remote_code)),
        system_prompt=str(section.get("system_prompt", AccuracyEvalConfig.system_prompt)),
    )
    return run_accuracy_eval(cfg)
