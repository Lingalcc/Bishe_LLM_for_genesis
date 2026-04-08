#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.inference_engines import build_inference_engine
from src.eval_core.performance_monitor import time_and_memory_tracker


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_vals = sorted(float(v) for v in values)
    rank = (len(sorted_vals) - 1) * (percentile / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    weight = rank - lo
    return float(sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight)


def _build_batches(prompts: list[str], batch_size: int) -> list[list[str]]:
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


def _load_prompts_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"prompts_file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            prompts = [item for item in payload if isinstance(item, str) and item.strip()]
            return prompts
        if isinstance(payload, dict):
            rows = payload.get("prompts", [])
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, str) and item.strip()]

    if suffix == ".jsonl":
        prompts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                value = obj.get("prompt") or obj.get("instruction")
                if isinstance(value, str) and value.strip():
                    prompts.append(value)
        return prompts

    raise ValueError(f"Unsupported prompts file format: {path}")


def _expand_prompts(
    seed_prompts: list[str],
    num_samples: int,
    *,
    shuffle_prompts: bool,
    random_seed: int,
) -> list[str]:
    if not seed_prompts:
        raise ValueError("No prompts available for benchmark.")
    if not shuffle_prompts or len(seed_prompts) <= 1:
        return [seed_prompts[i % len(seed_prompts)] for i in range(num_samples)]

    rng = random.Random(random_seed)
    expanded: list[str] = []
    while len(expanded) < num_samples:
        cycle = list(seed_prompts)
        rng.shuffle(cycle)
        expanded.extend(cycle)
    return expanded[:num_samples]


def _generate_batch(engine: Any, prompts: list[str], use_chat: bool = False) -> list[str]:
    if use_chat:
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        if hasattr(engine, "generate_chat_batch") and callable(engine.generate_chat_batch):
            outputs = engine.generate_chat_batch(messages_batch)
        elif hasattr(engine, "generate_chat") and callable(engine.generate_chat):
            outputs = [engine.generate_chat(msgs) for msgs in messages_batch]
        else:
            raise TypeError("Engine does not support chat generation interface.")
    else:
        if hasattr(engine, "generate_batch") and callable(engine.generate_batch):
            outputs = engine.generate_batch(prompts)
        elif hasattr(engine, "generate") and callable(engine.generate):
            outputs = [engine.generate(p) for p in prompts]
        else:
            raise TypeError("Engine does not support generation interface.")

    normalized: list[str] = []
    for out in outputs:
        if isinstance(out, str):
            normalized.append(out)
        else:
            normalized.append(json.dumps(out, ensure_ascii=False))
    return normalized


def _count_tokens(engine: Any, text: str) -> int:
    normalized = str(text)
    if hasattr(engine, "count_tokens") and callable(engine.count_tokens):
        try:
            return max(0, int(engine.count_tokens(normalized)))
        except Exception:
            pass
    from src.eval_core.performance_monitor import estimate_tokens_from_text

    return max(0, int(estimate_tokens_from_text(normalized)))


def _render_inputs_for_metrics(engine: Any, prompts: list[str], use_chat: bool = False) -> list[str]:
    if not use_chat:
        if hasattr(engine, "render_batch_for_metrics") and callable(engine.render_batch_for_metrics):
            rendered = engine.render_batch_for_metrics(prompts)
            return [str(item) for item in rendered]
        return [str(prompt) for prompt in prompts]

    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    if hasattr(engine, "render_chat_batch_for_metrics") and callable(engine.render_chat_batch_for_metrics):
        rendered = engine.render_chat_batch_for_metrics(messages_batch)
        return [str(item) for item in rendered]
    if hasattr(engine, "render_chat_for_metrics") and callable(engine.render_chat_for_metrics):
        return [str(engine.render_chat_for_metrics(messages)) for messages in messages_batch]
    return [str(prompt) for prompt in prompts]


def _metric_as_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class InferenceBenchmarkConfig:
    backend: str
    model_path: str
    tokenizer_path: str | None = None
    quantization: str | None = None
    require_gpu: bool = False
    batch_size: int = 1
    num_samples: int = 32
    prompt: str = "Generate one short JSON action for robot arm control."
    prompts_file: str | None = None
    use_chat: bool = False
    warmup_batches: int = 1
    shuffle_prompts: bool = True
    random_seed: int = 42

    max_new_tokens: int = 128
    temperature: float = 0.0
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    vllm_dtype: str | None = None
    trust_remote_code: bool = True
    use_flash_attention: bool = False

    output_json: str = "experiments/03_eval_exp/reports/inference_benchmark.json"
    output_csv: str | None = None


def run_inference_benchmark(cfg: InferenceBenchmarkConfig, *, engine: Any | None = None) -> dict[str, Any]:
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if cfg.warmup_batches < 0:
        raise ValueError("warmup_batches must be >= 0")

    backend = str(cfg.backend).strip().lower()

    seed_prompts: list[str]
    if cfg.prompts_file:
        seed_prompts = _load_prompts_file(Path(cfg.prompts_file))
    else:
        seed_prompts = [cfg.prompt]

    prompts = _expand_prompts(
        seed_prompts,
        cfg.num_samples,
        shuffle_prompts=cfg.shuffle_prompts,
        random_seed=cfg.random_seed,
    )
    batches = _build_batches(prompts, cfg.batch_size)

    if engine is None:
        engine_cfg: dict[str, Any] = {
            "backend": backend,
            "model_path": cfg.model_path,
            "tokenizer_path": cfg.tokenizer_path,
            "quantization": cfg.quantization,
            "require_gpu": cfg.require_gpu,
            "max_new_tokens": cfg.max_new_tokens,
            "max_model_len": cfg.max_model_len,
            "temperature": cfg.temperature,
            "trust_remote_code": cfg.trust_remote_code,
            "use_flash_attention": cfg.use_flash_attention,
        }
        if backend == "vllm":
            engine_cfg["gpu_memory_utilization"] = cfg.gpu_memory_utilization
            engine_cfg["vllm_dtype"] = cfg.vllm_dtype
        engine = build_inference_engine(engine_cfg)

    warmup_limit = min(cfg.warmup_batches, len(batches))
    for batch in batches[:warmup_limit]:
        try:
            _generate_batch(engine, batch, use_chat=cfg.use_chat)
        except Exception:
            # Warmup failure should not crash benchmark; real pass will record errors.
            break

    batch_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    peak_memory_values: list[float] = []
    process_rss_values: list[float] = []
    ttft_values: list[float] = []
    throughput_tps_values: list[float] = []
    decode_tps_values: list[float] = []
    tpot_values: list[float] = []
    output_tokens_values: list[float] = []
    input_tokens_values: list[float] = []
    e2e_time_per_output_token_values: list[float] = []
    success_samples = 0
    failed_samples = 0
    error_rows: list[dict[str, Any]] = []
    token_count_method = str(getattr(engine, "token_count_method", "heuristic_estimate"))

    for batch_idx, batch in enumerate(batches):
        rendered_inputs = _render_inputs_for_metrics(engine, batch, use_chat=cfg.use_chat)
        batch_input_tokens = sum(_count_tokens(engine, text) for text in rendered_inputs)
        monitor = time_and_memory_tracker(
            input_text="\n".join(rendered_inputs),
            input_tokens=batch_input_tokens,
            model_config={
                "backend": backend,
                "quantization": cfg.quantization,
                "batch_size": len(batch),
                "token_count_method": token_count_method,
            },
        )

        status = "ok"
        error_msg: str | None = None
        outputs: list[str] = []

        try:
            with monitor:
                outputs = _generate_batch(engine, batch, use_chat=cfg.use_chat)
                monitor.set_output_text("\n".join(outputs))
                monitor.set_output_tokens(sum(_count_tokens(engine, output) for output in outputs))
            success_samples += len(outputs)
        except Exception as exc:
            status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"
            failed_samples += len(batch)
            error_rows.append({"batch_index": batch_idx, "error": error_msg})

        try:
            metrics = monitor.metrics
        except Exception:
            metrics = {
                "latency_sec": 0.0,
                "peak_vram_mb": 0.0,
                "process_rss_mb": 0.0,
                "ttft_sec": None,
                "ttft_available": False,
                "throughput_tps": 0.0,
                "decode_tps": None,
                "output_tokens": 0.0,
                "input_tokens": float(batch_input_tokens),
                "total_tokens": 0.0,
            }

        latency = float(metrics.get("latency_sec", 0.0))
        peak_mem = float(metrics.get("peak_vram_mb", 0.0))
        process_rss = float(metrics.get("process_rss_mb", 0.0))
        ttft_sec = _metric_as_float(metrics, "ttft_sec")
        throughput_tps = float(metrics.get("throughput_tps", 0.0))
        decode_tps = _metric_as_float(metrics, "decode_tps")
        tpot_sec = (1.0 / decode_tps) if decode_tps is not None and decode_tps > 0 else None
        output_tokens = float(metrics.get("output_tokens", 0.0))
        input_tokens = float(metrics.get("input_tokens", batch_input_tokens))
        total_tokens = float(metrics.get("total_tokens", 0.0))
        e2e_time_per_output_token_sec = (latency / output_tokens) if output_tokens > 0 and latency > 0 else None
        ttft_available = bool(metrics.get("ttft_available", ttft_sec is not None))

        if status == "ok":
            latencies.append(latency)
            peak_memory_values.append(peak_mem)
            process_rss_values.append(process_rss)
            throughput_tps_values.append(throughput_tps)
            output_tokens_values.append(output_tokens)
            input_tokens_values.append(input_tokens)
            if ttft_sec is not None:
                ttft_values.append(ttft_sec)
            if decode_tps is not None:
                decode_tps_values.append(decode_tps)
            if tpot_sec is not None:
                tpot_values.append(tpot_sec)
            if e2e_time_per_output_token_sec is not None:
                e2e_time_per_output_token_values.append(e2e_time_per_output_token_sec)

        batch_rows.append(
            {
                "batch_index": batch_idx,
                "batch_size": len(batch),
                "status": status,
                "latency_sec": latency,
                "peak_memory_mb": peak_mem,
                "process_rss_mb": process_rss,
                "ttft_sec": ttft_sec,
                "ttft_available": ttft_available,
                "throughput_tps": throughput_tps,
                "decode_tps": decode_tps,
                "tpot_sec": tpot_sec,
                "e2e_time_per_output_token_sec": e2e_time_per_output_token_sec,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "error": error_msg,
            }
        )

    total_latency = float(sum(latencies))
    throughput = (success_samples / total_latency) if total_latency > 0 else 0.0
    total_output_tokens = float(sum(output_tokens_values))
    total_input_tokens = float(sum(input_tokens_values))
    sample_throughput_sps = throughput
    token_throughput_tps = (total_output_tokens / total_latency) if total_latency > 0 else 0.0

    result = {
        "backend": backend,
        "quantization": cfg.quantization,
        "batch_size": cfg.batch_size,
        "num_samples": cfg.num_samples,
        "seed_prompt_count": len(seed_prompts),
        "shuffle_prompts": cfg.shuffle_prompts,
        "random_seed": cfg.random_seed,
        "prompt_sampling_strategy": "cycle_shuffle" if cfg.shuffle_prompts and len(seed_prompts) > 1 else "sequential_repeat",
        "token_count_method": token_count_method,
        "avg_latency": _safe_mean(latencies),
        "latency_stddev": _safe_stdev(latencies),
        "p50_latency": _percentile(latencies, 50),
        "p95_latency": _percentile(latencies, 95),
        "throughput": throughput,
        "sample_throughput_sps": sample_throughput_sps,
        "token_throughput_tps": token_throughput_tps,
        "avg_ttft_sec": _safe_mean(ttft_values) if ttft_values else None,
        "avg_throughput_tps": _safe_mean(throughput_tps_values),
        "avg_decode_tps": _safe_mean(decode_tps_values) if decode_tps_values else None,
        "avg_tpot_sec": _safe_mean(tpot_values) if tpot_values else None,
        "avg_e2e_time_per_output_token_sec": _safe_mean(e2e_time_per_output_token_values) if e2e_time_per_output_token_values else None,
        "avg_input_tokens": _safe_mean(input_tokens_values),
        "avg_output_tokens": _safe_mean(output_tokens_values),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "peak_memory": max(peak_memory_values) if peak_memory_values else 0.0,
        "avg_peak_memory": _safe_mean(peak_memory_values),
        "avg_process_rss_mb": _safe_mean(process_rss_values),
        "max_process_rss_mb": max(process_rss_values) if process_rss_values else 0.0,
        "ttft_observed_batches": len(ttft_values),
        "decode_observed_batches": len(decode_tps_values),
        "errors": len(error_rows),
        "successful_samples": success_samples,
        "failed_samples": failed_samples,
        "num_batches": len(batches),
        "batch_metrics": batch_rows,
        "error_details": error_rows,
    }

    out_json = Path(cfg.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.output_csv:
        out_csv = Path(cfg.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "batch_index",
                    "batch_size",
                    "status",
                    "latency_sec",
                    "peak_memory_mb",
                    "process_rss_mb",
                    "ttft_sec",
                    "ttft_available",
                    "throughput_tps",
                    "decode_tps",
                    "tpot_sec",
                    "e2e_time_per_output_token_sec",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(batch_rows)

    return result


def parse_args(argv: list[str] | None = None) -> InferenceBenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run local inference benchmark for multiple local backends.")
    parser.add_argument("--backend", required=True, choices=["transformers", "vllm", "llama.cpp", "exllamav2"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--prompt", type=str, default=InferenceBenchmarkConfig.prompt)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--use-chat", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--no-shuffle-prompts", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=str, default="experiments/03_eval_exp/reports/inference_benchmark.json")
    parser.add_argument("--output-csv", type=str, default=None)
    ns = parser.parse_args(argv)
    return InferenceBenchmarkConfig(
        backend=ns.backend,
        model_path=ns.model_path,
        tokenizer_path=ns.tokenizer_path,
        quantization=ns.quantization,
        require_gpu=bool(ns.require_gpu),
        batch_size=ns.batch_size,
        num_samples=ns.num_samples,
        prompt=ns.prompt,
        prompts_file=ns.prompts_file,
        use_chat=ns.use_chat,
        warmup_batches=ns.warmup_batches,
        shuffle_prompts=not ns.no_shuffle_prompts,
        random_seed=ns.random_seed,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        max_model_len=ns.max_model_len,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        trust_remote_code=bool(ns.trust_remote_code and not ns.no_trust_remote_code),
        use_flash_attention=bool(ns.use_flash_attention),
        output_json=ns.output_json,
        output_csv=ns.output_csv,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    result = run_inference_benchmark(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
