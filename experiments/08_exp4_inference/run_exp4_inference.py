#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval_core.inference_benchmark import InferenceBenchmarkConfig, run_inference_benchmark
from src.eval_core.inference_engines import build_inference_engine
from src.eval_core.performance_monitor import time_and_memory_tracker


REPORTS_DIR = EXPERIMENT_DIR / "reports"
PROMPTS_PATH = EXPERIMENT_DIR / "prompts" / "default_prompts.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct"


@dataclass(frozen=True)
class QuantizationCase:
    name: str
    quantization: str | None
    dtype_label: str


CASES: tuple[QuantizationCase, ...] = (
    QuantizationCase(name="4bit", quantization="4bit", dtype_label="bitsandbytes-nf4"),
    QuantizationCase(name="8bit", quantization="8bit", dtype_label="bitsandbytes-int8"),
    QuantizationCase(name="16bit", quantization=None, dtype_label="float16"),
)


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ensure_dirs(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "logs").mkdir(parents=True, exist_ok=True)


def _quantization_label(quantization: str | None) -> str:
    return "16bit" if quantization in {None, "", "none"} else str(quantization)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _load_case_names(raw: str) -> list[str]:
    allowed = {case.name for case in CASES}
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("至少需要指定一个量化实验项。")
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"不支持的量化项: {invalid}，可选值: {sorted(allowed)}")
    return items


def _resolve_cases(case_names: list[str]) -> list[QuantizationCase]:
    mapping = {case.name: case for case in CASES}
    return [mapping[name] for name in case_names]


def _load_prompts(prompts_file: str | None) -> list[str]:
    path = Path(prompts_file) if prompts_file else PROMPTS_PATH
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"prompts 文件必须是非空 JSON 列表: {path}")
    prompts = [str(item).strip() for item in payload if str(item).strip()]
    if not prompts:
        raise ValueError(f"prompts 文件中没有可用文本: {path}")
    return prompts


def _build_prompts_report(prompts: list[str], output_path: Path) -> None:
    output_path.write_text(json.dumps({"prompts": prompts}, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_relative_metrics(rows: list[dict[str, Any]], baseline_name: str = "16bit") -> None:
    baseline = next((row for row in rows if row["precision"] == baseline_name), None)
    if baseline is None:
        for row in rows:
            row["latency_speedup_vs_16bit"] = None
            row["memory_saving_vs_16bit_pct"] = None
            row["load_memory_saving_vs_16bit_pct"] = None
        return

    base_latency = float(baseline.get("avg_latency_sec", 0.0))
    base_mem = float(baseline.get("max_infer_peak_vram_mb", 0.0))
    base_load_mem = float(baseline.get("load_peak_vram_mb", 0.0))

    for row in rows:
        latency = float(row.get("avg_latency_sec", 0.0))
        infer_mem = float(row.get("max_infer_peak_vram_mb", 0.0))
        load_mem = float(row.get("load_peak_vram_mb", 0.0))
        row["latency_speedup_vs_16bit"] = (base_latency / latency) if latency > 0 and base_latency > 0 else None
        row["memory_saving_vs_16bit_pct"] = ((base_mem - infer_mem) / base_mem * 100.0) if base_mem > 0 else None
        row["load_memory_saving_vs_16bit_pct"] = (
            ((base_load_mem - load_mem) / base_load_mem * 100.0) if base_load_mem > 0 else None
        )


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("没有可写入 CSV 的结果。")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_markdown_summary(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    backend: str,
    batch_size: int,
    num_samples: int,
    prompts_file: str,
) -> str:
    lines = [
        "# 实验08 Exp4 推理量化对比",
        "",
        f"- 模型：`{model_path}`",
        f"- 后端：`{backend}`",
        f"- batch size：`{batch_size}`",
        f"- 样本数：`{num_samples}`",
        f"- prompts：`{prompts_file}`",
        "",
        "| 量化 | 平均延迟(s) | P95延迟(s) | 样本吞吐(samples/s) | Token吞吐(tokens/s) | 峰值显存(MB) | 加载峰值显存(MB) | 相对16bit延迟加速比 | 相对16bit显存节省 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = row.get("latency_speedup_vs_16bit")
        mem_saving = row.get("memory_saving_vs_16bit_pct")
        lines.append(
            "| {precision} | {avg_latency_sec:.4f} | {p95_latency_sec:.4f} | {sample_throughput_sps:.3f} | "
            "{token_throughput_tps:.3f} | {max_infer_peak_vram_mb:.1f} | {load_peak_vram_mb:.1f} | "
            "{speedup} | {mem_saving} |".format(
                precision=row["precision"],
                avg_latency_sec=float(row["avg_latency_sec"]),
                p95_latency_sec=float(row["p95_latency_sec"]),
                sample_throughput_sps=float(row["sample_throughput_sps"]),
                token_throughput_tps=float(row["token_throughput_tps"]),
                max_infer_peak_vram_mb=float(row["max_infer_peak_vram_mb"]),
                load_peak_vram_mb=float(row["load_peak_vram_mb"]),
                speedup=f"{float(speedup):.3f}x" if speedup is not None else "-",
                mem_saving=f"{float(mem_saving):.1f}%" if mem_saving is not None else "-",
            )
        )
    return "\n".join(lines) + "\n"


def _plot_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [str(row["precision"]) for row in rows]
    avg_latency = [float(row["avg_latency_sec"]) for row in rows]
    p95_latency = [float(row["p95_latency_sec"]) for row in rows]
    sample_tps = [float(row["sample_throughput_sps"]) for row in rows]
    token_tps = [float(row["token_throughput_tps"]) for row in rows]
    infer_peak = [float(row["max_infer_peak_vram_mb"]) for row in rows]
    load_peak = [float(row["load_peak_vram_mb"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Exp4 Inference Benchmark: 3B Quantization Comparison", fontsize=16)

    axes[0, 0].bar(labels, avg_latency, color=["#d95f02", "#7570b3", "#1b9e77"])
    axes[0, 0].plot(labels, p95_latency, color="#111111", marker="o", linewidth=1.8, label="P95")
    axes[0, 0].set_title("Latency")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].legend()

    axes[0, 1].bar(labels, sample_tps, color=["#66a61e", "#e6ab02", "#1f78b4"], label="samples/s")
    axes[0, 1].plot(labels, token_tps, color="#c51b7d", marker="o", linewidth=1.8, label="tokens/s")
    axes[0, 1].set_title("Throughput")
    axes[0, 1].set_ylabel("Throughput")
    axes[0, 1].legend()

    axes[1, 0].bar(labels, infer_peak, color=["#8da0cb", "#fc8d62", "#66c2a5"])
    axes[1, 0].set_title("Inference Peak VRAM")
    axes[1, 0].set_ylabel("MB")

    axes[1, 1].bar(labels, load_peak, color=["#a6d854", "#ffd92f", "#e78ac3"])
    axes[1, 1].set_title("Model Load Peak VRAM")
    axes[1, 1].set_ylabel("MB")

    for ax in axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _summarize_case(case: QuantizationCase, benchmark_result: dict[str, Any], load_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": case.name,
        "quantization": case.quantization or "none",
        "dtype": case.dtype_label,
        "avg_latency_sec": float(benchmark_result.get("avg_latency", 0.0)),
        "p50_latency_sec": float(benchmark_result.get("p50_latency", 0.0)),
        "p95_latency_sec": float(benchmark_result.get("p95_latency", 0.0)),
        "sample_throughput_sps": float(benchmark_result.get("sample_throughput_sps", benchmark_result.get("throughput", 0.0))),
        "token_throughput_tps": float(benchmark_result.get("token_throughput_tps", 0.0)),
        "avg_token_throughput_tps": float(benchmark_result.get("avg_throughput_tps", 0.0)),
        "avg_decode_tps": float(benchmark_result.get("avg_decode_tps", 0.0)),
        "avg_ttft_sec": float(benchmark_result.get("avg_ttft_sec", 0.0)),
        "avg_output_tokens": float(benchmark_result.get("avg_output_tokens", 0.0)),
        "total_output_tokens": float(benchmark_result.get("total_output_tokens", 0.0)),
        "avg_infer_peak_vram_mb": float(benchmark_result.get("avg_peak_memory", 0.0)),
        "max_infer_peak_vram_mb": float(benchmark_result.get("peak_memory", 0.0)),
        "avg_process_rss_mb": float(benchmark_result.get("avg_process_rss_mb", 0.0)),
        "max_process_rss_mb": float(benchmark_result.get("max_process_rss_mb", 0.0)),
        "load_time_sec": float(load_metrics.get("latency_sec", 0.0)),
        "load_peak_vram_mb": float(load_metrics.get("peak_vram_mb", 0.0)),
        "load_process_rss_mb": float(load_metrics.get("process_rss_mb", 0.0)),
        "errors": int(benchmark_result.get("errors", 0)),
        "successful_samples": int(benchmark_result.get("successful_samples", 0)),
        "failed_samples": int(benchmark_result.get("failed_samples", 0)),
        "num_batches": int(benchmark_result.get("num_batches", 0)),
    }


def _default_engine_builder(engine_cfg: dict[str, Any]) -> Any:
    return build_inference_engine(engine_cfg)


def _default_benchmark_runner(cfg: InferenceBenchmarkConfig, *, engine: Any) -> dict[str, Any]:
    return run_inference_benchmark(cfg, engine=engine)


def run_quantization_experiment(
    *,
    model_path: str,
    backend: str,
    case_names: list[str],
    batch_size: int,
    num_samples: int,
    prompts_file: str | None,
    max_new_tokens: int,
    temperature: float,
    warmup_batches: int,
    use_chat: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    reports_dir: Path | None = None,
    engine_builder: Callable[[dict[str, Any]], Any] = _default_engine_builder,
    benchmark_runner: Callable[[InferenceBenchmarkConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = reports_dir or REPORTS_DIR
    _ensure_dirs(output_dir)
    _cleanup_cuda()

    prompts = _load_prompts(prompts_file)
    prompt_report_path = output_dir / "exp4_prompts_used.json"
    _build_prompts_report(prompts, prompt_report_path)

    selected_cases = _resolve_cases(case_names)
    rows: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []

    if benchmark_runner is None:
        def _runner(cfg: InferenceBenchmarkConfig) -> dict[str, Any]:
            engine_cfg = {
                "backend": backend,
                "model_path": model_path,
                "quantization": cfg.quantization,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "trust_remote_code": trust_remote_code,
            }
            if backend == "vllm":
                engine_cfg["max_model_len"] = max_model_len
                engine_cfg["gpu_memory_utilization"] = gpu_memory_utilization

            with time_and_memory_tracker(model_config={"stage": "load", "quantization": cfg.quantization}) as loader:
                engine = engine_builder(engine_cfg)
                loader.set_output_text("engine_ready")
            load_metrics = loader.metrics
            try:
                benchmark_result = run_inference_benchmark(cfg, engine=engine)
            finally:
                del engine
                gc.collect()
                _cleanup_cuda()
            return {"benchmark_result": benchmark_result, "load_metrics": load_metrics}
    else:
        def _runner(cfg: InferenceBenchmarkConfig) -> dict[str, Any]:
            wrapped = benchmark_runner(cfg)
            if "benchmark_result" not in wrapped or "load_metrics" not in wrapped:
                raise ValueError("自定义 benchmark_runner 必须返回包含 benchmark_result 和 load_metrics 的字典。")
            return wrapped

    for case in selected_cases:
        quant_label = _quantization_label(case.quantization)
        per_case_json = output_dir / f"exp4_{quant_label}_benchmark.json"
        per_case_csv = output_dir / f"exp4_{quant_label}_batches.csv"
        cfg = InferenceBenchmarkConfig(
            backend=backend,
            model_path=model_path,
            quantization=case.quantization,
            batch_size=batch_size,
            num_samples=num_samples,
            prompts_file=str(prompt_report_path),
            use_chat=use_chat,
            warmup_batches=warmup_batches,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            output_json=str(per_case_json),
            output_csv=str(per_case_csv),
        )
        result_bundle = _runner(cfg)
        benchmark_result = dict(result_bundle["benchmark_result"])
        load_metrics = dict(result_bundle["load_metrics"])
        summary_row = _summarize_case(case, benchmark_result, load_metrics)
        case_reports.append(
            {
                "case": case.name,
                "quantization": case.quantization,
                "dtype": case.dtype_label,
                "load_metrics": load_metrics,
                "benchmark_result": benchmark_result,
            }
        )
        rows.append(summary_row)

    _compute_relative_metrics(rows)

    csv_path = output_dir / "exp4_inference_results.csv"
    summary_json_path = output_dir / "exp4_inference_summary.json"
    summary_md_path = output_dir / "exp4_inference_summary.md"
    chart_path = output_dir / "exp4_inference_dashboard.png"

    _write_csv(rows, csv_path)
    _plot_results(rows, chart_path)

    summary = {
        "experiment": "08_exp4_inference",
        "model_path": model_path,
        "backend": backend,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "warmup_batches": warmup_batches,
        "use_chat": use_chat,
        "prompts_file": _repo_relative(prompt_report_path),
        "results_csv": _repo_relative(csv_path),
        "dashboard_png": _repo_relative(chart_path),
        "rows": rows,
        "cases": case_reports,
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        _build_markdown_summary(
            rows,
            model_path=model_path,
            backend=backend,
            batch_size=batch_size,
            num_samples=num_samples,
            prompts_file=_repo_relative(prompt_report_path),
        ),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验08 Exp4：统计不同量化下 3B 模型推理速度与资源占用。")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--backend", type=str, choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--cases", type=str, default="4bit,8bit,16bit")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--prompts-file", type=str, default=str(PROMPTS_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--use-chat", action="store_true", default=True)
    parser.add_argument("--no-use-chat", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    case_names = _load_case_names(args.cases)
    summary = run_quantization_experiment(
        model_path=args.model_path,
        backend=args.backend,
        case_names=case_names,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        prompts_file=args.prompts_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup_batches=args.warmup_batches,
        use_chat=bool(args.use_chat and not args.no_use_chat),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=bool(args.trust_remote_code and not args.no_trust_remote_code),
        reports_dir=Path(args.reports_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
