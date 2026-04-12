#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TEMP_DIR = EXPERIMENT_DIR / ".cache"
DEFAULT_BENCHMARK_PROMPTS = REPO_ROOT / "experiments" / "11_exp7_vllm" / "prompts" / "default_prompts.json"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_TEST_FILE = REPO_ROOT / "data_prepare" / "splits" / "test.json"
DEFAULT_DATASET_FILE = REPO_ROOT / "data_prepare" / "genesis_franka_toolcall_alpaca.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import configure_report_matplotlib, pick_plot_text
from src.utils.run_meta import record_run_meta


CASE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Transformers_16bit",
        "backend": "transformers",
        "runtime_quantization": None,
        "vllm_dtype": None,
        "gpu_memory_utilization": 0.9,
        "report_quantization": "16bit",
        "artifact_key": "merged_fp16",
        "stack_label": "Transformers + FP16",
        "format_label": "HF Safetensors",
        "quant_note": "Transformers 直接以 float16 加载 merged 模型，作为未量化基线。",
    },
    {
        "name": "vLLM_AWQ_CompressedTensors",
        "backend": "vllm",
        "runtime_quantization": "compressed-tensors",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.8,
        "report_quantization": "awq",
        "artifact_key": "merged_awq",
        "stack_label": "vLLM + AWQ",
        "format_label": "Compressed Tensors (AWQ)",
        "quant_note": "使用 llmcompressor 导出的 AWQ 压缩目录，由 vLLM 按 compressed-tensors 方式加载。",
    },
]


MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "merged_fp16": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
    "merged_awq": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-awq",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-awq",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
}


def build_override_artifacts(*, fp16_model_path: Path | None, awq_model_path: Path | None) -> dict[str, dict[str, Any]]:
    artifacts = {key: dict(value) for key, value in MODEL_ARTIFACTS.items()}
    if fp16_model_path is not None:
        resolved_fp16 = fp16_model_path.resolve()
        artifacts["merged_fp16"]["model_path"] = resolved_fp16
        artifacts["merged_fp16"]["tokenizer_path"] = resolved_fp16
    if awq_model_path is not None:
        resolved_awq = awq_model_path.resolve()
        artifacts["merged_awq"]["model_path"] = resolved_awq
        artifacts["merged_awq"]["tokenizer_path"] = resolved_awq
    return artifacts


def _load_exp7_module() -> Any:
    module_path = REPO_ROOT / "experiments" / "11_exp7_vllm" / "run_exp7_vllm_benchmark.py"
    spec = importlib.util.spec_from_file_location("exp7_vllm_benchmark_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 exp7 模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP7 = _load_exp7_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验 18 Exp14：AWQ 压缩模型与 Transformers 基线的速度和准确率对比。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--fp16-model-path", type=Path, default=None, help="Transformers 16bit 基线模型目录。")
    parser.add_argument("--awq-model-path", type=Path, default=None, help="vLLM + AWQ 对应的压缩模型目录。")
    parser.add_argument("--gpu-id", type=int, default=None, help="nvidia-smi 监控的物理 GPU 编号，默认自动推断。")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--benchmark-prompts-file", type=Path, default=DEFAULT_BENCHMARK_PROMPTS)
    parser.add_argument("--benchmark-num-samples", type=int, default=20)
    parser.add_argument("--accuracy-num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--accuracy-seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=int, default=10)
    parser.add_argument("--vram-poll-interval", type=float, default=0.2)
    parser.add_argument("--auto-install-deps", action="store_true", help="缺少依赖时自动执行 pip install。")
    parser.add_argument("--auto-download-missing-models", action="store_true", help="缺少模型资产时尝试自动下载。")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "")
    return parser.parse_args(argv)


def build_case_matrix(*, fp16_model_path: Path | None = None, awq_model_path: Path | None = None) -> list[dict[str, Any]]:
    artifacts = build_override_artifacts(fp16_model_path=fp16_model_path, awq_model_path=awq_model_path)
    cases: list[dict[str, Any]] = []
    for cfg in CASE_CONFIGS:
        case = dict(cfg)
        case["artifact"] = dict(artifacts[str(cfg["artifact_key"])])
        cases.append(case)
    return cases


def _fmt_metric(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    text = str(value).strip()
    return text or "-"


def draw_figures(df: pd.DataFrame, *, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((TEMP_DIR / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_report_matplotlib(matplotlib)

    try:
        import seaborn as sns
    except Exception:
        sns = None

    EXP7.ensure_dir(output_dir)
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    labels = df["Name"].tolist()
    x = list(range(len(df)))

    fig1, ax1 = plt.subplots(figsize=(14, 7))
    lat_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Avg Latency (s)"].tolist()]
    ax1.bar(labels, lat_values, color="#4c78a8")
    ax1.set_title(pick_plot_text("图1：AWQ 与 Transformers 延迟对比", "Figure 1: AWQ vs Transformers Latency"))
    ax1.set_xlabel(pick_plot_text("方案", "Case"))
    ax1.set_ylabel("Seconds")
    ax1.tick_params(axis="x", rotation=12)
    fig1.tight_layout()
    fig1.savefig(output_dir / "exp14_awq_vs_transformers_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    token_tps = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Token Throughput (tokens/s)"].tolist()]
    ax2.bar(labels, token_tps, color="#54a24b")
    ax2.set_title(pick_plot_text("图2：AWQ 与 Transformers 吞吐对比", "Figure 2: AWQ vs Transformers Throughput"))
    ax2.set_xlabel(pick_plot_text("方案", "Case"))
    ax2.set_ylabel("Tokens / Second")
    ax2.tick_params(axis="x", rotation=12)
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp14_awq_vs_transformers_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    width = 0.22
    exact_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Exact Match Rate"].tolist()]
    action_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Action Match Rate"].tolist()]
    parse_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Parse OK Rate"].tolist()]
    ax3.bar([item - width for item in x], parse_values, width=width, color="#72b7b2", label="Parse OK")
    ax3.bar(x, exact_values, width=width, color="#f58518", label="Exact Match")
    ax3.bar([item + width for item in x], action_values, width=width, color="#e45756", label="Action Match")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=12)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title(pick_plot_text("图3：AWQ 与 Transformers 准确率对比", "Figure 3: AWQ vs Transformers Accuracy"))
    ax3.set_xlabel(pick_plot_text("方案", "Case"))
    ax3.set_ylabel("Rate")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp14_awq_vs_transformers_accuracy_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(14, 7))
    benchmark_vram = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Peak VRAM (MB)"].tolist()]
    accuracy_vram = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Accuracy Max Peak VRAM (MB)"].tolist()]
    ax4.bar([item - 0.16 for item in x], benchmark_vram, width=0.32, color="#b279a2", label="Benchmark Peak VRAM")
    ax4.bar([item + 0.16 for item in x], accuracy_vram, width=0.32, color="#bab0ab", label="Accuracy Max VRAM")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=12)
    ax4.set_title(pick_plot_text("图4：AWQ 与 Transformers 显存对比", "Figure 4: AWQ vs Transformers VRAM"))
    ax4.set_xlabel(pick_plot_text("方案", "Case"))
    ax4.set_ylabel("MB")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(output_dir / "exp14_awq_vs_transformers_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)


def write_markdown_report(df: pd.DataFrame, *, output_path: Path) -> None:
    EXP7.ensure_dir(output_path.parent)
    lines: list[str] = [
        "# Exp14 AWQ 与 Transformers 对比实验报告",
        "",
        "## 实验目标",
        "",
        "- 对比 `Transformers 16bit` 基线与 `vLLM + AWQ` 压缩目录在同一任务上的速度、显存和准确率表现。",
        "- AWQ 模型来自 `llmcompressor` 导出的 `compressed-tensors` 目录，由项目内 vLLM 兼容层自动识别并加载。",
        "",
        "## 对比矩阵",
        "",
        "- `Transformers_16bit`：未量化基线。",
        "- `vLLM_AWQ_CompressedTensors`：AWQ 压缩部署方案。",
        "",
        "## 速度结果",
        "",
        "| 方案 | Backend | 载入格式 | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {backend} | {fmt} | {avg} | {p50} | {p95} | {sps} | {tps} | {vram} | {status} |".format(
                name=row["Name"],
                backend=row["Backend"],
                fmt=row["Model Format"],
                avg=_fmt_metric(row["Benchmark Avg Latency (s)"]),
                p50=_fmt_metric(row["Benchmark P50 Latency (s)"]),
                p95=_fmt_metric(row["Benchmark P95 Latency (s)"]),
                sps=_fmt_metric(row["Benchmark Sample Throughput (samples/s)"]),
                tps=_fmt_metric(row["Benchmark Token Throughput (tokens/s)"]),
                vram=_fmt_metric(row["Benchmark Peak VRAM (MB)"], digits=2),
                status=row["Benchmark Status"],
            )
        )

    lines.extend(
        [
            "",
            "## 精度结果",
            "",
            "| 方案 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | Accuracy Max VRAM (MB) | 状态 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {parse} | {exact} | {action} | {lat} | {tps} | {vram} | {status} |".format(
                name=row["Name"],
                parse=_fmt_metric(row["Parse OK Rate"]),
                exact=_fmt_metric(row["Exact Match Rate"]),
                action=_fmt_metric(row["Action Match Rate"]),
                lat=_fmt_metric(row["Accuracy Avg Latency (s)"]),
                tps=_fmt_metric(row["Accuracy Avg Throughput (tokens/s)"]),
                vram=_fmt_metric(row["Accuracy Max Peak VRAM (MB)"], digits=2),
                status=row["Accuracy Status"],
            )
        )

    success_df = df[df["Overall Status"] == "success"].copy()
    lines.extend(["", "## 结果分析", ""])
    if success_df.empty:
        lines.append("- 当前没有方案同时完成 benchmark 与 accuracy，请优先检查对应日志。")
    else:
        fastest_row = success_df.sort_values("Benchmark Avg Latency (s)", ascending=True).iloc[0]
        best_exact_row = success_df.sort_values("Exact Match Rate", ascending=False).iloc[0]
        best_action_row = success_df.sort_values("Action Match Rate", ascending=False).iloc[0]
        lines.append(
            f"- 速度最优方案为 `{fastest_row['Name']}`，benchmark 平均延迟为 `{float(fastest_row['Benchmark Avg Latency (s)']):.4f}s`。"
        )
        lines.append(
            f"- `Exact Match` 最高方案为 `{best_exact_row['Name']}`，精确匹配率为 `{float(best_exact_row['Exact Match Rate']):.4f}`。"
        )
        lines.append(
            f"- `Action Match` 最高方案为 `{best_action_row['Name']}`，动作匹配率为 `{float(best_action_row['Action Match Rate']):.4f}`。"
        )
        lines.append("- 如果速度领先和准确率领先不是同一个方案，就说明当前部署仍存在明确权衡。")

    lines.extend(
        [
            "",
            "## 解读建议",
            "",
            "- 如果你关注部署时延，优先看 `Benchmark Avg Latency (s)` 和 `Benchmark Token Throughput (tokens/s)`。",
            "- 如果你关注任务可用性，优先看 `Exact Match Rate` 和 `Action Match Rate`。",
            "- 当前 AWQ 目录实际由 `llmcompressor` 导出，因此这里的结论应理解为“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    EXP7.RESULTS_DIR = args.results_dir
    EXP7.LOGS_DIR = LOGS_DIR
    EXP7.TEMP_DIR = TEMP_DIR

    EXP7.ensure_dir(args.results_dir)
    EXP7.ensure_dir(LOGS_DIR)
    EXP7.ensure_dir(TEMP_DIR)

    gpu_id = EXP7.infer_gpu_id(args.gpu_id)
    EXP7.print_info(
        "Exp14 对比实验开始："
        f" GPU={gpu_id}, benchmark_samples={args.benchmark_num_samples}, accuracy_samples={args.accuracy_num_samples}"
    )

    rows: list[dict[str, Any]] = []
    cases = build_case_matrix(fp16_model_path=args.fp16_model_path, awq_model_path=args.awq_model_path)
    for case_cfg in cases:
        ok, reason = EXP7.prepare_case(case_cfg, args=args)
        if not ok:
            EXP7.print_error(f"[{case_cfg['name']}] 预检查失败：{reason}")
            rows.append(
                {
                    "Name": case_cfg["name"],
                    "Stack Label": case_cfg["stack_label"],
                    "Backend": case_cfg["backend"],
                    "Quantization": case_cfg["report_quantization"],
                    "Runtime Quantization": str(case_cfg.get("runtime_quantization") or ""),
                    "Model Format": case_cfg["format_label"],
                    "Quantization Note": case_cfg["quant_note"],
                    "Benchmark Num Samples": 0,
                    "Benchmark Avg Latency (s)": math.nan,
                    "Benchmark P50 Latency (s)": math.nan,
                    "Benchmark P95 Latency (s)": math.nan,
                    "Benchmark Sample Throughput (samples/s)": math.nan,
                    "Benchmark Token Throughput (tokens/s)": math.nan,
                    "Benchmark Peak VRAM (MB)": math.nan,
                    "Benchmark Avg Process RSS (MB)": math.nan,
                    "Accuracy Num Samples": 0,
                    "Parse OK Rate": math.nan,
                    "Exact Match Rate": math.nan,
                    "Action Match Rate": math.nan,
                    "Accuracy Avg Latency (s)": math.nan,
                    "Accuracy Avg Throughput (tokens/s)": math.nan,
                    "Accuracy Avg Peak VRAM (MB)": math.nan,
                    "Accuracy Max Peak VRAM (MB)": math.nan,
                    "Benchmark Status": "precheck_failed",
                    "Accuracy Status": "precheck_failed",
                    "Overall Status": reason,
                    "Benchmark Report": "",
                    "Accuracy Report": "",
                    "Model Path": str(Path(case_cfg["artifact"]["model_path"]).resolve()),
                    "Tokenizer Path": str(Path(case_cfg["artifact"]["tokenizer_path"]).resolve()) if case_cfg["artifact"].get("tokenizer_path") else "",
                }
            )
            continue
        rows.append(EXP7.run_single_case(case_cfg, args=args, results_dir=args.results_dir, gpu_id=gpu_id))

    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "exp14_awq_vs_transformers_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    draw_figures(df, output_dir=args.results_dir)

    markdown_path = args.results_dir / "exp14_awq_vs_transformers_report.md"
    write_markdown_report(df, output_path=markdown_path)

    success_df = df[df["Overall Status"] == "success"].copy()
    summary_payload = {
        "experiment": "exp14_awq_vs_transformers",
        "comparison_scope": "Transformers 16bit vs vLLM AWQ (compressed-tensors)",
        "fp16_model_path": str(Path(cases[0]["artifact"]["model_path"]).resolve()),
        "awq_model_path": str(Path(cases[1]["artifact"]["model_path"]).resolve()),
        "benchmark_num_samples": int(args.benchmark_num_samples),
        "accuracy_num_samples": int(args.accuracy_num_samples),
        "gpu_id": gpu_id,
        "fairness_notes": [
            "两组方案使用同一份 merged 模型家族与相同评测参数。",
            "AWQ 目录由 llmcompressor 导出，运行时走 vLLM 的 compressed-tensors 兼容加载路径。",
            "benchmark 与 accuracy 分开执行，减少引擎残留状态对结果的影响。",
        ],
        "best_benchmark_latency_case": (
            success_df.sort_values("Benchmark Avg Latency (s)", ascending=True).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "best_exact_match_case": (
            success_df.sort_values("Exact Match Rate", ascending=False).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "best_action_match_case": (
            success_df.sort_values("Action Match Rate", ascending=False).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = args.results_dir / "exp14_awq_vs_transformers_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    record_run_meta(
        args.results_dir,
        cli_args=vars(args),
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        data_paths=[args.benchmark_prompts_file, args.test_file, args.dataset_file],
        extra_meta={
            "entry": "experiments/18_exp14_awq_vs_transformers/run_exp14_awq_vs_transformers.py",
            "stage": "exp14_awq_vs_transformers",
            "comparison_scope": summary_payload["comparison_scope"],
            "result_csv": str(csv_path.resolve()),
            "result_markdown": str(markdown_path.resolve()),
            "result_summary": str(summary_path.resolve()),
        },
    )

    EXP7.print_info(f"Exp14 对比实验完成，CSV 输出：{csv_path}")
    EXP7.print_info(f"Markdown 报告：{markdown_path}")
    EXP7.print_info(f"Summary JSON：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
