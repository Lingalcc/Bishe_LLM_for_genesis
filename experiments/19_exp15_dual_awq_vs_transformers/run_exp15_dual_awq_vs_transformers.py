#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from copy import deepcopy
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
        "name": "LoRA4_Transformers_16bit",
        "backend": "transformers",
        "runtime_quantization": None,
        "vllm_dtype": None,
        "gpu_memory_utilization": 0.9,
        "report_quantization": "16bit",
        "artifact_key": "lora4_fp16",
        "family": "LoRA Rank4 Merged",
        "stack_label": "LoRA4 + Transformers FP16",
        "format_label": "HF Safetensors",
        "quant_note": "LoRA rank4 merged 模型，使用 Transformers 直接以 float16 加载。",
    },
    {
        "name": "LoRA4_vLLM_AWQ",
        "backend": "vllm",
        "runtime_quantization": "compressed-tensors",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.8,
        "report_quantization": "awq",
        "artifact_key": "lora4_awq",
        "family": "LoRA Rank4 Merged",
        "stack_label": "LoRA4 + vLLM AWQ",
        "format_label": "Compressed Tensors (AWQ)",
        "quant_note": "LoRA rank4 merged 的 AWQ 压缩目录，由 vLLM 按 compressed-tensors 方式加载。",
    },
    {
        "name": "Top18Rank8_Transformers_16bit",
        "backend": "transformers",
        "runtime_quantization": None,
        "vllm_dtype": None,
        "gpu_memory_utilization": 0.9,
        "report_quantization": "16bit",
        "artifact_key": "top18_rank8_fp16",
        "family": "Top18 Rank8 Merged",
        "stack_label": "Top18 Rank8 + Transformers FP16",
        "format_label": "HF Safetensors",
        "quant_note": "Top18 rank8 merged 模型，使用 Transformers 直接以 float16 加载。",
    },
    {
        "name": "Top18Rank8_vLLM_AWQ",
        "backend": "vllm",
        "runtime_quantization": "compressed-tensors",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.8,
        "report_quantization": "awq",
        "artifact_key": "top18_rank8_awq",
        "family": "Top18 Rank8 Merged",
        "stack_label": "Top18 Rank8 + vLLM AWQ",
        "format_label": "Compressed Tensors (AWQ)",
        "quant_note": "Top18 rank8 merged 的 AWQ 压缩目录，由 vLLM 按 compressed-tensors 方式加载。",
    },
]


FOUR_GB_AWQ_CASE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "LoRA4_vLLM_AWQ_4GB",
        "backend": "vllm",
        "runtime_quantization": "compressed-tensors",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.5,
        "report_quantization": "awq",
        "artifact_key": "lora4_awq",
        "family": "LoRA Rank4 Merged",
        "stack_label": "LoRA4 + vLLM AWQ (4GB)",
        "format_label": "Compressed Tensors (AWQ)",
        "quant_note": "LoRA rank4 merged 的 AWQ 压缩目录，在 4GB 目标显存模式下由 vLLM 按 compressed-tensors 方式加载。",
        "case_max_model_len": 1024,
        "comparison_group": "4GB AWQ Extension",
    },
    {
        "name": "Top18Rank8_vLLM_AWQ_4GB",
        "backend": "vllm",
        "runtime_quantization": "compressed-tensors",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.5,
        "report_quantization": "awq",
        "artifact_key": "top18_rank8_awq",
        "family": "Top18 Rank8 Merged",
        "stack_label": "Top18 Rank8 + vLLM AWQ (4GB)",
        "format_label": "Compressed Tensors (AWQ)",
        "quant_note": "Top18 rank8 merged 的 AWQ 压缩目录，在 4GB 目标显存模式下由 vLLM 按 compressed-tensors 方式加载。",
        "case_max_model_len": 1024,
        "comparison_group": "4GB AWQ Extension",
    },
]


MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "lora4_fp16": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
    "lora4_awq": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-awq",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-awq",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
    "top18_rank8_fp16": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-top18-rank8-merged",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-top18-rank8-merged",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
    "top18_rank8_awq": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-top18-rank8-merged-awq",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-top18-rank8-merged-awq",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
}


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
    parser = argparse.ArgumentParser(description="实验 19 Exp15：LoRA4 / Top18Rank8 在 Transformers 与 AWQ 部署下的四组对比。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
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
    parser.add_argument("--skip-vllm-compat-check", action="store_true", help="显式跳过当前 vLLM / compressed-tensors 版本检查。")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "")
    parser.add_argument(
        "--append-4gb-awq-cases",
        action="store_true",
        help="读取既有四组结果，只新增运行两组 4GB AWQ，并输出新的 6 组对比文件。",
    )
    parser.add_argument(
        "--existing-summary-json",
        type=Path,
        default=RESULTS_DIR / "exp15_dual_awq_vs_transformers_summary.json",
        help="扩展模式下读取的既有四组 summary 文件。",
    )
    parser.add_argument(
        "--extended-output-prefix",
        type=str,
        default="exp15_dual_awq_vs_transformers_with_4gb_awq",
        help="扩展模式输出文件名前缀，不会覆盖原始四组产物。",
    )
    return parser.parse_args(argv)


def build_case_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for cfg in CASE_CONFIGS:
        case = dict(cfg)
        case["artifact"] = dict(MODEL_ARTIFACTS[str(cfg["artifact_key"])])
        cases.append(case)
    return cases


def build_four_gb_awq_case_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for cfg in FOUR_GB_AWQ_CASE_CONFIGS:
        case = dict(cfg)
        case["artifact"] = dict(MODEL_ARTIFACTS[str(cfg["artifact_key"])])
        cases.append(case)
    return cases


def _case_palette(size: int) -> list[str]:
    base = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#e45756", "#b279a2", "#ff9da6", "#9d755d"]
    if size <= len(base):
        return base[:size]
    return [base[index % len(base)] for index in range(size)]


def _fmt_metric(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    text = str(value).strip()
    return text or "-"


def draw_figures(df: pd.DataFrame, *, output_dir: Path, output_prefix: str) -> None:
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
    colors = _case_palette(len(df))

    fig1, ax1 = plt.subplots(figsize=(16, 8))
    lat_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Avg Latency (s)"].tolist()]
    ax1.bar(labels, lat_values, color=colors)
    ax1.set_title(pick_plot_text("图1：多组方案延迟对比", "Figure 1: Latency Across Cases"))
    ax1.set_xlabel(pick_plot_text("方案", "Case"))
    ax1.set_ylabel("Seconds")
    ax1.tick_params(axis="x", rotation=15)
    fig1.tight_layout()
    fig1.savefig(output_dir / f"{output_prefix}_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 8))
    token_tps = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Token Throughput (tokens/s)"].tolist()]
    ax2.bar(labels, token_tps, color=colors)
    ax2.set_title(pick_plot_text("图2：多组方案吞吐对比", "Figure 2: Throughput Across Cases"))
    ax2.set_xlabel(pick_plot_text("方案", "Case"))
    ax2.set_ylabel("Tokens / Second")
    ax2.tick_params(axis="x", rotation=15)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"{output_prefix}_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(16, 8))
    width = 0.22
    exact_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Exact Match Rate"].tolist()]
    action_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Action Match Rate"].tolist()]
    parse_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Parse OK Rate"].tolist()]
    ax3.bar([item - width for item in x], parse_values, width=width, color="#72b7b2", label="Parse OK")
    ax3.bar(x, exact_values, width=width, color="#f58518", label="Exact Match")
    ax3.bar([item + width for item in x], action_values, width=width, color="#e45756", label="Action Match")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title(pick_plot_text("图3：多组方案准确率对比", "Figure 3: Accuracy Across Cases"))
    ax3.set_xlabel(pick_plot_text("方案", "Case"))
    ax3.set_ylabel("Rate")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_dir / f"{output_prefix}_accuracy_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(16, 8))
    benchmark_vram = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Peak VRAM (MB)"].tolist()]
    accuracy_vram = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Accuracy Max Peak VRAM (MB)"].tolist()]
    ax4.bar([item - 0.16 for item in x], benchmark_vram, width=0.32, color="#b279a2", label="Benchmark Peak VRAM")
    ax4.bar([item + 0.16 for item in x], accuracy_vram, width=0.32, color="#bab0ab", label="Accuracy Max VRAM")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15)
    ax4.set_title(pick_plot_text("图4：多组方案显存对比", "Figure 4: VRAM Across Cases"))
    ax4.set_xlabel(pick_plot_text("方案", "Case"))
    ax4.set_ylabel("MB")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(output_dir / f"{output_prefix}_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)


def write_markdown_report(df: pd.DataFrame, *, output_path: Path, report_title: str, goal_lines: list[str], matrix_lines: list[str], advice_lines: list[str]) -> None:
    EXP7.ensure_dir(output_path.parent)
    lines: list[str] = [
        report_title,
        "",
        "## 实验目标",
        "",
        *goal_lines,
        "",
        "## 对比矩阵",
        "",
        *matrix_lines,
        "",
        "## 速度结果",
        "",
        "| 方案 | 模型家族 | Backend | 载入格式 | Avg Latency (s) | Tokens/s | Peak VRAM (MB) | 状态 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {family} | {backend} | {fmt} | {avg} | {tps} | {vram} | {status} |".format(
                name=row["Name"],
                family=row["Family"],
                backend=row["Backend"],
                fmt=row["Model Format"],
                avg=_fmt_metric(row["Benchmark Avg Latency (s)"]),
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
            "| 方案 | 模型家族 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | 状态 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {family} | {parse} | {exact} | {action} | {lat} | {tps} | {status} |".format(
                name=row["Name"],
                family=row["Family"],
                parse=_fmt_metric(row["Parse OK Rate"]),
                exact=_fmt_metric(row["Exact Match Rate"]),
                action=_fmt_metric(row["Action Match Rate"]),
                lat=_fmt_metric(row["Accuracy Avg Latency (s)"]),
                tps=_fmt_metric(row["Accuracy Avg Throughput (tokens/s)"]),
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
        lines.append("- 由于这些方案同时包含“模型家族差异”和“部署栈差异”，解读时建议先做同家族内横向比较，再做跨家族比较。")

    lines.extend(
        [
            "",
            "## 解读建议",
            "",
            *advice_lines,
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _default_report_text() -> tuple[str, list[str], list[str], list[str]]:
    return (
        "# Exp15 LoRA4 / Top18Rank8 的 AWQ 与 Transformers 四组对比实验报告",
        [
            "- 同时比较两类模型家族：`LoRA rank4 merged` 与 `Top18 rank8 merged`。",
            "- 每个模型家族各跑两种部署：`Transformers 16bit` 与 `vLLM + AWQ`。",
            "- 关注速度、显存与准确率，避免覆盖既有 `EXP14` 结果。",
        ],
        [
            "- `LoRA4_Transformers_16bit`：LoRA4 merged 的未量化基线。",
            "- `LoRA4_vLLM_AWQ`：LoRA4 merged 的 AWQ 压缩部署方案。",
            "- `Top18Rank8_Transformers_16bit`：Top18 rank8 merged 的未量化基线。",
            "- `Top18Rank8_vLLM_AWQ`：Top18 rank8 merged 的 AWQ 压缩部署方案。",
        ],
        [
            "- 若关注同一模型在不同部署下的收益，可比较 `LoRA4_Transformers_16bit vs LoRA4_vLLM_AWQ`，以及 `Top18Rank8_Transformers_16bit vs Top18Rank8_vLLM_AWQ`。",
            "- 若关注不同模型家族本身谁更强，可比较两组 `Transformers 16bit`，或比较两组 `vLLM + AWQ`。",
            "- AWQ 目录均由 `llmcompressor` 导出，因此这里的 AWQ 结论应理解为“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比。",
        ],
    )


def _extended_report_text() -> tuple[str, list[str], list[str], list[str]]:
    return (
        "# Exp15 LoRA4 / Top18Rank8 的 AWQ 扩展对比实验报告（含 4GB AWQ）",
        [
            "- 保留既有四组结果：`LoRA4/Top18Rank8` 在 `Transformers 16bit` 与 `vLLM + AWQ` 下的原始对比。",
            "- 新增两组 `4GB AWQ` 扩展案例：`LoRA4_vLLM_AWQ_4GB` 与 `Top18Rank8_vLLM_AWQ_4GB`。",
            "- 扩展模式只运行新增两组，并与旧四组结果拼接输出，不覆盖之前的 CSV、JSON、Markdown 和图表。",
        ],
        [
            "- `LoRA4_Transformers_16bit`：LoRA4 merged 的未量化基线（复用既有结果）。",
            "- `LoRA4_vLLM_AWQ`：LoRA4 merged 的常规 AWQ 部署（复用既有结果）。",
            "- `Top18Rank8_Transformers_16bit`：Top18 rank8 merged 的未量化基线（复用既有结果）。",
            "- `Top18Rank8_vLLM_AWQ`：Top18 rank8 merged 的常规 AWQ 部署（复用既有结果）。",
            "- `LoRA4_vLLM_AWQ_4GB`：LoRA4 merged 的 4GB 目标显存 AWQ 扩展组。",
            "- `Top18Rank8_vLLM_AWQ_4GB`：Top18 rank8 merged 的 4GB 目标显存 AWQ 扩展组。",
        ],
        [
            "- 若关注 4GB 约束下的可运行性，可直接比较 `LoRA4_vLLM_AWQ vs LoRA4_vLLM_AWQ_4GB`，以及 `Top18Rank8_vLLM_AWQ vs Top18Rank8_vLLM_AWQ_4GB`。",
            "- 若关注整体最优方案，可在 6 组结果里同时观察延迟、吞吐、显存和动作匹配率，但要注意 4GB 组与常规组的显存目标不同。",
            "- 4GB AWQ 组默认使用更保守的运行参数，目的是在小显存约束下完成推理，不应直接等同于常规高缓存配置。",
            "- AWQ 目录均由 `llmcompressor` 导出，因此这里的 AWQ 结论应理解为“AWQ 压缩部署方案 vs Transformers 基线”的端到端对比。",
        ],
    )


def _build_case_args(args: argparse.Namespace, case_cfg: dict[str, Any]) -> argparse.Namespace:
    case_args = deepcopy(args)
    if "case_max_model_len" in case_cfg:
        case_args.max_model_len = int(case_cfg["case_max_model_len"])
    return case_args


def _run_case_rows(cases: list[dict[str, Any]], *, args: argparse.Namespace, results_dir: Path, gpu_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_cfg in cases:
        case_args = _build_case_args(args, case_cfg)
        ok, reason = EXP7.prepare_case(case_cfg, args=case_args)
        if not ok:
            EXP7.print_error(f"[{case_cfg['name']}] 预检查失败：{reason}")
            rows.append(
                {
                    "Name": case_cfg["name"],
                    "Family": case_cfg["family"],
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
        row = EXP7.run_single_case(case_cfg, args=case_args, results_dir=results_dir, gpu_id=gpu_id)
        row["Family"] = case_cfg["family"]
        if case_cfg.get("comparison_group"):
            row["Comparison Group"] = str(case_cfg["comparison_group"])
        rows.append(row)
    return rows


def _finalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    preferred = ["Name", "Family", "Comparison Group"]
    columns = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in set(preferred)]
    return df[columns]


def _load_existing_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"既有 summary 中 rows 字段格式不正确: {summary_path}")
    loaded_rows: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError(f"既有 summary 中存在非法 row: {summary_path}")
        row = dict(item)
        row.setdefault("Comparison Group", "Existing Four Cases")
        loaded_rows.append(row)
    return loaded_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.skip_vllm_compat_check:
        os.environ["LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK"] = "1"

    EXP7.RESULTS_DIR = args.results_dir
    EXP7.LOGS_DIR = LOGS_DIR
    EXP7.TEMP_DIR = TEMP_DIR

    EXP7.ensure_dir(args.results_dir)
    EXP7.ensure_dir(LOGS_DIR)
    EXP7.ensure_dir(TEMP_DIR)

    gpu_id = EXP7.infer_gpu_id(args.gpu_id)
    EXP7.print_info(
        "Exp15 四组对比实验开始："
        f" GPU={gpu_id}, benchmark_samples={args.benchmark_num_samples}, accuracy_samples={args.accuracy_num_samples}"
    )

    if args.append_4gb_awq_cases:
        if not args.existing_summary_json.exists():
            raise FileNotFoundError(f"扩展模式需要先存在四组 summary：{args.existing_summary_json}")
        EXP7.print_info(f"Exp15 扩展模式：读取既有四组结果 {args.existing_summary_json}")
        base_rows = _load_existing_rows(args.existing_summary_json)
        new_rows = _run_case_rows(build_four_gb_awq_case_matrix(), args=args, results_dir=args.results_dir, gpu_id=gpu_id)
        rows = base_rows + new_rows
        output_prefix = args.extended_output_prefix
        comparison_scope = "Existing four cases + 4GB AWQ extension for LoRA4 / Top18 rank8"
        fairness_notes = [
            "既有四组结果直接读取已有 summary，不重新跑，避免影响原始产物。",
            "新增仅运行两组 4GB AWQ 扩展 case，并与原始四组按统一字段拼接。",
            "4GB AWQ 扩展组默认使用更保守的显存参数：gpu_memory_utilization=0.5，max_model_len=1024。",
            "扩展结果写入新的输出前缀，不覆盖原始四组的 CSV、JSON、Markdown 和图表。",
        ]
        report_title, goal_lines, matrix_lines, advice_lines = _extended_report_text()
    else:
        rows = _run_case_rows(build_case_matrix(), args=args, results_dir=args.results_dir, gpu_id=gpu_id)
        output_prefix = "exp15_dual_awq_vs_transformers"
        comparison_scope = "LoRA4 merged / Top18 rank8 merged under Transformers 16bit vs vLLM AWQ"
        fairness_notes = [
            "四组方案使用同一份 benchmark prompts、相同 batch size、相同 max_new_tokens 与相同 max_model_len。",
            "LoRA4 与 Top18 Rank8 各自内部都做 Transformers 16bit 与 vLLM AWQ 的成对对比。",
            "两组 AWQ 目录均由 llmcompressor 导出，运行时走 vLLM 的 compressed-tensors 兼容加载路径。",
            "benchmark 与 accuracy 分开执行，减少引擎残留状态对结果的影响。",
        ]
        report_title, goal_lines, matrix_lines, advice_lines = _default_report_text()

    df = _finalize_rows(rows)

    csv_path = args.results_dir / f"{output_prefix}_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    draw_figures(df, output_dir=args.results_dir, output_prefix=output_prefix)

    markdown_path = args.results_dir / f"{output_prefix}_report.md"
    write_markdown_report(
        df,
        output_path=markdown_path,
        report_title=report_title,
        goal_lines=goal_lines,
        matrix_lines=matrix_lines,
        advice_lines=advice_lines,
    )

    success_df = df[df["Overall Status"] == "success"].copy()
    summary_payload = {
        "experiment": output_prefix,
        "comparison_scope": comparison_scope,
        "benchmark_num_samples": int(args.benchmark_num_samples),
        "accuracy_num_samples": int(args.accuracy_num_samples),
        "gpu_id": gpu_id,
        "skip_vllm_compat_check": bool(args.skip_vllm_compat_check),
        "append_4gb_awq_cases": bool(args.append_4gb_awq_cases),
        "existing_summary_json": str(args.existing_summary_json.resolve()) if args.append_4gb_awq_cases else "",
        "fairness_notes": fairness_notes,
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
    summary_path = args.results_dir / f"{output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    record_run_meta(
        args.results_dir,
        cli_args=vars(args),
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        data_paths=[args.benchmark_prompts_file, args.test_file, args.dataset_file],
        extra_meta={
            "entry": "experiments/19_exp15_dual_awq_vs_transformers/run_exp15_dual_awq_vs_transformers.py",
            "stage": output_prefix,
            "comparison_scope": summary_payload["comparison_scope"],
            "result_csv": str(csv_path.resolve()),
            "result_markdown": str(markdown_path.resolve()),
            "result_summary": str(summary_path.resolve()),
        },
    )

    EXP7.print_info(f"Exp15 对比实验完成，CSV 输出：{csv_path}")
    EXP7.print_info(f"Markdown 报告：{markdown_path}")
    EXP7.print_info(f"Summary JSON：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
