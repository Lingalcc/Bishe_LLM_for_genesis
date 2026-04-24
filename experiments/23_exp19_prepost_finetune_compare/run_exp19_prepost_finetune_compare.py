#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import configure_report_matplotlib, pick_plot_text


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TEMP_DIR = EXPERIMENT_DIR / ".cache"

BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
EVAL_RUNNER = REPO_ROOT / "experiments" / "03_eval_exp" / "run_accuracy.py"

SUMMARY_JSON_PATH = REPORTS_DIR / "exp19_prepost_finetune_summary.json"
SUMMARY_CSV_PATH = REPORTS_DIR / "exp19_prepost_finetune_summary.csv"
SUMMARY_MD_PATH = REPORTS_DIR / "exp19_prepost_finetune_summary.md"
ACCURACY_FIGURE_PATH = REPORTS_DIR / "exp19_accuracy_comparison.png"
PERFORMANCE_FIGURE_PATH = REPORTS_DIR / "exp19_performance_comparison.png"


@dataclass(frozen=True)
class ModelCase:
    name: str
    display_name: str
    stage: str
    family: str
    model_path: str
    report_path: Path
    notes: str = ""


DEFAULT_CASES: tuple[ModelCase, ...] = (
    ModelCase(
        name="pretrain_base",
        display_name="Pretrain Base",
        stage="pre_finetune",
        family="base",
        model_path="model/Qwen_Qwen2.5-3B-Instruct",
        report_path=REPORTS_DIR / "accuracy_report_pretrain_base.json",
        notes="未经过机器人指令数据微调的 Qwen2.5-3B-Instruct 基座模型。",
    ),
    ModelCase(
        name="lora_full_rank4",
        display_name="LoRA Full r4",
        stage="post_finetune",
        family="lora_rank",
        model_path="model/qwen2.5-3b-genesis-lora-rank-4",
        report_path=REPO_ROOT / "experiments/22_exp18_topk_scan/reports/accuracy_report_full_rank4.json",
        notes="全层 LoRA rank=4 基线。",
    ),
    ModelCase(
        name="lora_rank8",
        display_name="LoRA r8",
        stage="post_finetune",
        family="lora_rank",
        model_path="output/qwen2.5-3b-genesis-lora-rank-8",
        report_path=REPO_ROOT / "experiments/06_exp2_lora_rank/reports/accuracy_report_rank_8.json",
    ),
    ModelCase(
        name="lora_rank16",
        display_name="LoRA r16",
        stage="post_finetune",
        family="lora_rank",
        model_path="output/qwen2.5-3b-genesis-lora-rank-16",
        report_path=REPO_ROOT / "experiments/06_exp2_lora_rank/reports/accuracy_report_rank_16.json",
    ),
    ModelCase(
        name="lora_rank32",
        display_name="LoRA r32",
        stage="post_finetune",
        family="lora_rank",
        model_path="output/qwen2.5-3b-genesis-lora-rank-32",
        report_path=REPO_ROOT / "experiments/06_exp2_lora_rank/reports/accuracy_report_rank_32.json",
    ),
    ModelCase(
        name="lora_rank64",
        display_name="LoRA r64",
        stage="post_finetune",
        family="lora_rank",
        model_path="output/qwen2.5-3b-genesis-lora-rank-64",
        report_path=REPO_ROOT / "experiments/06_exp2_lora_rank/reports/accuracy_report_rank_64.json",
    ),
    ModelCase(
        name="top18_rank8",
        display_name="Top18 r8",
        stage="post_finetune",
        family="layer_select",
        model_path="output/exp18_topk_scan/top18_rank8",
        report_path=REPO_ROOT / "experiments/22_exp18_topk_scan/reports/accuracy_report_top18_rank8.json",
        notes="按重要性打分选择 18 层，rank=8。",
    ),
    ModelCase(
        name="top24_rank8",
        display_name="Top24 r8",
        stage="post_finetune",
        family="layer_select",
        model_path="output/exp18_topk_scan/top24_rank8",
        report_path=REPO_ROOT / "experiments/22_exp18_topk_scan/reports/accuracy_report_top24_rank8.json",
        notes="按重要性打分选择 24 层，rank=8。",
    ),
    ModelCase(
        name="top28_rank8",
        display_name="Top28 r8",
        stage="post_finetune",
        family="layer_select",
        model_path="output/exp18_topk_scan/top28_rank8",
        report_path=REPO_ROOT / "experiments/22_exp18_topk_scan/reports/accuracy_report_top28_rank8.json",
        notes="按重要性打分选择 28 层，rank=8。",
    ),
    ModelCase(
        name="qlora",
        display_name="QLoRA",
        stage="post_finetune",
        family="method",
        model_path="output/qwen2.5-3b-genesis-qlora",
        report_path=REPO_ROOT / "experiments/07_exp3_methods/reports/accuracy_report_qlora.json",
    ),
    ModelCase(
        name="dora",
        display_name="DoRA",
        stage="post_finetune",
        family="method",
        model_path="output/qwen2.5-3b-genesis-dora",
        report_path=REPO_ROOT / "experiments/07_exp3_methods/reports/accuracy_report_dora.json",
    ),
    ModelCase(
        name="galore",
        display_name="GaLore",
        stage="post_finetune",
        family="method",
        model_path="output/qwen2.5-3b-genesis-galore",
        report_path=REPO_ROOT / "experiments/07_exp3_methods/reports/accuracy_report_galore.json",
    ),
)


METRIC_KEYS = (
    "parse_ok_rate",
    "exact_match_rate",
    "action_match_rate",
    "avg_latency_sec",
    "avg_throughput_tps",
    "avg_peak_vram_mb",
    "max_peak_vram_mb",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp19：比较微调前基座模型与多个微调后模型的准确率和推理性能。"
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG_PATH, help="全局基础配置。")
    parser.add_argument(
        "--eval-missing-base",
        action="store_true",
        help="如果未微调基座模型 report 缺失，则先调用 accuracy runner 生成。",
    )
    parser.add_argument(
        "--base-num-samples",
        type=int,
        default=200,
        help="补跑未微调基座模型时使用的样本数，建议与其他报告保持一致。",
    )
    parser.add_argument("--seed", type=int, default=42, help="补跑未微调基座模型时使用的随机种子。")
    parser.add_argument("--backend", default="transformers", help="补跑未微调基座模型时使用的推理后端。")
    parser.add_argument(
        "--base-quantization",
        default=None,
        help="补跑未微调基座模型时使用的量化方式，例如 4bit/8bit；默认不量化。",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="在 CSV/Markdown 中保留缺失 report 的占位行。",
    )
    parser.add_argument("--no-plots", action="store_true", help="只生成表格报告，不生成 PNG 图。")
    return parser.parse_args(argv)


def ensure_dirs() -> None:
    for path in (REPORTS_DIR, LOGS_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _metric(report: dict[str, Any], key: str) -> float | None:
    value = report.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def build_eval_override(
    *,
    model_path: str,
    report_path: Path,
    num_samples: int,
    seed: int,
    backend: str,
    quantization: str | None,
) -> Path:
    override_path = TEMP_DIR / "eval_pretrain_base_override.yaml"
    payload = {
        "test": {
            "accuracy_eval": {
                "mode": "local",
                "report_file": repo_rel(report_path),
                "model_path": model_path,
                "backend": backend,
                "quantization": quantization,
                "num_samples": int(num_samples),
                "seed": int(seed),
                "temperature": 0.0,
                "trust_remote_code": True,
            }
        }
    }
    write_yaml(override_path, payload)
    return override_path


def run_stage(stage_name: str, command: list[str], log_path: Path) -> None:
    command_str = " ".join(command)
    print(f"[{stage_name}] command: {command_str}", flush=True)
    print(f"[{stage_name}] log    : {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {stage_name} =====\n")
        f.write(f"cwd: {REPO_ROOT}\n")
        f.write(f"command: {command_str}\n\n")
        f.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"阶段 {stage_name} 执行失败，退出码 {return_code}。日志: {log_path}")


def maybe_eval_pretrain_base(args: argparse.Namespace, case: ModelCase) -> None:
    if case.report_path.exists():
        return
    if not args.eval_missing_base:
        return
    override_path = build_eval_override(
        model_path=case.model_path,
        report_path=case.report_path,
        num_samples=args.base_num_samples,
        seed=args.seed,
        backend=args.backend,
        quantization=args.base_quantization,
    )
    run_stage(
        "eval_pretrain_base",
        [
            sys.executable,
            str(EVAL_RUNNER),
            "--base-config",
            str(args.base_config),
            "--config",
            str(override_path),
        ],
        LOGS_DIR / "eval_pretrain_base.log",
    )


def collect_rows(cases: tuple[ModelCase, ...], *, include_missing: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    baseline_metrics: dict[str, float | None] = {}

    baseline_case = next((case for case in cases if case.stage == "pre_finetune"), None)
    if baseline_case is not None and baseline_case.report_path.exists():
        baseline_report = load_json(baseline_case.report_path)
        baseline_metrics = {key: _metric(baseline_report, key) for key in METRIC_KEYS}

    for case in cases:
        report_exists = case.report_path.exists()
        row: dict[str, Any] = {
            "name": case.name,
            "display_name": case.display_name,
            "stage": case.stage,
            "family": case.family,
            "model_path": case.model_path,
            "report_path": repo_rel(case.report_path),
            "report_exists": report_exists,
            "notes": case.notes,
        }
        if not report_exists:
            missing.append(row)
            if include_missing:
                rows.append(row)
            continue

        report = load_json(case.report_path)
        row.update(
            {
                "dataset_file": report.get("dataset_file"),
                "seed": report.get("seed"),
                "num_samples_evaluated": report.get("num_samples_evaluated"),
                "backend": report.get("backend"),
                "quantization": report.get("quantization"),
            }
        )
        for key in METRIC_KEYS:
            row[key] = _metric(report, key)

        if baseline_metrics:
            for key in METRIC_KEYS:
                row[f"delta_vs_pretrain_{key}"] = _delta(row.get(key), baseline_metrics.get(key))
        rows.append(row)

    return rows, missing


def _best_row(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool = False) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row.get(key), (int, float))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key])) if lower_is_better else max(candidates, key=lambda row: float(row[key]))


def build_summary(rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    comparable_rows = [row for row in rows if row.get("report_exists")]
    post_rows = [row for row in comparable_rows if row.get("stage") == "post_finetune"]
    baseline = next((row for row in comparable_rows if row.get("stage") == "pre_finetune"), None)
    return {
        "experiment": "exp19_prepost_finetune_compare",
        "question": "微调前基座模型与多个微调后模型在结构化动作生成准确率和推理性能上的差异。",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline,
        "cases": rows,
        "missing_reports": missing,
        "best_exact_match": _best_row(post_rows, "exact_match_rate"),
        "best_action_match": _best_row(post_rows, "action_match_rate"),
        "best_parse_ok": _best_row(post_rows, "parse_ok_rate"),
        "best_latency": _best_row(post_rows, "avg_latency_sec", lower_is_better=True),
        "best_throughput": _best_row(post_rows, "avg_throughput_tps"),
        "artifacts": {
            "summary_json": repo_rel(SUMMARY_JSON_PATH),
            "summary_csv": repo_rel(SUMMARY_CSV_PATH),
            "summary_markdown": repo_rel(SUMMARY_MD_PATH),
            "accuracy_figure": repo_rel(ACCURACY_FIGURE_PATH),
            "performance_figure": repo_rel(PERFORMANCE_FIGURE_PATH),
        },
    }


def write_summary_csv(summary: dict[str, Any], output_path: Path) -> None:
    rows = summary.get("cases", [])
    fieldnames = [
        "name",
        "display_name",
        "stage",
        "family",
        "report_exists",
        "num_samples_evaluated",
        "parse_ok_rate",
        "exact_match_rate",
        "action_match_rate",
        "avg_latency_sec",
        "avg_throughput_tps",
        "avg_peak_vram_mb",
        "max_peak_vram_mb",
        "delta_vs_pretrain_parse_ok_rate",
        "delta_vs_pretrain_exact_match_rate",
        "delta_vs_pretrain_action_match_rate",
        "delta_vs_pretrain_avg_latency_sec",
        "delta_vs_pretrain_avg_throughput_tps",
        "model_path",
        "report_path",
        "notes",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _fmt_rate(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "-"


def _fmt_num(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "-"


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    baseline = summary.get("baseline")
    rows = [row for row in summary.get("cases", []) if row.get("report_exists")]
    missing = summary.get("missing_reports", [])
    lines = [
        "# Exp19 微调前后模型对比实验",
        "",
        f"- 实验问题：{summary.get('question', '')}",
        "- 评价指标：`parse_ok_rate`、`exact_match_rate`、`action_match_rate`、`avg_latency_sec`、`avg_throughput_tps`、显存占用。",
        "- 所有已有报告均来自 `experiments/*/reports/*accuracy*.json`，未重新训练模型。",
        "",
    ]

    if isinstance(baseline, dict):
        lines.extend(
            [
                "## 微调前基座模型",
                "",
                f"- 模型：`{baseline.get('model_path')}`",
                f"- Exact Match：`{_fmt_rate(baseline.get('exact_match_rate'))}`",
                f"- Action Match：`{_fmt_rate(baseline.get('action_match_rate'))}`",
                f"- 平均延迟：`{_fmt_num(baseline.get('avg_latency_sec'))} s`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 微调前基座模型",
                "",
                "- 当前尚未生成未微调基座模型的 accuracy report。可运行：",
                "",
                "```bash",
                "python experiments/23_exp19_prepost_finetune_compare/run_exp19_prepost_finetune_compare.py --eval-missing-base",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 对比结果",
            "",
            "| Model | Stage | Family | Samples | Parse OK | Exact Match | Action Match | Latency(s) | Throughput(tokens/s) | VRAM(MB) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('display_name')} | "
            f"{row.get('stage')} | "
            f"{row.get('family')} | "
            f"{row.get('num_samples_evaluated', '-')} | "
            f"{_fmt_rate(row.get('parse_ok_rate'))} | "
            f"{_fmt_rate(row.get('exact_match_rate'))} | "
            f"{_fmt_rate(row.get('action_match_rate'))} | "
            f"{_fmt_num(row.get('avg_latency_sec'))} | "
            f"{_fmt_num(row.get('avg_throughput_tps'))} | "
            f"{_fmt_num(row.get('avg_peak_vram_mb'))} |"
        )

    if missing:
        lines.extend(["", "## 缺失报告", ""])
        for item in missing:
            lines.append(f"- `{item.get('display_name')}`: `{item.get('report_path')}`")

    for title, key, metric in (
        ("最佳 Exact Match", "best_exact_match", "exact_match_rate"),
        ("最佳 Action Match", "best_action_match", "action_match_rate"),
        ("最低平均延迟", "best_latency", "avg_latency_sec"),
        ("最高吞吐", "best_throughput", "avg_throughput_tps"),
    ):
        item = summary.get(key)
        if isinstance(item, dict):
            lines.extend(["", f"## {title}", "", f"- 模型：`{item.get('display_name')}`", f"- 指标值：`{_fmt_num(item.get(metric))}`"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(summary: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[exp19] skip plots: matplotlib unavailable ({exc})", flush=True)
        return

    configure_report_matplotlib(matplotlib)
    rows = [
        row
        for row in summary.get("cases", [])
        if row.get("report_exists") and isinstance(row.get("exact_match_rate"), (int, float))
    ]
    if not rows:
        return

    names = [str(row.get("display_name")) for row in rows]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.9), 5.2))
    width = 0.25
    ax.bar(x - width, [float(row.get("parse_ok_rate", 0.0)) for row in rows], width, label="Parse OK")
    ax.bar(x, [float(row.get("exact_match_rate", 0.0)) for row in rows], width, label="Exact Match")
    ax.bar(x + width, [float(row.get("action_match_rate", 0.0)) for row in rows], width, label="Action Match")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(pick_plot_text("比例", "Rate"))
    ax.set_title(pick_plot_text("微调前后准确率对比", "Pre/Post Fine-tuning Accuracy"))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(ACCURACY_FIGURE_PATH, dpi=180)
    plt.close(fig)

    perf_rows = [
        row
        for row in rows
        if isinstance(row.get("avg_latency_sec"), (int, float)) and isinstance(row.get("avg_throughput_tps"), (int, float))
    ]
    if not perf_rows:
        return
    perf_names = [str(row.get("display_name")) for row in perf_rows]
    px = np.arange(len(perf_names))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(perf_rows) * 0.95), 4.8))
    axes[0].bar(px, [float(row.get("avg_latency_sec", 0.0)) for row in perf_rows], color="#4c78a8")
    axes[0].set_title(pick_plot_text("平均延迟", "Avg Latency"))
    axes[0].set_ylabel("s")
    axes[0].set_xticks(px)
    axes[0].set_xticklabels(perf_names, rotation=35, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].bar(px, [float(row.get("avg_throughput_tps", 0.0)) for row in perf_rows], color="#59a14f")
    axes[1].set_title(pick_plot_text("平均吞吐", "Avg Throughput"))
    axes[1].set_ylabel("tokens/s")
    axes[1].set_xticks(px)
    axes[1].set_xticklabels(perf_names, rotation=35, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle(pick_plot_text("微调前后推理性能对比", "Pre/Post Fine-tuning Performance"))
    fig.tight_layout()
    fig.savefig(PERFORMANCE_FIGURE_PATH, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dirs()

    pretrain_case = next(case for case in DEFAULT_CASES if case.stage == "pre_finetune")
    maybe_eval_pretrain_base(args, pretrain_case)

    rows, missing = collect_rows(DEFAULT_CASES, include_missing=args.include_missing)
    summary = build_summary(rows, missing)
    write_json(SUMMARY_JSON_PATH, summary)
    write_summary_csv(summary, SUMMARY_CSV_PATH)
    write_summary_markdown(summary, SUMMARY_MD_PATH)
    if not args.no_plots:
        write_plots(summary)

    print(f"[exp19] summary json : {SUMMARY_JSON_PATH}", flush=True)
    print(f"[exp19] summary csv  : {SUMMARY_CSV_PATH}", flush=True)
    print(f"[exp19] summary md   : {SUMMARY_MD_PATH}", flush=True)
    if not args.no_plots:
        print(f"[exp19] accuracy fig : {ACCURACY_FIGURE_PATH}", flush=True)
        print(f"[exp19] perf fig     : {PERFORMANCE_FIGURE_PATH}", flush=True)


if __name__ == "__main__":
    main()
