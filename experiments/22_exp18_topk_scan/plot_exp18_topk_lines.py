#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXPERIMENT_DIR / "reports"
TEMP_DIR = EXPERIMENT_DIR / ".cache"
DEFAULT_SUMMARY_PATH = REPORTS_DIR / "exp18_topk_summary.json"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "exp18_topk_linecharts.png"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((TEMP_DIR / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("缺少 matplotlib，请先安装：pip install matplotlib") from exc

from src.utils.plotting import configure_report_matplotlib, pick_plot_text


configure_report_matplotlib(matplotlib)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Exp18 Top-K 扫描结果绘制成折线图。")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Exp18 汇总结果 JSON 路径。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="导出的折线图 PNG 路径。",
    )
    parser.add_argument(
        "--hide-baseline",
        action="store_true",
        help="不绘制 full_rank4 baseline 参考线。",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def normalize_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("cases")
    if not isinstance(rows, list):
        raise ValueError("summary 缺少 cases 数组。")

    normalized: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if "k" not in item:
            continue
        try:
            k_value = int(item["k"])
        except Exception:
            continue
        item_copy = dict(item)
        item_copy["k"] = k_value
        normalized.append(item_copy)

    normalized.sort(key=lambda row: row["k"])
    if not normalized:
        raise ValueError("summary 中没有可用于画图的 case。")
    return normalized


def has_metric(row: dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    return isinstance(value, (int, float))


def build_metric_specs(force_ascii: bool) -> list[dict[str, Any]]:
    return [
        {
            "key": "parse_ok_rate",
            "title": pick_plot_text("Parse 成功率", "Parse OK Rate", force_ascii=force_ascii),
            "ylabel": pick_plot_text("比例", "Rate", force_ascii=force_ascii),
            "ylim": (0.0, 1.0),
            "color": "#1f77b4",
        },
        {
            "key": "exact_match_rate",
            "title": pick_plot_text("Exact Match", "Exact Match", force_ascii=force_ascii),
            "ylabel": pick_plot_text("比例", "Rate", force_ascii=force_ascii),
            "ylim": (0.0, 1.0),
            "color": "#d62728",
        },
        {
            "key": "action_match_rate",
            "title": pick_plot_text("Action Match", "Action Match", force_ascii=force_ascii),
            "ylabel": pick_plot_text("比例", "Rate", force_ascii=force_ascii),
            "ylim": (0.0, 1.0),
            "color": "#2ca02c",
        },
        {
            "key": "avg_latency_sec",
            "title": pick_plot_text("平均时延", "Average Latency", force_ascii=force_ascii),
            "ylabel": pick_plot_text("秒", "Seconds", force_ascii=force_ascii),
            "ylim": None,
            "color": "#ff7f0e",
        },
        {
            "key": "avg_throughput_tps",
            "title": pick_plot_text("平均吞吐", "Average Throughput", force_ascii=force_ascii),
            "ylabel": pick_plot_text("tokens/s", "tokens/s", force_ascii=force_ascii),
            "ylim": None,
            "color": "#9467bd",
        },
        {
            "key": "avg_peak_vram_mb",
            "title": pick_plot_text("平均峰值显存", "Average Peak VRAM", force_ascii=force_ascii),
            "ylabel": "MB",
            "ylim": None,
            "color": "#8c564b",
        },
    ]


def available_metric_specs(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    force_ascii: bool,
) -> list[dict[str, Any]]:
    specs = build_metric_specs(force_ascii)
    return [
        spec
        for spec in specs
        if all(has_metric(row, spec["key"]) for row in rows) or has_metric(baseline, spec["key"])
    ]


def plot_summary(
    summary: dict[str, Any],
    output_path: Path,
    *,
    show_baseline: bool,
) -> Path:
    rows = normalize_rows(summary)
    baseline = summary.get("baseline", {})
    if not isinstance(baseline, dict):
        baseline = {}

    force_ascii = not configure_report_matplotlib(matplotlib)
    specs = available_metric_specs(rows, baseline, force_ascii=force_ascii)
    if not specs:
        raise ValueError("没有可绘制的数值指标。")

    k_values = [row["k"] for row in rows]
    cols = 3
    total = len(specs)
    rows_count = math.ceil(total / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(5.6 * cols, 3.8 * rows_count))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index, spec in enumerate(specs):
        ax = axes_list[index]
        metric = spec["key"]
        values = [float(row[metric]) for row in rows]
        ax.plot(
            k_values,
            values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=spec["color"],
            label=pick_plot_text("Top-K rank8", "Top-K rank8", force_ascii=force_ascii),
        )

        best_idx = max(range(len(values)), key=lambda i: values[i]) if "throughput" in metric or "rate" in metric else min(range(len(values)), key=lambda i: values[i])
        ax.scatter(
            [k_values[best_idx]],
            [values[best_idx]],
            s=70,
            color=spec["color"],
            edgecolors="white",
            linewidth=1.1,
            zorder=3,
        )

        if show_baseline and has_metric(baseline, metric):
            baseline_value = float(baseline[metric])
            ax.axhline(
                baseline_value,
                linestyle="--",
                linewidth=1.5,
                color="#444444",
                alpha=0.9,
                label=pick_plot_text("full_rank4 baseline", "full_rank4 baseline", force_ascii=force_ascii),
            )

        ax.set_title(spec["title"], fontsize=11)
        ax.set_xlabel("K")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xticks(k_values)
        ax.grid(True, linestyle="--", alpha=0.25)
        if spec["ylim"] is not None:
            ax.set_ylim(*spec["ylim"])
        ax.legend(fontsize=8, frameon=False, loc="best")

    for index in range(total, len(axes_list)):
        fig.delaxes(axes_list[index])

    title = pick_plot_text("Exp18 Top-K 扫描折线图", "Exp18 Top-K Line Charts", force_ascii=force_ascii)
    question = summary.get("question")
    question_text = pick_plot_text(
        "按重要性排名选前 K 层做 rank8 微调时，K 如何影响任务性能与推理成本。",
        "How does K affect task quality and inference cost when rank8 is applied to the top-K layers by importance?",
        force_ascii=force_ascii,
    )
    fig.suptitle(title, fontsize=15, y=1.02)
    if isinstance(question, str) and question.strip():
        fig.text(0.5, 0.99, question_text, ha="center", va="top", fontsize=10, color="#555555")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path = args.summary_path.resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary 文件不存在: {summary_path}")

    summary = load_json(summary_path)
    output_path = args.output_path.resolve()
    plot_summary(summary, output_path, show_baseline=not args.hide_baseline)
    print(f"[exp18_plot] 已生成折线图: {output_path}")


if __name__ == "__main__":
    main()
