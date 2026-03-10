#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluate_dataset_with_performance import evaluate_dataset
from src.eval.inference_engines import build_inference_engine


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 `pyyaml`，请先安装。") from exc

    if not path.exists():
        raise FileNotFoundError(f"实验配置文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML 根节点必须是映射对象(dict)。")
    return data


def _load_dataset_rows(dataset_file: Path) -> list[dict[str, Any]]:
    if not dataset_file.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_file}")

    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("数据集 JSON 顶层必须是 list。")

    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        instruction = row.get("instruction")
        output = row.get("output")
        if isinstance(instruction, str) and instruction.strip() and isinstance(output, str) and output.strip():
            valid_rows.append(row)

    if not valid_rows:
        raise RuntimeError("数据集中没有有效样本（需要 instruction/output 均为非空字符串）。")
    return valid_rows


def _sample_rows(rows: list[dict[str, Any]], num_samples: int, seed: int) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples 必须 > 0")
    if num_samples >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, k=num_samples)


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "experiment"


def _resolve_path(path_str: str, *, default_base: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return default_base / path


def _build_result_row(model_setup: str, report: dict[str, Any]) -> dict[str, Any]:
    avg_peak_vram_mb = float(report.get("avg_peak_vram_mb", 0.0))
    avg_throughput_tps = float(report.get("avg_throughput_tps", 0.0))
    avg_latency_sec = float(report.get("avg_latency_sec", 0.0))
    accuracy = float(report.get("exact_match_rate", 0.0))
    return {
        "Model_Setup": model_setup,
        "VRAM_Peak_GB": avg_peak_vram_mb / 1024.0,
        "Throughput_tps": avg_throughput_tps,
        "Latency_ms": avg_latency_sec * 1000.0,
        "Accuracy": accuracy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 experiments_matrix.yaml 自动运行多组推理实验并导出指标 CSV。"
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments_matrix.yaml",
        help="实验矩阵配置文件路径。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matrix_cfg = _load_yaml(args.matrix_config)

    common_settings = matrix_cfg.get("common_settings", {})
    if not isinstance(common_settings, dict):
        raise ValueError("common_settings 必须是 dict。")

    experiments = matrix_cfg.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments 必须是非空列表。")

    dataset_file_raw = common_settings.get("dataset_file")
    if not isinstance(dataset_file_raw, str) or not dataset_file_raw.strip():
        raise ValueError("common_settings.dataset_file 必填。")
    dataset_file = _resolve_path(dataset_file_raw, default_base=REPO_ROOT)

    num_samples = int(common_settings.get("num_samples", 200))
    seed = int(common_settings.get("seed", 42))

    output_csv_raw = common_settings.get("output_csv", "results/experiment_metrics.csv")
    if not isinstance(output_csv_raw, str) or not output_csv_raw.strip():
        raise ValueError("common_settings.output_csv 必须是非空字符串。")
    output_csv = _resolve_path(output_csv_raw, default_base=REPO_ROOT)

    all_rows = _load_dataset_rows(dataset_file)
    sampled_rows = _sample_rows(all_rows, num_samples=num_samples, seed=seed)

    reports_dir = output_csv.parent / "experiment_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []

    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("experiments 中的每个条目都必须是 dict。")

        exp_name = str(exp.get("name", "")).strip()
        if not exp_name:
            raise ValueError("每个实验都必须提供 name 字段。")

        exp_cfg = dict(exp)
        if "model_path" not in exp_cfg:
            raise ValueError(f"实验 {exp_name} 缺少 model_path。")
        exp_cfg["model_path"] = str(_resolve_path(str(exp_cfg["model_path"]), default_base=REPO_ROOT))

        print(f"\n[experiment] {exp_name}")
        print(f"  backend    : {exp_cfg.get('backend')}")
        print(f"  model_path : {exp_cfg.get('model_path')}")

        try:
            # 工厂模式：由配置驱动创建推理引擎
            engine = build_inference_engine(exp_cfg)

            # evaluate_dataset 内部已使用 PerformanceMonitor/ResourceMonitor 记录性能指标
            report_path = reports_dir / f"{_sanitize_name(exp_name)}_report.json"
            report = evaluate_dataset(sampled_rows, engine, report_file=report_path)
            result_rows.append(_build_result_row(exp_name, report))
            print(f"  accuracy   : {report.get('exact_match_rate', 0.0):.4f}")
            print(f"  report     : {report_path}")
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}")
            result_rows.append(
                {
                    "Model_Setup": exp_name,
                    "VRAM_Peak_GB": float("nan"),
                    "Throughput_tps": float("nan"),
                    "Latency_ms": float("nan"),
                    "Accuracy": float("nan"),
                }
            )

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 `pandas`，请先安装。") from exc

    df = pd.DataFrame(
        result_rows,
        columns=["Model_Setup", "VRAM_Peak_GB", "Throughput_tps", "Latency_ms", "Accuracy"],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n[done] 实验汇总：")
    print(df.to_string(index=False))
    print(f"\n[done] CSV 已导出: {output_csv}")


if __name__ == "__main__":
    main()
