#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.accuracy import run_accuracy_from_merged_config
from src.utils.config import load_merged_config
from src.utils.run_meta import record_run_meta
from src.utils.secrets import safe_json_dumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最终方案统一总对比实验。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/16_exp12_final_compare/configs/compare.yaml"),
        help="实验配置文件路径。",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="全局基础配置路径。",
    )
    return parser.parse_args()


def _get_compare_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("final_compare", {})
    if not isinstance(section, dict):
        raise TypeError("final_compare 配置必须是对象。")
    return section


def _ensure_accuracy_section(config: dict[str, Any]) -> dict[str, Any]:
    test_cfg = config.setdefault("test", {})
    if not isinstance(test_cfg, dict):
        raise TypeError("test 配置必须是对象。")
    acc_cfg = test_cfg.setdefault("accuracy_eval", {})
    if not isinstance(acc_cfg, dict):
        raise TypeError("test.accuracy_eval 配置必须是对象。")
    return acc_cfg


def _read_prompt_text(prompt_file: str | None) -> str | None:
    if not prompt_file:
        return None
    path = Path(prompt_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.read_text(encoding="utf-8").strip()


def _build_case_config(base_config: dict[str, Any], case_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_config)
    acc_cfg = _ensure_accuracy_section(merged)
    acc_cfg["mode"] = "local"
    acc_cfg["model_path"] = str(case_cfg.get("model_path", "")).strip()
    acc_cfg["backend"] = str(case_cfg.get("backend", acc_cfg.get("backend", "transformers")))
    acc_cfg["quantization"] = case_cfg.get("quantization", acc_cfg.get("quantization"))
    acc_cfg["report_file"] = str(case_cfg.get("report_file", acc_cfg.get("report_file", "")))

    if case_cfg.get("max_new_tokens") is not None:
        acc_cfg["max_new_tokens"] = int(case_cfg["max_new_tokens"])
    if case_cfg.get("max_model_len") is not None:
        acc_cfg["max_model_len"] = int(case_cfg["max_model_len"])
    if case_cfg.get("gpu_memory_utilization") is not None:
        acc_cfg["gpu_memory_utilization"] = float(case_cfg["gpu_memory_utilization"])
    if case_cfg.get("temperature") is not None:
        acc_cfg["temperature"] = float(case_cfg["temperature"])
    if case_cfg.get("trust_remote_code") is not None:
        acc_cfg["trust_remote_code"] = bool(case_cfg["trust_remote_code"])

    system_prompt_text = _read_prompt_text(case_cfg.get("system_prompt_file"))
    if system_prompt_text is not None:
        acc_cfg["system_prompt"] = system_prompt_text
    elif case_cfg.get("system_prompt") is not None:
        acc_cfg["system_prompt"] = str(case_cfg["system_prompt"])

    return merged


def _summarize_case(case_cfg: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(case_cfg.get("name", "")),
        "label": str(case_cfg.get("label", case_cfg.get("name", ""))),
        "model_path": str(case_cfg.get("model_path", "")),
        "backend": str(report.get("backend", case_cfg.get("backend", ""))),
        "quantization": report.get("quantization"),
        "num_samples_evaluated": int(report.get("num_samples_evaluated", 0)),
        "parse_ok_rate": float(report.get("parse_ok_rate", 0.0)),
        "exact_match_rate": float(report.get("exact_match_rate", 0.0)),
        "action_match_rate": float(report.get("action_match_rate", 0.0)),
        "avg_latency_sec": float(report.get("avg_latency_sec", 0.0)),
        "avg_throughput_tps": float(report.get("avg_throughput_tps", 0.0)),
        "avg_peak_vram_mb": float(report.get("avg_peak_vram_mb", 0.0)),
        "max_peak_vram_mb": float(report.get("max_peak_vram_mb", 0.0)),
        "report_file": str(case_cfg.get("report_file", "")),
    }


def _select_best_case(rows: list[dict[str, Any]], rules: list[str]) -> dict[str, Any] | None:
    if not rows:
        return None

    def key_fn(item: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for rule in rules:
            if rule.endswith("_asc"):
                metric = rule[: -len("_asc")]
                values.append(float(item.get(metric, 0.0)))
            else:
                values.append(-float(item.get(rule, 0.0)))
        return tuple(values)

    return min(rows, key=key_fn)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "label",
        "model_path",
        "backend",
        "quantization",
        "num_samples_evaluated",
        "parse_ok_rate",
        "exact_match_rate",
        "action_match_rate",
        "avg_latency_sec",
        "avg_throughput_tps",
        "avg_peak_vram_mb",
        "max_peak_vram_mb",
        "report_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], best_case: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp12 最终方案总对比报告",
        "",
        "## 结果总表",
        "",
        "| 方案 | Parse OK | Exact Match | Action Match | Avg Latency (s) | Tokens/s | Avg VRAM (MB) | Max VRAM (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {parse:.4f} | {exact:.4f} | {action:.4f} | {lat:.4f} | {tps:.4f} | {avg_vram:.1f} | {max_vram:.1f} |".format(
                label=row["label"],
                parse=float(row["parse_ok_rate"]),
                exact=float(row["exact_match_rate"]),
                action=float(row["action_match_rate"]),
                lat=float(row["avg_latency_sec"]),
                tps=float(row["avg_throughput_tps"]),
                avg_vram=float(row["avg_peak_vram_mb"]),
                max_vram=float(row["max_peak_vram_mb"]),
            )
        )

    lines.extend(["", "## 简要结论", ""])
    if best_case is None:
        lines.append("- 当前没有可用结果。")
    else:
        lines.append(
            "- 按当前配置的主排序规则，综合最优方案为 `{label}`，其 `exact_match_rate={exact:.4f}`，`action_match_rate={action:.4f}`。".format(
                label=best_case["label"],
                exact=float(best_case["exact_match_rate"]),
                action=float(best_case["action_match_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )
    compare_cfg = _get_compare_section(merged_config)
    reports_dir = Path(compare_cfg.get("reports_dir", "experiments/16_exp12_final_compare/reports"))
    cases = compare_cfg.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("final_compare.cases 至少需要配置一个方案。")

    meta_path = record_run_meta(
        reports_dir,
        merged_config=merged_config,
        cli_args=vars(args),
        argv=sys.argv,
        seed=int(_ensure_accuracy_section(merged_config).get("seed", 42)),
        data_paths=[_ensure_accuracy_section(merged_config).get("test_file", "")],
        extra_meta={"entry": "experiments/16_exp12_final_compare/run_exp12_final_compare.py", "stage": "final_compare"},
    )
    print(f"[ok] run meta          : {meta_path}")

    summaries: list[dict[str, Any]] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            continue
        case_name = str(raw_case.get("name", "")).strip()
        label = str(raw_case.get("label", case_name)).strip()
        if not case_name:
            raise ValueError("存在未配置 name 的 case。")
        print(f"[exp12] running case  : {label}")
        case_config = _build_case_config(merged_config, raw_case)
        report = run_accuracy_from_merged_config(case_config)
        summary = _summarize_case(raw_case, report)
        summaries.append(summary)
        print(
            "[ok] {name:<16} parse_ok={parse:.4f}  exact={exact:.4f}  action={action:.4f}  latency={lat:.4f}s".format(
                name=case_name,
                parse=summary["parse_ok_rate"],
                exact=summary["exact_match_rate"],
                action=summary["action_match_rate"],
                lat=summary["avg_latency_sec"],
            )
        )

    best_rules = compare_cfg.get(
        "select_best_by",
        ["exact_match_rate", "action_match_rate", "parse_ok_rate", "avg_latency_sec_asc"],
    )
    if not isinstance(best_rules, list):
        raise TypeError("final_compare.select_best_by 必须是数组。")
    best_case = _select_best_case(summaries, [str(item) for item in best_rules])

    summary_payload = {
        "experiment": "exp12_final_compare",
        "cases": summaries,
        "best_case": best_case,
        "selection_rules": [str(item) for item in best_rules],
        "run_meta_file": str(meta_path),
    }

    summary_json = Path(compare_cfg.get("summary_json", reports_dir / "final_compare_summary.json"))
    summary_csv = Path(compare_cfg.get("summary_csv", reports_dir / "final_compare_summary.csv"))
    summary_md = Path(compare_cfg.get("summary_md", reports_dir / "final_compare_summary.md"))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(safe_json_dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv, summaries)
    _write_md(summary_md, summaries, best_case)

    print(f"[ok] summary json      : {summary_json}")
    print(f"[ok] summary csv       : {summary_csv}")
    print(f"[ok] summary md        : {summary_md}")


if __name__ == "__main__":
    main()
