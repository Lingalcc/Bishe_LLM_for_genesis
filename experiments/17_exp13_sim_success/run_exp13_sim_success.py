#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.evaluate_toolcall_accuracy import canonicalize_commands, payload_to_commands
from src.sim_core.runtime import SimRuntimeConfig, run_action_to_motion, run_instruction_to_action
from src.utils.config import load_merged_config
from src.utils.run_meta import record_run_meta
from src.utils.secrets import safe_json_dumps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端仿真成功率实验。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/17_exp13_sim_success/configs/sim_success.yaml"),
        help="实验配置文件路径。",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="全局基础配置路径。",
    )
    return parser.parse_args()


def _get_exp_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("exp13_sim_success", {})
    if not isinstance(section, dict):
        raise TypeError("exp13_sim_success 配置必须是对象。")
    return section


def _load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"数据集必须是 JSON 数组：{path}")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction", "")
        output_text = item.get("output", "")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        if not isinstance(output_text, str) or not output_text.strip():
            continue
        try:
            gt_commands = payload_to_commands(output_text)
        except Exception:
            continue
        rows.append(
            {
                "dataset_index": idx,
                "instruction": instruction,
                "gt_output": output_text,
                "gt_commands": gt_commands,
            }
        )
    return rows


def _classify_failure(stage: str, exc: Exception | None, execution_results: list[dict[str, Any]] | None = None) -> str:
    if stage == "execution" and execution_results:
        first_error = next((item for item in execution_results if item.get("status") != "ok"), None)
        if first_error is not None:
            action = str(first_error.get("action", "unknown"))
            error = str(first_error.get("error", "")).strip()
            if error:
                return f"execution:{action}:{error.split(':', 1)[0]}"
            return f"execution:{action}:command_error"
    if exc is None:
        return f"{stage}:unknown"
    return f"{stage}:{type(exc).__name__}"


def _build_summary(rows: list[dict[str, Any]], failure_counter: Counter[str]) -> dict[str, Any]:
    total = len(rows)
    parse_ok = sum(1 for row in rows if row["parse_ok"])
    exact = sum(1 for row in rows if row["exact_match"])
    action = sum(1 for row in rows if row["action_match"])
    exec_success = sum(1 for row in rows if row["execution_success"])
    total_commands = sum(int(row["num_commands"]) for row in rows)
    ok_commands = sum(int(row["num_ok_commands"]) for row in rows)
    e2e_times = [float(row["end_to_end_sec"]) for row in rows if float(row["end_to_end_sec"]) > 0.0]
    command_counts = [int(row["num_commands"]) for row in rows]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "num_samples": total,
        "parse_ok_count": parse_ok,
        "parse_ok_rate": (parse_ok / total) if total else 0.0,
        "exact_match_count": exact,
        "exact_match_rate": (exact / total) if total else 0.0,
        "action_match_count": action,
        "action_match_rate": (action / total) if total else 0.0,
        "execution_success_count": exec_success,
        "execution_success_rate": (exec_success / total) if total else 0.0,
        "command_success_rate": (ok_commands / total_commands) if total_commands else 0.0,
        "avg_end_to_end_sec": _mean(e2e_times),
        "avg_num_commands": _mean([float(v) for v in command_counts]),
        "total_commands": total_commands,
        "ok_commands": ok_commands,
        "failure_type_counts": dict(sorted(failure_counter.items(), key=lambda item: (-item[1], item[0]))),
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "num_samples", "value": summary["num_samples"]},
        {"metric": "parse_ok_rate", "value": summary["parse_ok_rate"]},
        {"metric": "exact_match_rate", "value": summary["exact_match_rate"]},
        {"metric": "action_match_rate", "value": summary["action_match_rate"]},
        {"metric": "execution_success_rate", "value": summary["execution_success_rate"]},
        {"metric": "command_success_rate", "value": summary["command_success_rate"]},
        {"metric": "avg_end_to_end_sec", "value": summary["avg_end_to_end_sec"]},
        {"metric": "avg_num_commands", "value": summary["avg_num_commands"]},
        {"metric": "total_commands", "value": summary["total_commands"]},
        {"metric": "ok_commands", "value": summary["ok_commands"]},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp13 端到端仿真成功率报告",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Parse OK Rate | {float(summary['parse_ok_rate']):.4f} |",
        f"| Exact Match Rate | {float(summary['exact_match_rate']):.4f} |",
        f"| Action Match Rate | {float(summary['action_match_rate']):.4f} |",
        f"| Execution Success Rate | {float(summary['execution_success_rate']):.4f} |",
        f"| Command Success Rate | {float(summary['command_success_rate']):.4f} |",
        f"| Avg End-to-End (s) | {float(summary['avg_end_to_end_sec']):.4f} |",
        f"| Avg Num Commands | {float(summary['avg_num_commands']):.2f} |",
        "",
        "## 失败类型统计",
        "",
    ]
    failure_counts = summary.get("failure_type_counts", {})
    if isinstance(failure_counts, dict) and failure_counts:
        lines.extend(["| 类型 | 次数 |", "| --- | ---: |"])
        for name, count in failure_counts.items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("- 当前没有失败样本。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )
    exp_cfg = _get_exp_config(merged_config)
    dataset_file = Path(exp_cfg.get("dataset_file", "data_prepare/splits/test.json"))
    if not dataset_file.is_absolute():
        dataset_file = (REPO_ROOT / dataset_file).resolve()
    rows = _load_dataset_rows(dataset_file)

    num_samples = min(int(exp_cfg.get("num_samples", 20)), len(rows))
    seed = int(exp_cfg.get("seed", 42))
    rng = random.Random(seed)
    selected = rng.sample(rows, num_samples) if num_samples < len(rows) else list(rows)

    reports_dir = Path(exp_cfg.get("reports_dir", "experiments/17_exp13_sim_success/reports"))
    meta_path = record_run_meta(
        reports_dir,
        merged_config=merged_config,
        cli_args=vars(args),
        argv=sys.argv,
        seed=seed,
        data_paths=[dataset_file],
        extra_meta={"entry": "experiments/17_exp13_sim_success/run_exp13_sim_success.py", "stage": "sim_success"},
    )
    print(f"[ok] run meta          : {meta_path}")

    stop_on_error = bool(exp_cfg.get("stop_on_error", False))
    details: list[dict[str, Any]] = []
    failure_counter: Counter[str] = Counter()

    for idx, sample in enumerate(selected, start=1):
        instruction = str(sample["instruction"])
        gt_commands = sample["gt_commands"]
        stage = "generation"
        generated_payload: dict[str, Any] | None = None
        execution_results: list[dict[str, Any]] | None = None
        current_error: Exception | None = None
        record: dict[str, Any] = {
            "sample_index": idx,
            "dataset_index": int(sample["dataset_index"]),
            "instruction": instruction,
            "parse_ok": False,
            "exact_match": False,
            "action_match": False,
            "execution_success": False,
            "num_commands": 0,
            "num_ok_commands": 0,
            "end_to_end_sec": 0.0,
            "failure_stage": None,
            "failure_type": None,
            "error": None,
        }

        started_at = time.perf_counter()
        try:
            action_result = run_instruction_to_action(
                SimRuntimeConfig(
                    instruction=instruction,
                    print_raw=False,
                    disable_sim_state=False,
                ),
                merged_config=copy.deepcopy(merged_config),
            )
            generated_payload = action_result["payload"]
            pred_commands = payload_to_commands(generated_payload)
            record["parse_ok"] = True
            record["num_commands"] = len(pred_commands)
            record["exact_match"] = canonicalize_commands(pred_commands) == canonicalize_commands(gt_commands)
            record["action_match"] = [str(c.get("action", "")) for c in pred_commands] == [
                str(c.get("action", "")) for c in gt_commands
            ]

            stage = "execution"
            execution_result = run_action_to_motion(
                SimRuntimeConfig(
                    action=json.dumps(generated_payload, ensure_ascii=False),
                    print_raw=False,
                    disable_sim_state=False,
                ),
                merged_config=copy.deepcopy(merged_config),
            )
            execution_results = execution_result.get("results", [])
            if not isinstance(execution_results, list):
                execution_results = []

            ok_commands = sum(1 for item in execution_results if item.get("status") == "ok")
            record["num_ok_commands"] = ok_commands
            record["execution_success"] = bool(execution_results) and ok_commands == len(execution_results)
            if record["num_commands"] == 0:
                record["num_commands"] = len(execution_results)
        except Exception as exc:
            current_error = exc
            record["failure_stage"] = stage
            record["error"] = f"{type(exc).__name__}: {exc}"
            failure_type = _classify_failure(stage, exc, execution_results)
            record["failure_type"] = failure_type
            failure_counter[failure_type] += 1
            if stop_on_error:
                raise
        finally:
            record["end_to_end_sec"] = time.perf_counter() - started_at

        if record["parse_ok"] and not record["execution_success"] and record["failure_type"] is None:
            failure_type = _classify_failure("execution", current_error, execution_results)
            record["failure_stage"] = "execution"
            record["failure_type"] = failure_type
            failure_counter[failure_type] += 1

        if generated_payload is not None:
            record["payload"] = generated_payload
        if execution_results is not None:
            record["execution_results"] = execution_results
        details.append(record)

        print(
            "[exp13] {idx}/{total} parse_ok={parse} exact={exact} action={action} exec_success={exec_ok} time={cost:.4f}s".format(
                idx=idx,
                total=num_samples,
                parse=int(record["parse_ok"]),
                exact=int(record["exact_match"]),
                action=int(record["action_match"]),
                exec_ok=int(record["execution_success"]),
                cost=float(record["end_to_end_sec"]),
            )
        )

    summary = _build_summary(details, failure_counter)
    report = {
        "experiment": "exp13_sim_success",
        "dataset_file": str(dataset_file),
        "seed": seed,
        "summary": summary,
        "samples": details,
        "run_meta_file": str(meta_path),
    }

    report_json = Path(exp_cfg.get("report_json", reports_dir / "sim_success_report.json"))
    summary_csv = Path(exp_cfg.get("summary_csv", reports_dir / "sim_success_summary.csv"))
    summary_md = Path(exp_cfg.get("summary_md", reports_dir / "sim_success_summary.md"))
    samples_json = Path(exp_cfg.get("samples_json", reports_dir / "sim_success_samples.json"))

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(safe_json_dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    samples_json.write_text(safe_json_dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, summary)
    _write_summary_md(summary_md, summary)

    print(f"[ok] report json       : {report_json}")
    print(f"[ok] samples json      : {samples_json}")
    print(f"[ok] summary csv       : {summary_csv}")
    print(f"[ok] summary md        : {summary_md}")


if __name__ == "__main__":
    main()
