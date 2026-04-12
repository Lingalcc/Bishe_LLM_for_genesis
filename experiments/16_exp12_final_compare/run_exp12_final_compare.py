#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.evaluate_toolcall_accuracy import canonicalize_commands, payload_to_commands
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


def _load_exp8_module() -> Any:
    module_path = REPO_ROOT / "experiments" / "12_exp8_speculative_decoding" / "run_speculative_benchmark.py"
    spec = spec_from_file_location("exp8_speculative_benchmark_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 exp8 模块: {module_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_peft_awq_dispatch_import_error() -> None:
    try:
        import peft.tuners.lora.model as lora_model
    except Exception:
        return

    if getattr(lora_model, "_genesis_awq_safe_patch_applied", False):
        return

    original_dispatch_awq = getattr(lora_model, "dispatch_awq", None)
    if not callable(original_dispatch_awq):
        return

    def _safe_dispatch_awq(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_dispatch_awq(*args, **kwargs)
        except ImportError:
            return None

    lora_model.dispatch_awq = _safe_dispatch_awq
    lora_model._genesis_awq_safe_patch_applied = True


def _build_speculative_runtime_config(base_config: dict[str, Any], case_cfg: dict[str, Any]) -> dict[str, Any]:
    acc_cfg = _ensure_accuracy_section(base_config)
    num_samples = int(case_cfg.get("num_samples", acc_cfg.get("num_samples", 100)))
    dataset_path_raw = str(
        case_cfg.get("dataset_path", acc_cfg.get("test_file", acc_cfg.get("dataset_file", "")))
    ).strip()
    if dataset_path_raw and not Path(dataset_path_raw).is_absolute():
        dataset_path_raw = str((REPO_ROOT / dataset_path_raw).resolve())
    runtime_config = {
        "target_model_path": str(case_cfg.get("target_model_path", case_cfg.get("model_path", ""))).strip(),
        "assistant_model_path": str(case_cfg.get("assistant_model_path", "")).strip(),
        "dataset_path": dataset_path_raw,
        "num_samples": num_samples,
        "batch_size": int(case_cfg.get("batch_size", 1)),
        "max_new_tokens": int(case_cfg.get("max_new_tokens", acc_cfg.get("max_new_tokens", 512))),
        "temperature": float(case_cfg.get("temperature", acc_cfg.get("temperature", 0.0))),
        "warmup_samples": int(case_cfg.get("warmup_samples", 5)),
        "assistant_num_tokens": int(case_cfg.get("assistant_num_tokens", 8)),
        "assistant_confidence_threshold": float(case_cfg.get("assistant_confidence_threshold", 0.55)),
        "assistant_num_tokens_schedule": str(case_cfg.get("assistant_num_tokens_schedule", "constant")),
        "prefer_same_gpu": bool(case_cfg.get("prefer_same_gpu", True)),
        "preferred_cuda_device": int(case_cfg.get("preferred_cuda_device", 0)),
        "allow_auto_device_map_fallback": bool(case_cfg.get("allow_auto_device_map_fallback", True)),
        "trust_remote_code": bool(case_cfg.get("trust_remote_code", acc_cfg.get("trust_remote_code", True))),
        "prefer_flash_attention_2": bool(case_cfg.get("prefer_flash_attention_2", True)),
        "report_path": str(case_cfg.get("report_file", "")),
    }
    if not runtime_config["target_model_path"]:
        raise ValueError("speculative case 缺少 target_model_path/model_path。")
    if not runtime_config["assistant_model_path"]:
        raise ValueError("speculative case 缺少 assistant_model_path。")
    if not runtime_config["dataset_path"]:
        raise ValueError("speculative case 缺少 dataset_path/test_file。")
    return runtime_config


def _evaluate_prediction_metrics(prediction: str, expected_output: str) -> dict[str, Any]:
    try:
        pred_cmds = payload_to_commands(prediction)
        gt_cmds = payload_to_commands(expected_output)
    except Exception as exc:
        return {
            "parse_ok": False,
            "exact_match": False,
            "action_match": False,
            "error": str(exc),
        }

    pred_sig = [str(c.get("action", "")) for c in pred_cmds]
    gt_sig = [str(c.get("action", "")) for c in gt_cmds]
    return {
        "parse_ok": True,
        "exact_match": canonicalize_commands(pred_cmds) == canonicalize_commands(gt_cmds),
        "action_match": pred_sig == gt_sig,
        "error": None,
    }


def _run_speculative_case(base_config: dict[str, Any], case_cfg: dict[str, Any]) -> dict[str, Any]:
    _patch_peft_awq_dispatch_import_error()
    exp8_module = _load_exp8_module()
    runtime_config = _build_speculative_runtime_config(base_config, case_cfg)
    system_prompt = exp8_module._load_system_prompt(REPO_ROOT / "configs" / "base.yaml")
    dataset_path = exp8_module._resolve_dataset_path(runtime_config["dataset_path"])
    dataset = exp8_module._load_dataset(dataset_path, int(runtime_config["num_samples"]))

    import torch

    dtype, dtype_label = exp8_module._select_torch_dtype(torch)
    tokenizer_source = exp8_module._maybe_load_adapter(
        exp8_module._resolve_local_or_hf_ref(str(runtime_config["target_model_path"]))
    )[0]
    tokenizer = exp8_module._load_tokenizer(
        tokenizer_source=tokenizer_source,
        trust_remote_code=bool(runtime_config.get("trust_remote_code", True)),
    )
    case_name = str(case_cfg.get("name", "speculative_case")).strip() or "speculative_case"
    raw_report = exp8_module.run_case(
        case_name=case_name,
        assistant_enabled=True,
        config=runtime_config,
        dataset=dataset,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        dtype=dtype,
        dtype_label=dtype_label,
    )

    details: list[dict[str, Any]] = []
    parse_ok = 0
    exact_match = 0
    action_match = 0
    latencies: list[float] = []
    throughputs: list[float] = []
    peak_vrams: list[float] = []
    for item in raw_report.get("samples", []):
        if not isinstance(item, dict):
            continue
        metrics = _evaluate_prediction_metrics(
            prediction=str(item.get("prediction", "")),
            expected_output=str(item.get("expected_output", "")),
        )
        lat = float(item.get("latency_sec", 0.0))
        tps = float(item.get("batch_throughput_tps", 0.0))
        vram = float(item.get("batch_peak_vram_mb", 0.0))
        latencies.append(lat)
        throughputs.append(tps)
        peak_vrams.append(vram)
        parse_ok += int(bool(metrics["parse_ok"]))
        exact_match += int(bool(metrics["exact_match"]))
        action_match += int(bool(metrics["action_match"]))
        details.append(
            {
                "dataset_index": int(item.get("sample_index", -1)),
                "parse_ok": bool(metrics["parse_ok"]),
                "exact_match": bool(metrics["exact_match"]),
                "action_match": bool(metrics["action_match"]),
                "error": metrics["error"],
                "latency_sec": lat,
                "throughput_tps": tps,
                "peak_vram_mb": vram,
            }
        )

    total = len(details)
    report = {
        "mode": "local",
        "model_path": runtime_config["target_model_path"],
        "assistant_model_path": runtime_config["assistant_model_path"],
        "backend": "transformers",
        "quantization": None,
        "dataset_file": str(dataset_path),
        "seed": int(_ensure_accuracy_section(base_config).get("seed", 42)),
        "num_samples_evaluated": total,
        "num_valid_rows_in_dataset": total,
        "parse_ok": parse_ok,
        "parse_ok_rate": (parse_ok / total) if total else 0.0,
        "exact_match": exact_match,
        "exact_match_rate": (exact_match / total) if total else 0.0,
        "action_match": action_match,
        "action_match_rate": (action_match / total) if total else 0.0,
        "avg_latency_sec": float(raw_report.get("avg_latency_sec_per_sample", 0.0)),
        "avg_throughput_tps": float(raw_report.get("token_throughput_tps", 0.0)),
        "avg_peak_vram_mb": (sum(peak_vrams) / len(peak_vrams)) if peak_vrams else float(raw_report.get("peak_vram_mb", 0.0)),
        "max_peak_vram_mb": max(peak_vrams) if peak_vrams else float(raw_report.get("peak_vram_mb", 0.0)),
        "details": details,
        "speculative": {
            "assistant_num_tokens": raw_report.get("assistant_num_tokens"),
            "assistant_confidence_threshold": raw_report.get("assistant_confidence_threshold"),
            "assistant_num_tokens_schedule": raw_report.get("assistant_num_tokens_schedule"),
            "dtype": raw_report.get("dtype"),
            "attn_implementation": raw_report.get("attn_implementation"),
            "target_primary_device": raw_report.get("target_primary_device"),
            "assistant_primary_device": raw_report.get("assistant_primary_device"),
            "target_load_strategy": raw_report.get("target_load_strategy"),
            "assistant_load_strategy": raw_report.get("assistant_load_strategy"),
            "raw_report": raw_report,
        },
    }

    report_file_raw = str(case_cfg.get("report_file", "")).strip()
    if report_file_raw:
        report_file = Path(report_file_raw)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(safe_json_dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _summarize_case(case_cfg: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(case_cfg.get("name", "")),
        "label": str(case_cfg.get("label", case_cfg.get("name", ""))),
        "model_path": str(report.get("model_path", case_cfg.get("model_path", ""))),
        "target_model_path": str(case_cfg.get("target_model_path", report.get("model_path", case_cfg.get("model_path", "")))),
        "assistant_model_path": str(case_cfg.get("assistant_model_path", report.get("assistant_model_path", ""))),
        "runner": str(case_cfg.get("runner", "accuracy")),
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
        "target_model_path",
        "assistant_model_path",
        "runner",
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
                label=f'{row["label"]} ({row.get("runner", "accuracy")})',
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
    _patch_peft_awq_dispatch_import_error()
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
        runner = str(raw_case.get("runner", "accuracy")).strip().lower()
        if not case_name:
            raise ValueError("存在未配置 name 的 case。")
        print(f"[exp12] running case  : {label} ({runner})")
        if runner == "speculative":
            report = _run_speculative_case(merged_config, raw_case)
        else:
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
