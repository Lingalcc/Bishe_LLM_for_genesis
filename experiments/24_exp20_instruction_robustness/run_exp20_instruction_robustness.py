#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.evaluate_toolcall_accuracy import canonicalize_commands, normalize_text, payload_to_commands
from src.eval_core.performance_monitor import estimate_tokens_from_text, time_and_memory_tracker
from src.eval_core.prompting import DEFAULT_EVAL_SYSTEM_PROMPT, build_eval_messages


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "robustness.yaml"

STRESS_DATASET_PATH = REPORTS_DIR / "exp20_stress_dataset.json"
SUMMARY_JSON_PATH = REPORTS_DIR / "exp20_instruction_robustness_summary.json"
SUMMARY_CSV_PATH = REPORTS_DIR / "exp20_instruction_robustness_summary.csv"
SUMMARY_MD_PATH = REPORTS_DIR / "exp20_instruction_robustness_summary.md"
DETAILS_JSONL_PATH = REPORTS_DIR / "exp20_instruction_robustness_details.jsonl"


CATEGORY_LABELS: dict[str, str] = {
    "standard": "标准指令",
    "complex_multistep": "复杂多步指令",
    "noise_redundant": "含噪声/冗余描述指令",
    "long_context": "长文本指令",
    "boundary_ambiguous": "边界或含糊指令",
}

CATEGORY_ORDER = tuple(CATEGORY_LABELS.keys())


@dataclass(frozen=True)
class SourceRow:
    dataset_index: int
    instruction: str
    output: str
    system: str
    gt_commands: list[dict[str, Any]]


@dataclass(frozen=True)
class StressCase:
    case_id: str
    category: str
    category_label: str
    source_dataset_index: int
    instruction: str
    output: str
    system: str
    gt_commands: list[dict[str, Any]]
    input_chars: int
    estimated_input_tokens: int
    command_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp20：按输入类型评测最终微调推理方案的指令鲁棒性。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="实验配置 YAML。")
    parser.add_argument(
        "--mode",
        choices=("build-only", "predictions", "local"),
        default=None,
        help="build-only 只生成 stress set；predictions 离线评测预测文件；local 调用本地最终模型。",
    )
    parser.add_argument("--dataset-file", type=Path, default=None, help="原始测试集 JSON。")
    parser.add_argument("--predictions-file", type=Path, default=None, help="离线预测 JSON/JSONL 文件。")
    parser.add_argument("--model-path", default=None, help="本地模型路径。")
    parser.add_argument("--backend", default=None, help="本地推理后端，如 transformers/vllm。")
    parser.add_argument("--quantization", default=None, help="量化方式，如 compressed-tensors/awq/4bit。")
    parser.add_argument("--per-category", type=int, default=None, help="每类样本数。")
    parser.add_argument("--seed", type=int, default=None, help="抽样随机种子。")
    parser.add_argument("--max-model-len", type=int, default=None, help="本地推理最大上下文长度。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="本地推理最大生成长度。")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度。")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM 显存利用率。")
    parser.add_argument("--trust-remote-code", action="store_true", help="本地模型加载时启用 trust_remote_code。")
    parser.add_argument(
        "--skip-vllm-compat-check",
        action="store_true",
        help="跳过仓库内对 vLLM / compressed-tensors 锁定组合的保守检查。",
    )
    parser.add_argument(
        "--strict-vllm-compat-check",
        action="store_true",
        help="强制启用 vLLM 兼容性检查，覆盖配置文件中的 skip_vllm_compat_check。",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return payload


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_source_rows(dataset_file: Path) -> list[SourceRow]:
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"dataset must be a JSON list: {dataset_file}")

    valid_rows: list[SourceRow] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        instruction = row.get("instruction")
        output = row.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        if not isinstance(output, str) or not output.strip():
            continue
        try:
            gt_commands = payload_to_commands(output)
        except Exception:
            continue
        system = row.get("system", "")
        valid_rows.append(
            SourceRow(
                dataset_index=i,
                instruction=instruction.strip(),
                output=output,
                system=system if isinstance(system, str) else "",
                gt_commands=gt_commands,
            )
        )
    if not valid_rows:
        raise RuntimeError(f"no valid rows found in dataset: {dataset_file}")
    return valid_rows


def _has_state_context(text: str) -> bool:
    return "[STATE_CONTEXT]" in text and "[/STATE_CONTEXT]" in text


def _is_complex(row: SourceRow) -> bool:
    instruction = row.instruction
    return (
        len(row.gt_commands) >= 3
        or "先" in instruction and ("然后" in instruction or "最后" in instruction)
        or "1)" in instruction
        or "复杂" in instruction
    )


def _is_boundary_like(row: SourceRow) -> bool:
    text = row.instruction
    boundary_terms = ("边界", "上方", "靠近", "安全", "中心", "默认", "初始", "稍微", "轻微", "大约")
    return _has_state_context(text) or any(term in text for term in boundary_terms)


def _sample_rows(
    rows: list[SourceRow],
    *,
    rng: random.Random,
    per_category: int,
    predicate: Any,
) -> list[SourceRow]:
    candidates = [row for row in rows if predicate(row)]
    if len(candidates) < per_category:
        seen = {row.dataset_index for row in candidates}
        candidates.extend(row for row in rows if row.dataset_index not in seen)
    if len(candidates) <= per_category:
        return list(candidates)
    return rng.sample(candidates, k=per_category)


def make_noisy_instruction(instruction: str) -> str:
    return (
        "下面这段话包含任务背景、操作者备注和无关信息。请忽略闲聊、重复说明和安全提醒，"
        "只把真正的机器人动作要求转换为 JSON action。"
        "背景：今天的实验台已经完成清洁，摄像头编号为 C2，日志文件稍后再保存。"
        "备注：不要输出解释，不要把本句当成机械臂动作。"
        f"真正需要执行的指令是：{instruction}。"
        "重复确认：以上背景不是动作，动作只来自“真正需要执行的指令”。"
    )


def make_long_instruction(instruction: str, *, target_chars: int) -> str:
    context = (
        "实验记录：本轮评测用于观察长上下文条件下模型是否仍能定位核心任务。"
        "前面的文字包括设备状态、人员记录、桌面清理情况、标定说明和历史操作摘要，"
        "它们不应被转换为动作。机械臂已经完成安全检查，夹爪无异常，工作空间无遮挡。"
        "如果上下文中出现多个背景描述，模型应只执行明确标出的当前用户指令。"
    )
    chunks: list[str] = []
    while sum(len(chunk) for chunk in chunks) < target_chars:
        chunks.append(context)
    return (
        "".join(chunks)
        + "当前用户指令开始："
        + instruction
        + "。当前用户指令结束。请只输出该指令对应的 JSON action。"
    )


def make_boundary_instruction(instruction: str) -> str:
    return (
        "用户表达中可能包含“靠近、稍微、安全距离、默认姿态”等边界或含糊说法。"
        "若没有给出新数值，请沿用训练集中该指令原本隐含的常用默认值，不要额外发明动作。"
        f"核心指令：{instruction}"
    )


def make_complex_instruction(instruction: str) -> str:
    return (
        "请严格保持动作顺序，把下面的多步任务逐步转换为 commands 数组；"
        "不要省略等待、夹爪或状态查询动作，也不要把后续动作提前。"
        f"任务：{instruction}"
    )


def build_stress_cases(
    rows: list[SourceRow],
    *,
    per_category: int,
    seed: int,
    long_target_chars: int,
) -> list[StressCase]:
    rng = random.Random(seed)
    selected: dict[str, list[SourceRow]] = {
        "standard": _sample_rows(rows, rng=rng, per_category=per_category, predicate=lambda _: True),
        "complex_multistep": _sample_rows(rows, rng=rng, per_category=per_category, predicate=_is_complex),
        "noise_redundant": _sample_rows(rows, rng=rng, per_category=per_category, predicate=lambda _: True),
        "long_context": _sample_rows(rows, rng=rng, per_category=per_category, predicate=lambda _: True),
        "boundary_ambiguous": _sample_rows(rows, rng=rng, per_category=per_category, predicate=_is_boundary_like),
    }

    cases: list[StressCase] = []
    for category in CATEGORY_ORDER:
        for local_idx, row in enumerate(selected[category]):
            if category == "standard":
                instruction = row.instruction
            elif category == "complex_multistep":
                instruction = make_complex_instruction(row.instruction)
            elif category == "noise_redundant":
                instruction = make_noisy_instruction(row.instruction)
            elif category == "long_context":
                instruction = make_long_instruction(row.instruction, target_chars=long_target_chars)
            elif category == "boundary_ambiguous":
                instruction = make_boundary_instruction(row.instruction)
            else:
                raise ValueError(f"unknown category: {category}")

            cases.append(
                StressCase(
                    case_id=f"{category}_{local_idx:03d}_src{row.dataset_index}",
                    category=category,
                    category_label=CATEGORY_LABELS[category],
                    source_dataset_index=row.dataset_index,
                    instruction=instruction,
                    output=row.output,
                    system=row.system,
                    gt_commands=row.gt_commands,
                    input_chars=len(instruction),
                    estimated_input_tokens=estimate_tokens_from_text(instruction),
                    command_count=len(row.gt_commands),
                )
            )
    return cases


def stress_case_to_dataset_row(case: StressCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "category_label": case.category_label,
        "source_dataset_index": case.source_dataset_index,
        "instruction": case.instruction,
        "output": case.output,
        "system": case.system,
        "input_chars": case.input_chars,
        "estimated_input_tokens": case.estimated_input_tokens,
        "command_count": case.command_count,
    }


def write_stress_dataset(cases: list[StressCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [stress_case_to_dataset_row(case) for case in cases]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_predictions(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_text_and_perf(predictions_blob: Any, case: StressCase, ordinal: int) -> tuple[str, dict[str, Any]]:
    item: Any
    if isinstance(predictions_blob, dict):
        item = predictions_blob.get(case.case_id)
        if item is None:
            item = predictions_blob.get(str(ordinal))
        if item is None:
            raise KeyError(f"predictions missing case_id={case.case_id}")
    elif isinstance(predictions_blob, list):
        if ordinal >= len(predictions_blob):
            raise IndexError("predictions list shorter than stress cases")
        item = predictions_blob[ordinal]
    else:
        raise TypeError("predictions file must contain a JSON object, list, or JSONL records")

    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        text = (
            item.get("prediction")
            or item.get("prediction_text")
            or item.get("content")
            or item.get("output")
            or item.get("response")
        )
        if not isinstance(text, str):
            raise ValueError(f"prediction item has no text field: {case.case_id}")
        perf = {
            "latency_sec": item.get("latency_sec"),
            "throughput_tps": item.get("throughput_tps"),
            "peak_vram_mb": item.get("peak_vram_mb"),
        }
        return text, perf
    raise TypeError(f"unsupported prediction item type for {case.case_id}: {type(item).__name__}")


def action_signature(commands: list[dict[str, Any]]) -> list[str]:
    return [str(command.get("action", "")) for command in commands]


def classify_error(
    *,
    parse_ok: bool,
    exact_match: bool,
    action_match: bool,
    pred_commands: list[dict[str, Any]] | None,
    gt_commands: list[dict[str, Any]],
) -> str:
    if not parse_ok:
        return "json_parse"
    if exact_match:
        return "none"
    if action_match:
        return "parameter_mapping"

    pred_sig = action_signature(pred_commands or [])
    gt_sig = action_signature(gt_commands)
    if len(pred_sig) == len(gt_sig) and Counter(pred_sig) == Counter(gt_sig):
        return "action_order"
    return "semantic_understanding"


def evaluate_prediction(case: StressCase, prediction_text: str, perf: dict[str, Any] | None = None) -> dict[str, Any]:
    perf = perf or {}
    gt_commands = case.gt_commands
    pred_commands: list[dict[str, Any]] | None = None
    parse_ok = False
    exact_match = False
    action_match = False
    error_message: str | None = None

    try:
        pred_commands = payload_to_commands(prediction_text)
        parse_ok = True
        exact_match = canonicalize_commands(pred_commands) == canonicalize_commands(gt_commands)
        action_match = action_signature(pred_commands) == action_signature(gt_commands)
    except Exception as exc:
        error_message = str(exc)

    error_type = classify_error(
        parse_ok=parse_ok,
        exact_match=exact_match,
        action_match=action_match,
        pred_commands=pred_commands,
        gt_commands=gt_commands,
    )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "category_label": case.category_label,
        "source_dataset_index": case.source_dataset_index,
        "input_chars": case.input_chars,
        "estimated_input_tokens": case.estimated_input_tokens,
        "command_count": case.command_count,
        "parse_ok": parse_ok,
        "exact_match": exact_match,
        "action_match": action_match,
        "error_type": error_type,
        "error": error_message,
        "gt_actions": action_signature(gt_commands),
        "pred_actions": action_signature(pred_commands or []),
        "latency_sec": _optional_float(perf.get("latency_sec")),
        "throughput_tps": _optional_float(perf.get("throughput_tps")),
        "peak_vram_mb": _optional_float(perf.get("peak_vram_mb")),
        "prediction_preview": normalize_text(prediction_text)[:220],
    }


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def run_predictions_eval(cases: list[StressCase], predictions_file: Path) -> list[dict[str, Any]]:
    predictions_blob = load_predictions(predictions_file)
    details = []
    for i, case in enumerate(cases):
        prediction_text, perf = _prediction_text_and_perf(predictions_blob, case, i)
        details.append(evaluate_prediction(case, prediction_text, perf))
    return details


def build_engine_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    local_cfg = cfg_get(config, "local", {}) or {}
    generation_cfg = local_cfg.get("generation", {}) if isinstance(local_cfg.get("generation"), dict) else {}
    engine_cfg = {
        "backend": args.backend or local_cfg.get("backend", "vllm"),
        "model_path": args.model_path or local_cfg.get("model_path", "model/qwen2.5-3b-top18-rank8-merged-awq"),
        "tokenizer_path": local_cfg.get("tokenizer_path"),
        "quantization": args.quantization if args.quantization is not None else local_cfg.get("quantization"),
        "max_new_tokens": args.max_new_tokens or generation_cfg.get("max_new_tokens", 512),
        "max_model_len": args.max_model_len or local_cfg.get("max_model_len", 4096),
        "temperature": args.temperature if args.temperature is not None else generation_cfg.get("temperature", 0.0),
        "trust_remote_code": bool(args.trust_remote_code or local_cfg.get("trust_remote_code", True)),
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else local_cfg.get("gpu_memory_utilization", 0.8)
        ),
        "vllm_dtype": local_cfg.get("vllm_dtype"),
        "use_flash_attention": bool(local_cfg.get("use_flash_attention", False)),
    }
    return engine_cfg


def resolve_skip_vllm_compat_check(config: dict[str, Any], args: argparse.Namespace) -> bool:
    skip_check = bool(cfg_get(config, "local.skip_vllm_compat_check", False))
    if args.skip_vllm_compat_check:
        skip_check = True
    if args.strict_vllm_compat_check:
        skip_check = False
    return skip_check


def apply_vllm_compat_policy(skip_check: bool) -> None:
    if skip_check:
        os.environ["LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK"] = "1"
    else:
        os.environ.pop("LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK", None)


def run_local_eval(cases: list[StressCase], config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    from src.eval_core.inference_engines import build_inference_engine

    skip_check = resolve_skip_vllm_compat_check(config, args)
    apply_vllm_compat_policy(skip_check)
    if skip_check:
        print("[info] skip vLLM compatibility check: LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK=1", flush=True)

    engine_cfg = build_engine_config(config, args)
    engine = build_inference_engine(engine_cfg)
    system_prompt = str(cfg_get(config, "system_prompt", DEFAULT_EVAL_SYSTEM_PROMPT))
    details = []
    for i, case in enumerate(cases, start=1):
        messages = build_eval_messages(
            instruction=case.instruction,
            cfg_system_prompt=system_prompt,
            sample_system_prompt=case.system,
        )
        prediction_text = ""
        infer_error: str | None = None
        perf: dict[str, Any] = {}
        try:
            with time_and_memory_tracker(input_text=case.instruction) as tracker:
                prediction_text = engine.generate_chat(messages)
                tracker.set_output_text(prediction_text)
            perf = tracker.metrics
        except Exception as exc:
            infer_error = f"{type(exc).__name__}: {exc}"
            prediction_text = ""

        detail = evaluate_prediction(case, prediction_text, perf)
        if infer_error is not None:
            detail["error"] = infer_error
            detail["error_type"] = "inference_runtime"
        details.append(detail)
        if i % 10 == 0:
            print(f"[progress] evaluated {i}/{len(cases)}", flush=True)
    return details


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def summarize_details(details: list[dict[str, Any]], *, mode: str, cases: list[StressCase]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_category[str(detail["category"])].append(detail)

    category_rows: list[dict[str, Any]] = []
    standard_parse: float | None = None
    standard_action: float | None = None
    for category in CATEGORY_ORDER:
        items = by_category.get(category, [])
        total = len(items)
        latencies = [float(item["latency_sec"]) for item in items if item.get("latency_sec") is not None]
        error_counts = Counter(str(item.get("error_type", "unknown")) for item in items)
        row = {
            "category": category,
            "category_label": CATEGORY_LABELS[category],
            "total": total,
            "parse_ok_rate": _rate(sum(1 for item in items if item.get("parse_ok")), total),
            "exact_match_rate": _rate(sum(1 for item in items if item.get("exact_match")), total),
            "action_match_rate": _rate(sum(1 for item in items if item.get("action_match")), total),
            "avg_latency_sec": _mean(latencies),
            "avg_input_chars": _mean([float(item["input_chars"]) for item in items]),
            "avg_input_tokens_est": _mean([float(item["estimated_input_tokens"]) for item in items]),
            "avg_command_count": _mean([float(item["command_count"]) for item in items]),
            "error_counts": dict(error_counts),
            "semantic_understanding_errors": int(error_counts.get("semantic_understanding", 0)),
            "action_order_errors": int(error_counts.get("action_order", 0)),
            "parameter_mapping_errors": int(error_counts.get("parameter_mapping", 0)),
            "json_parse_errors": int(error_counts.get("json_parse", 0)),
        }
        if category == "standard":
            standard_parse = float(row["parse_ok_rate"])
            standard_action = float(row["action_match_rate"])
            row["parse_drop_vs_standard"] = 0.0
            row["action_drop_vs_standard"] = 0.0
        else:
            row["parse_drop_vs_standard"] = (
                None if standard_parse is None else standard_parse - float(row["parse_ok_rate"])
            )
            row["action_drop_vs_standard"] = (
                None if standard_action is None else standard_action - float(row["action_match_rate"])
            )
        category_rows.append(row)

    latency_items = [item for item in details if item.get("latency_sec") is not None]
    length_latency = {
        "num_latency_points": len(latency_items),
        "input_chars_latency_pearson": pearson(
            [float(item["input_chars"]) for item in latency_items],
            [float(item["latency_sec"]) for item in latency_items],
        ),
        "estimated_tokens_latency_pearson": pearson(
            [float(item["estimated_input_tokens"]) for item in latency_items],
            [float(item["latency_sec"]) for item in latency_items],
        ),
    }

    total = len(details)
    return {
        "experiment": "exp20_instruction_robustness",
        "mode": mode,
        "stress_dataset": str(STRESS_DATASET_PATH),
        "num_cases": len(cases),
        "num_evaluated": total,
        "overall": {
            "parse_ok_rate": _rate(sum(1 for item in details if item.get("parse_ok")), total),
            "exact_match_rate": _rate(sum(1 for item in details if item.get("exact_match")), total),
            "action_match_rate": _rate(sum(1 for item in details if item.get("action_match")), total),
            "avg_latency_sec": _mean(
                [float(item["latency_sec"]) for item in details if item.get("latency_sec") is not None]
            ),
            "error_counts": dict(Counter(str(item.get("error_type", "unknown")) for item in details)),
        },
        "category_rows": category_rows,
        "length_latency": length_latency,
    }


def write_details_jsonl(details: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for detail in details:
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "category",
        "category_label",
        "total",
        "parse_ok_rate",
        "parse_drop_vs_standard",
        "exact_match_rate",
        "action_match_rate",
        "action_drop_vs_standard",
        "avg_latency_sec",
        "avg_input_chars",
        "avg_input_tokens_est",
        "semantic_understanding_errors",
        "action_order_errors",
        "parameter_mapping_errors",
        "json_parse_errors",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["category_rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _dominant_error(row: dict[str, Any]) -> str:
    candidates = {
        "语义理解": int(row.get("semantic_understanding_errors", 0)),
        "动作顺序": int(row.get("action_order_errors", 0)),
        "参数映射": int(row.get("parameter_mapping_errors", 0)),
        "JSON 解析": int(row.get("json_parse_errors", 0)),
    }
    label, count = max(candidates.items(), key=lambda item: item[1])
    return "暂无明显错误" if count == 0 else label


def _interpret_category(row: dict[str, Any]) -> str:
    parse_drop = row.get("parse_drop_vs_standard")
    action_drop = row.get("action_drop_vs_standard")
    parse_text = "未观察到 JSON 可解析率相对标准指令下降"
    action_text = "未观察到动作匹配率相对标准指令下降"
    if isinstance(parse_drop, float) and parse_drop > 0.005:
        parse_text = f"JSON 可解析率下降约 {parse_drop:.4f}"
    if isinstance(action_drop, float) and action_drop > 0.005:
        action_text = f"动作匹配率下降约 {action_drop:.4f}"
    error_text = _dominant_error(row)
    latency = _fmt(row.get("avg_latency_sec"))
    return f"{parse_text}；{action_text}；主要错误来源为{error_text}；平均推理时延为 {latency}s。"


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exp20 复杂/噪声/长文本指令鲁棒性分析",
        "",
        "本实验按输入类型组织同一最终微调推理方案的表现，用于回应“复杂指令、噪声指令、长文本指令是否会带来额外问题”的意见。",
        "",
        "## 指标总表",
        "",
        "| 输入类型 | 样本数 | JSON 可解析率 | 相对标准下降 | 动作匹配率 | 相对标准下降 | 平均时延(s) | 主要错误来源 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["category_rows"]:
        lines.append(
            "| {label} | {total} | {parse_ok} | {parse_drop} | {action_match} | {action_drop} | {latency} | {error} |".format(
                label=row["category_label"],
                total=row["total"],
                parse_ok=_fmt(row["parse_ok_rate"]),
                parse_drop=_fmt(row.get("parse_drop_vs_standard")),
                action_match=_fmt(row["action_match_rate"]),
                action_drop=_fmt(row.get("action_drop_vs_standard")),
                latency=_fmt(row.get("avg_latency_sec")),
                error=_dominant_error(row),
            )
        )

    lines.extend(
        [
            "",
            "## 按输入类型分析",
            "",
        ]
    )
    for row in summary["category_rows"]:
        lines.extend(
            [
                f"### {row['category_label']}",
                _interpret_category(row),
                (
                    "该类重点观察：JSON 可解析率是否下降、动作匹配率是否下降、错误主要来自语义理解/"
                    "动作顺序/参数映射中的哪一类，以及时延是否随输入长度增加。"
                ),
                "",
            ]
        )

    length_latency = summary.get("length_latency", {})
    lines.extend(
        [
            "## 长度与时延",
            "",
            (
                f"有效时延样本数为 {length_latency.get('num_latency_points', 0)}，"
                f"字符长度与时延 Pearson 相关系数为 {_fmt(length_latency.get('input_chars_latency_pearson'))}，"
                f"估算 token 数与时延 Pearson 相关系数为 {_fmt(length_latency.get('estimated_tokens_latency_pearson'))}。"
            ),
            "",
            "若长文本类平均时延显著高于标准指令，且相关系数为正，可在论文中表述为：最终方案的结构化输出能力主要保持稳定，但推理时延会随输入长度增加。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_build_only_md(cases: list[StressCase], path: Path) -> None:
    category_counts = Counter(case.category for case in cases)
    avg_chars = {
        category: _mean([float(case.input_chars) for case in cases if case.category == category])
        for category in CATEGORY_ORDER
    }
    lines = [
        "# Exp20 复杂/噪声/长文本指令鲁棒性实验设计",
        "",
        "已生成 stress dataset，尚未运行模型推理。本文件说明论文中可新增的实验口径。",
        "",
        "| 输入类型 | 样本数 | 设计目的 | 平均输入长度(字符) |",
        "| --- | ---: | --- | ---: |",
    ]
    purpose = {
        "standard": "作为对照组，观察常规指令下 JSON 可解析率、动作匹配率和基础时延。",
        "complex_multistep": "检验多动作序列中是否出现动作遗漏、动作调换或步骤合并。",
        "noise_redundant": "检验模型能否忽略背景、闲聊、重复说明等非动作信息。",
        "long_context": "检验长输入下核心指令定位能力和推理时延增长。",
        "boundary_ambiguous": "检验含默认值、边界描述、模糊词时是否发生参数发明或语义误解。",
    }
    for category in CATEGORY_ORDER:
        lines.append(
            f"| {CATEGORY_LABELS[category]} | {category_counts[category]} | {purpose[category]} | {_fmt(avg_chars[category], 1)} |"
        )
    lines.extend(
        [
            "",
            "后续运行 `--mode local` 可得到每类 JSON 可解析率、动作匹配率、错误来源和长度-时延相关性；运行 `--mode predictions` 可用已保存预测离线复算。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(summary: dict[str, Any], details: list[dict[str, Any]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary, SUMMARY_CSV_PATH)
    write_summary_md(summary, SUMMARY_MD_PATH)
    write_details_jsonl(details, DETAILS_JSONL_PATH)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)

    mode = args.mode or str(cfg_get(config, "mode", "build-only"))
    dataset_file = repo_path(args.dataset_file or cfg_get(config, "dataset_file", "data_prepare/splits/test.json"))
    per_category = int(args.per_category or cfg_get(config, "per_category", 20))
    seed = int(args.seed if args.seed is not None else cfg_get(config, "seed", 42))
    long_target_chars = int(cfg_get(config, "long_context.target_chars", 1400))

    rows = load_source_rows(dataset_file)
    cases = build_stress_cases(rows, per_category=per_category, seed=seed, long_target_chars=long_target_chars)
    write_stress_dataset(cases, STRESS_DATASET_PATH)
    print(f"[ok] stress dataset: {STRESS_DATASET_PATH} ({len(cases)} cases)")

    if mode == "build-only":
        write_build_only_md(cases, SUMMARY_MD_PATH)
        print(f"[ok] design report : {SUMMARY_MD_PATH}")
        return

    if mode == "predictions":
        predictions_file = args.predictions_file or cfg_get(config, "predictions_file")
        if not predictions_file:
            raise ValueError("--predictions-file is required in predictions mode")
        details = run_predictions_eval(cases, repo_path(predictions_file))
    elif mode == "local":
        details = run_local_eval(cases, config, args)
    else:
        raise ValueError(f"unknown mode: {mode}")

    summary = summarize_details(details, mode=mode, cases=cases)
    write_outputs(summary, details)
    print(f"[ok] summary json   : {SUMMARY_JSON_PATH}")
    print(f"[ok] summary csv    : {SUMMARY_CSV_PATH}")
    print(f"[ok] summary md     : {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
