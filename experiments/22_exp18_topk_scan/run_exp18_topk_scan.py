#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = EXPERIMENT_DIR / "configs"
LOGS_DIR = EXPERIMENT_DIR / "logs"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
TEMP_DIR = EXPERIMENT_DIR / ".cache"

BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TRAIN_RUNNER = REPO_ROOT / "experiments" / "02_finetune_exp" / "run_train.py"
EVAL_RUNNER = REPO_ROOT / "experiments" / "03_eval_exp" / "run_accuracy.py"
SCORING_RUNNER = REPO_ROOT / "experiments" / "13_exp11" / "run_layer_scoring.py"
RANK8_TEMPLATE_CONFIG_PATH = REPO_ROOT / "experiments" / "13_exp11" / "configs" / "train_rank8_subset_template.yaml"

DEFAULT_BASELINE_MODEL_PATH = REPO_ROOT / "model" / "qwen2.5-3b-genesis-lora-rank-4"
DEFAULT_LAYER_MODEL_PATH = REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct"
DEFAULT_LAYER_DATA_PATH = REPO_ROOT / "data_prepare" / "splits" / "train.json"
TRAIN_SUBSET_SIZE = 600
DEFAULT_K_VALUES = [4, 8, 12, 18, 24, 28]
DEFAULT_LORA_ALPHA = 32

LORA_MODULE_PATHS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
ALL_LORA_MODULE_TYPES = tuple(LORA_MODULE_PATHS.keys())

TRAIN_SUBSET_PATH = TEMP_DIR / f"train_subset_{TRAIN_SUBSET_SIZE}.json"
LAYER_SCORE_REPORT_PATH = REPORTS_DIR / "layer_scores.json"
BASELINE_REPORT_PATH = REPORTS_DIR / "accuracy_report_full_rank4.json"
BASELINE_EVAL_OVERRIDE_PATH = TEMP_DIR / "eval_full_rank4_override.yaml"
SUMMARY_JSON_PATH = REPORTS_DIR / "exp18_topk_summary.json"
SUMMARY_CSV_PATH = REPORTS_DIR / "exp18_topk_summary.csv"
SUMMARY_MD_PATH = REPORTS_DIR / "exp18_topk_summary.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验 22 / Exp18：按重要性 Top-K 层做 rank8 微调扫描。")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG_PATH, help="全局基础配置。")
    parser.add_argument("--baseline-model-path", type=Path, default=DEFAULT_BASELINE_MODEL_PATH, help="全层 rank4 baseline 模型，仅做评测。")
    parser.add_argument("--layer-model-path", type=Path, default=DEFAULT_LAYER_MODEL_PATH, help="层打分使用的基础模型路径。")
    parser.add_argument("--layer-data-path", type=Path, default=DEFAULT_LAYER_DATA_PATH, help="层打分数据集；若保持默认，则自动替换为固定训练子集。")
    parser.add_argument("--layer-sample-size", type=int, default=100, help="层打分采样条数。")
    parser.add_argument("--k-values", type=str, default=",".join(str(item) for item in DEFAULT_K_VALUES), help="扫描的 K 列表，例如 4,8,12,18,24。")
    parser.add_argument("--rank8-module-types", type=str, default=",".join(ALL_LORA_MODULE_TYPES), help="rank8 分支统一使用的模块类型白名单。")
    parser.add_argument("--gpus", type=str, default="0", help="传给训练脚本的 GPU 编号字符串。")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练阶段，只生成配置并可直接评测现有模型目录。")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评测阶段。")
    parser.add_argument("--skip-baseline-eval", action="store_true", help="跳过 baseline 评测。")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不真正执行。")
    return parser.parse_args(argv)


def ensure_dirs() -> None:
    for path in (LOGS_DIR, REPORTS_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 根节点必须是字典: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"JSON 根节点必须是数组: {path}")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"JSON 数组元素必须全部是对象: {path}")
    return payload


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_k_values(raw: str | list[int] | tuple[int, ...]) -> list[int]:
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    values: list[int] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        try:
            value = int(text)
        except Exception:
            continue
        if value <= 0:
            continue
        if value not in values:
            values.append(value)
    return sorted(values)


def run_stage(
    stage_name: str,
    command: list[str],
    log_path: Path,
    *,
    cwd: Path = REPO_ROOT,
    dry_run: bool = False,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_str = " ".join(command)
    print(f"[{stage_name}] 开始执行", flush=True)
    print(f"[{stage_name}] 命令: {command_str}", flush=True)
    print(f"[{stage_name}] 日志: {log_path}", flush=True)

    if dry_run:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n===== {timestamp()} | {stage_name} | DRY RUN =====\n")
            f.write(f"cwd: {cwd}\n")
            f.write(f"command: {command_str}\n")
        return

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {timestamp()} | {stage_name} =====\n")
        f.write(f"cwd: {cwd}\n")
        f.write(f"command: {command_str}\n\n")
        f.flush()

        process = subprocess.Popen(
            command,
            cwd=cwd,
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
            raise RuntimeError(f"阶段 {stage_name} 执行失败，退出码 {return_code}。请查看日志: {log_path}")


def resolve_train_file(base_config_path: Path) -> Path:
    config = load_yaml(base_config_path)
    train_section = config.get("finetune", {}).get("train", {})
    train_file = train_section.get("train_file")
    if not train_file:
        return DEFAULT_LAYER_DATA_PATH

    resolved = Path(str(train_file))
    if not resolved.is_absolute():
        repo_candidate = REPO_ROOT / resolved
        if repo_candidate.exists():
            resolved = repo_candidate
        else:
            resolved = base_config_path.resolve().parent / resolved
    return resolved.resolve()


def prepare_train_subset(base_config_path: Path, subset_size: int = TRAIN_SUBSET_SIZE) -> Path:
    source_train_path = resolve_train_file(base_config_path)
    if not source_train_path.exists():
        raise FileNotFoundError(f"训练集文件不存在: {source_train_path}")

    rows = load_json_list(source_train_path)
    if len(rows) < subset_size:
        raise ValueError(f"训练集总条数只有 {len(rows)}，无法截取固定 {subset_size} 条样本。")

    subset = rows[:subset_size]
    write_json(TRAIN_SUBSET_PATH, subset)
    print(f"[exp18] 已生成固定训练子集: {TRAIN_SUBSET_PATH} (source={source_train_path}, size={subset_size}/{len(rows)})", flush=True)
    return TRAIN_SUBSET_PATH


def resolve_existing_model_path(model_path: Path) -> Path:
    resolved = model_path.expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"模型不存在: {resolved}")
    return resolved


def infer_num_hidden_layers(model_path: Path) -> int | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _extract_layer_index(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.startswith("layer_"):
        return int(text.split("_", 1)[1])
    return int(text)


def load_layer_ranking(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"层打分报告缺少 ranking 或 ranking 为空: {path}")
    return [item for item in ranking if isinstance(item, dict)]


def normalize_lora_module_types(module_types: str | list[str] | tuple[str, ...]) -> list[str]:
    raw_items = module_types.split(",") if isinstance(module_types, str) else list(module_types)
    normalized: list[str] = []
    unknown: list[str] = []
    for raw in raw_items:
        module_type = str(raw).strip()
        if not module_type:
            continue
        if module_type not in LORA_MODULE_PATHS:
            unknown.append(module_type)
            continue
        if module_type not in normalized:
            normalized.append(module_type)
    if unknown:
        supported = ", ".join(LORA_MODULE_PATHS.keys())
        raise ValueError(f"存在不支持的模块类型: {', '.join(unknown)}；当前仅支持: {supported}")
    if not normalized:
        raise ValueError("至少需要一个有效模块类型。")
    return normalized


def build_lora_target_modules(layer_indices: list[int], module_types: list[str]) -> list[str]:
    targets: list[str] = []
    for layer_idx in sorted(set(layer_indices)):
        for module_type in module_types:
            targets.append(f"model.layers.{layer_idx}.{LORA_MODULE_PATHS[module_type]}")
    return targets


def select_top_layers_by_score(ranking: list[dict[str, Any]], selected_count: int) -> list[int]:
    if selected_count <= 0:
        raise ValueError("selected_count 必须大于 0。")
    return sorted({_extract_layer_index(item["layer"]) for item in ranking[:selected_count]})


def preview_top_layers(model_path: Path, selected_count: int) -> list[int]:
    layer_count = infer_num_hidden_layers(model_path)
    if layer_count is None:
        layer_count = 32
        print("[exp18] dry-run 无法解析层数，按 32 层模型生成占位选择。", flush=True)
    actual_count = min(selected_count, layer_count)
    return list(range(layer_count - actual_count, layer_count))


def build_output_dir(case_name: str) -> Path:
    return REPO_ROOT / "output" / "exp18_topk_scan" / case_name


def build_generated_config_path(case_name: str) -> Path:
    return TEMP_DIR / f"train_{case_name}_generated.yaml"


def build_train_override_path(case_name: str) -> Path:
    return TEMP_DIR / f"train_{case_name}_override.yaml"


def build_eval_override_path(case_name: str) -> Path:
    return TEMP_DIR / f"eval_{case_name}_override.yaml"


def build_eval_report_path(case_name: str) -> Path:
    return REPORTS_DIR / f"accuracy_report_{case_name}.json"


def build_selection_report_path(case_name: str) -> Path:
    return REPORTS_DIR / f"{case_name}_layers.json"


def build_case_specs(
    *,
    k_values: list[int],
    ranking: list[dict[str, Any]] | None,
    layer_model_path: Path,
    module_types: list[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for k in k_values:
        case_name = f"top{k}_rank8"
        if ranking is None:
            layers = preview_top_layers(layer_model_path, k)
            selection_label = f"按打分选 Top-{k} 层（dry-run 占位）"
        else:
            layers = select_top_layers_by_score(ranking, k)
            selection_label = f"按重要性打分选 Top-{k} 层"
        specs.append(
            {
                "name": case_name,
                "k": int(k),
                "selection_label": selection_label,
                "layers": layers,
                "module_types": list(module_types),
                "output_dir": build_output_dir(case_name),
                "generated_config_path": build_generated_config_path(case_name),
                "train_override_path": build_train_override_path(case_name),
                "eval_override_path": build_eval_override_path(case_name),
                "eval_report_path": build_eval_report_path(case_name),
                "selection_report_path": build_selection_report_path(case_name),
            }
        )
    return specs


def prepare_rank8_config(*, case: dict[str, Any]) -> list[str]:
    config = load_yaml(RANK8_TEMPLATE_CONFIG_PATH)
    target_modules = build_lora_target_modules(case["layers"], case["module_types"])
    config["lora_rank"] = 8
    config["lora_alpha"] = DEFAULT_LORA_ALPHA
    config["lora_target"] = ",".join(target_modules)
    config["output_dir"] = repo_rel(case["output_dir"])
    write_yaml(case["generated_config_path"], config)
    return target_modules


def prepare_train_override(config_path: Path, override_path: Path, gpus: str, *, train_file: Path) -> Path:
    override = {
        "finetune": {
            "train": {
                "config": repo_rel(config_path),
                "gpus": gpus,
                "dry_run": False,
                "finetune_method": "lora",
                "train_file": repo_rel(train_file),
            }
        }
    }
    write_yaml(override_path, override)
    return override_path


def prepare_eval_override(model_path: Path, report_path: Path, override_path: Path) -> Path:
    override = {
        "test": {
            "accuracy_eval": {
                "mode": "local",
                "report_file": repo_rel(report_path),
                "model_path": repo_rel(model_path),
                "backend": "transformers",
                "temperature": 0.0,
                "trust_remote_code": True,
            }
        }
    }
    write_yaml(override_path, override)
    return override_path


def write_selection_report(
    *,
    case: dict[str, Any],
    target_modules: list[str],
    train_subset_path: Path,
    baseline_model_path: Path,
    score_report_path: Path | None,
    preview_only: bool,
) -> None:
    payload = {
        "experiment": "exp18_topk_scan",
        "variant_name": case["name"],
        "selection_label": case["selection_label"],
        "baseline_model_path": repo_rel(baseline_model_path),
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": TRAIN_SUBSET_SIZE,
        },
        "selection_strategy": {
            "selected_layer_count": int(case["k"]),
            "score_based": True,
            "preview_only": preview_only,
        },
        "layer_score_report": repo_rel(score_report_path) if score_report_path is not None else None,
        "target_layers": case["layers"],
        "num_target_layers": len(case["layers"]),
        "module_types": case["module_types"],
        "lora_target_modules": target_modules,
        "output_dir": repo_rel(case["output_dir"]),
    }
    write_json(case["selection_report_path"], payload)


def extract_eval_metrics(report_path: Path) -> dict[str, float | int]:
    report = load_json(report_path)
    return {
        "num_samples_evaluated": int(report.get("num_samples_evaluated", 0)),
        "parse_ok_rate": float(report.get("parse_ok_rate", 0.0)),
        "exact_match_rate": float(report.get("exact_match_rate", 0.0)),
        "action_match_rate": float(report.get("action_match_rate", 0.0)),
        "avg_latency_sec": float(report.get("avg_latency_sec", 0.0)),
        "avg_throughput_tps": float(report.get("avg_throughput_tps", 0.0)),
        "avg_peak_vram_mb": float(report.get("avg_peak_vram_mb", 0.0)),
        "max_peak_vram_mb": float(report.get("max_peak_vram_mb", 0.0)),
    }


def summarize_cases(*, baseline_model_path: Path, train_subset_path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        "name": "full_rank4",
        "model_path": repo_rel(baseline_model_path),
        "report_path": repo_rel(BASELINE_REPORT_PATH) if BASELINE_REPORT_PATH.exists() else None,
    }
    if BASELINE_REPORT_PATH.exists():
        baseline.update(extract_eval_metrics(BASELINE_REPORT_PATH))

    rows: list[dict[str, Any]] = []
    for case in cases:
        row = {
            "name": case["name"],
            "k": int(case["k"]),
            "selection_label": case["selection_label"],
            "report_path": repo_rel(case["eval_report_path"]) if case["eval_report_path"].exists() else None,
            "output_dir": repo_rel(case["output_dir"]),
            "target_layers": case["layers"],
            "num_target_layers": len(case["layers"]),
            "module_types": case["module_types"],
        }
        if case["eval_report_path"].exists():
            row.update(extract_eval_metrics(case["eval_report_path"]))
        rows.append(row)

    def _best(metric: str, *, reverse: bool) -> dict[str, Any] | None:
        candidates = [row for row in rows if metric in row]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (float(item[metric]), -int(item["k"])), reverse=reverse)[0]

    return {
        "experiment": "exp18_topk_scan",
        "question": "按重要性排名选前 K 层做 rank8 微调时，K 如何影响任务性能与推理成本。",
        "baseline": baseline,
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": TRAIN_SUBSET_SIZE,
        },
        "cases": rows,
        "best_exact_match": _best("exact_match_rate", reverse=True),
        "best_action_match": _best("action_match_rate", reverse=True),
        "best_parse_ok": _best("parse_ok_rate", reverse=True),
        "best_latency": _best("avg_latency_sec", reverse=False),
        "best_throughput": _best("avg_throughput_tps", reverse=True),
    }


def write_summary_csv(summary: dict[str, Any], output_path: Path) -> None:
    rows = summary.get("cases", [])
    fieldnames = [
        "name",
        "k",
        "num_target_layers",
        "parse_ok_rate",
        "exact_match_rate",
        "action_match_rate",
        "avg_latency_sec",
        "avg_throughput_tps",
        "avg_peak_vram_mb",
        "max_peak_vram_mb",
        "report_path",
        "output_dir",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    baseline = summary.get("baseline", {})
    rows = summary.get("cases", [])
    lines = [
        "# Exp18 Top-K 重要层扫描实验",
        "",
        f"- 实验问题：{summary.get('question', '')}",
        f"- baseline：`{baseline.get('model_path')}`",
        f"- 固定训练集：`{summary.get('train_subset', {}).get('size', 0)}` 条（`{summary.get('train_subset', {}).get('path')}`）",
        "",
    ]
    if baseline.get("report_path") is not None:
        lines.extend(
            [
                "## Baseline",
                "",
                f"- `parse_ok_rate`: {float(baseline.get('parse_ok_rate', 0.0)):.4f}",
                f"- `exact_match_rate`: {float(baseline.get('exact_match_rate', 0.0)):.4f}",
                f"- `action_match_rate`: {float(baseline.get('action_match_rate', 0.0)):.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## K 扫描结果",
            "",
            "| K | Parse OK | Exact Match | Action Match | Avg Latency (s) | Avg Throughput (tokens/s) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if "parse_ok_rate" not in row:
            continue
        lines.append(
            "| "
            f"{int(row['k'])} | "
            f"{float(row['parse_ok_rate']):.4f} | "
            f"{float(row['exact_match_rate']):.4f} | "
            f"{float(row['action_match_rate']):.4f} | "
            f"{float(row['avg_latency_sec']):.4f} | "
            f"{float(row['avg_throughput_tps']):.4f} |"
        )

    for title, key in (
        ("最佳 Exact Match", "best_exact_match"),
        ("最佳 Action Match", "best_action_match"),
        ("最佳 Parse OK", "best_parse_ok"),
        ("最低时延", "best_latency"),
        ("最高吞吐", "best_throughput"),
    ):
        item = summary.get(key)
        if not isinstance(item, dict):
            continue
        lines.extend(["", f"## {title}", "", f"- K：`{int(item['k'])}`", f"- case：`{item['name']}`"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dirs()

    k_values = parse_k_values(args.k_values)
    if not k_values:
        raise ValueError("至少需要一个有效 K 值。")

    baseline_model_path = resolve_existing_model_path(args.baseline_model_path)
    train_subset_path = prepare_train_subset(args.base_config, TRAIN_SUBSET_SIZE)
    layer_data_path = args.layer_data_path.resolve()
    if layer_data_path == DEFAULT_LAYER_DATA_PATH.resolve():
        layer_data_path = train_subset_path
    print(
        "[exp18] "
        f"baseline 模型: {baseline_model_path}，固定训练集大小: {TRAIN_SUBSET_SIZE}，"
        f"K 扫描: {k_values}，层打分数据集: {layer_data_path}",
        flush=True,
    )

    run_stage(
        "layer_scoring",
        [
            sys.executable,
            str(SCORING_RUNNER),
            "--model-path",
            str(args.layer_model_path),
            "--data-path",
            str(layer_data_path),
            "--output-path",
            str(LAYER_SCORE_REPORT_PATH),
            "--sample-size",
            str(args.layer_sample_size),
        ],
        LOGS_DIR / "layer_scoring.log",
        dry_run=args.dry_run,
    )

    ranking: list[dict[str, Any]] | None = None
    score_report_path: Path | None = None
    if args.dry_run:
        if LAYER_SCORE_REPORT_PATH.exists():
            ranking = load_layer_ranking(LAYER_SCORE_REPORT_PATH)
            score_report_path = LAYER_SCORE_REPORT_PATH
    else:
        ranking = load_layer_ranking(LAYER_SCORE_REPORT_PATH)
        score_report_path = LAYER_SCORE_REPORT_PATH

    module_types = normalize_lora_module_types(args.rank8_module_types)
    cases = build_case_specs(
        k_values=k_values,
        ranking=ranking,
        layer_model_path=args.layer_model_path,
        module_types=module_types,
    )

    for case in cases:
        target_modules = prepare_rank8_config(case=case)
        prepare_train_override(
            case["generated_config_path"],
            case["train_override_path"],
            args.gpus,
            train_file=train_subset_path,
        )
        write_selection_report(
            case=case,
            target_modules=target_modules,
            train_subset_path=train_subset_path,
            baseline_model_path=baseline_model_path,
            score_report_path=score_report_path,
            preview_only=args.dry_run,
        )
        print(f"[exp18] 已生成 case 配置: {case['generated_config_path']}", flush=True)

    if not args.skip_train:
        for case in cases:
            run_stage(
                f"train_{case['name']}",
                [
                    sys.executable,
                    str(TRAIN_RUNNER),
                    "--base-config",
                    str(args.base_config),
                    "--config",
                    str(case["train_override_path"]),
                    *(["--dry-run"] if args.dry_run else []),
                    "output_dir=" + str(case["output_dir"]),
                ],
                LOGS_DIR / f"train_{case['name']}.log",
                dry_run=args.dry_run,
            )
    else:
        print("[exp18] 已跳过训练阶段。", flush=True)

    if args.skip_eval:
        print("[exp18] 已跳过评测阶段。", flush=True)
        return

    if not args.skip_baseline_eval:
        prepare_eval_override(
            model_path=baseline_model_path,
            report_path=BASELINE_REPORT_PATH,
            override_path=BASELINE_EVAL_OVERRIDE_PATH,
        )
        run_stage(
            "eval_full_rank4",
            [
                sys.executable,
                str(EVAL_RUNNER),
                "--base-config",
                str(args.base_config),
                "--config",
                str(BASELINE_EVAL_OVERRIDE_PATH),
            ],
            LOGS_DIR / "eval_full_rank4.log",
            dry_run=args.dry_run,
        )

    for case in cases:
        prepare_eval_override(
            model_path=case["output_dir"],
            report_path=case["eval_report_path"],
            override_path=case["eval_override_path"],
        )
        run_stage(
            f"eval_{case['name']}",
            [
                sys.executable,
                str(EVAL_RUNNER),
                "--base-config",
                str(args.base_config),
                "--config",
                str(case["eval_override_path"]),
            ],
            LOGS_DIR / f"eval_{case['name']}.log",
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[exp18] dry-run 已完成，真实运行后会生成 summary。", flush=True)
        return

    summary = summarize_cases(
        baseline_model_path=baseline_model_path,
        train_subset_path=train_subset_path,
        cases=cases,
    )
    write_json(SUMMARY_JSON_PATH, summary)
    write_summary_csv(summary, SUMMARY_CSV_PATH)
    write_summary_markdown(summary, SUMMARY_MD_PATH)
    print(f"[exp18] summary json : {SUMMARY_JSON_PATH}", flush=True)
    print(f"[exp18] summary csv  : {SUMMARY_CSV_PATH}", flush=True)
    print(f"[exp18] summary md   : {SUMMARY_MD_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[exp18] 失败: {exc}", file=sys.stderr)
        raise
