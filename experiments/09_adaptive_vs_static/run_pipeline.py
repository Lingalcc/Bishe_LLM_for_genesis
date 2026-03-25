#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
SCORING_RUNNER = EXPERIMENT_DIR / "run_layer_scoring.py"

FULL_RANK4_CONFIG_PATH = CONFIG_DIR / "train_full_rank4.yaml"
IMPORTANT_TEMPLATE_CONFIG_PATH = CONFIG_DIR / "train_important_rank4_template.yaml"
GENERATED_IMPORTANT_CONFIG_PATH = TEMP_DIR / "train_important_rank4_generated.yaml"
HIGH18_RANK8_TEMPLATE_CONFIG_PATH = CONFIG_DIR / "train_high18_rank8_template.yaml"
GENERATED_HIGH18_RANK8_CONFIG_PATH = TEMP_DIR / "train_high18_rank8_generated.yaml"
GENERATED_HIGH18_RANK8_2D_CONFIG_PATH = TEMP_DIR / "train_high18_rank8_2d_generated.yaml"

FULL_RANK4_OUTPUT_DIR = REPO_ROOT / "output" / "exp11_exp7_adarank" / "full_rank4"
IMPORTANT_RANK4_OUTPUT_DIR = REPO_ROOT / "output" / "exp11_exp7_adarank" / "important_rank4"
HIGH18_RANK8_OUTPUT_DIR = REPO_ROOT / "output" / "exp11_exp7_adarank" / "high18_rank8"
HIGH18_RANK8_2D_OUTPUT_DIR = REPO_ROOT / "output" / "exp11_exp7_adarank" / "high18_rank8_2d"
DEFAULT_BASELINE_MODEL_PATH = REPO_ROOT / "model" / "qwen2.5-3b-genesis-lora-rank-4"

FULL_RANK4_TRAIN_OVERRIDE = TEMP_DIR / "train_full_rank4_override.yaml"
IMPORTANT_RANK4_TRAIN_OVERRIDE = TEMP_DIR / "train_important_rank4_override.yaml"
HIGH18_RANK8_TRAIN_OVERRIDE = TEMP_DIR / "train_high18_rank8_override.yaml"
HIGH18_RANK8_2D_TRAIN_OVERRIDE = TEMP_DIR / "train_high18_rank8_2d_override.yaml"
FULL_RANK4_EVAL_OVERRIDE = TEMP_DIR / "eval_full_rank4_override.yaml"
IMPORTANT_RANK4_EVAL_OVERRIDE = TEMP_DIR / "eval_important_rank4_override.yaml"
HIGH18_RANK8_EVAL_OVERRIDE = TEMP_DIR / "eval_high18_rank8_override.yaml"
HIGH18_RANK8_2D_EVAL_OVERRIDE = TEMP_DIR / "eval_high18_rank8_2d_override.yaml"

LAYER_SCORE_REPORT_PATH = REPORTS_DIR / "layer_scores.json"
IMPORTANT_LAYER_SELECTION_PATH = REPORTS_DIR / "important_layers.json"
HIGH18_LAYER_SELECTION_PATH = REPORTS_DIR / "high18_rank8_layers.json"
HIGH18_2D_LAYER_SELECTION_PATH = REPORTS_DIR / "high18_rank8_2d_layers.json"
COMPARISON_SUMMARY_JSON_PATH = REPORTS_DIR / "comparison_summary.json"
COMPARISON_SUMMARY_MD_PATH = REPORTS_DIR / "comparison_summary.md"
COMPARISON_SUMMARY_2D_JSON_PATH = REPORTS_DIR / "comparison_summary_2d.json"
COMPARISON_SUMMARY_2D_MD_PATH = REPORTS_DIR / "comparison_summary_2d.md"

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
DEFAULT_TWO_DIMENSIONAL_MODULE_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
)
DEFAULT_LAYER_DATA_PATH = REPO_ROOT / "data_prepare" / "splits" / "train.json"
TRAIN_SUBSET_SIZE = 600
TRAIN_SUBSET_PATH = TEMP_DIR / f"train_subset_{TRAIN_SUBSET_SIZE}.json"
DEFAULT_LORA_ALPHA = 32
DEFAULT_HIGH_LAYER_COUNT = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "实验 11_exp7_adarank：固定 600 条训练样本，比较全层 LoRA rank4、"
            "打分后重要层 LoRA rank4，以及基于 layer_scores Top-18 层的 LoRA rank8。"
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=BASE_CONFIG_PATH,
        help="全局基础配置文件。",
    )
    parser.add_argument(
        "--baseline-model-path",
        type=Path,
        default=DEFAULT_BASELINE_MODEL_PATH,
        help="现成的全层 rank4 LoRA 模型路径，用作 baseline 评测，不再重复训练。",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="传给训练脚本的 GPU 编号字符串，例如 0 或 0,1。",
    )
    parser.add_argument(
        "--layer-model-path",
        type=Path,
        default=REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
        help="层打分阶段使用的基础模型路径。",
    )
    parser.add_argument(
        "--layer-data-path",
        type=Path,
        default=DEFAULT_LAYER_DATA_PATH,
        help="层打分阶段使用的数据集路径。",
    )
    parser.add_argument(
        "--layer-sample-size",
        type=int,
        default=100,
        help="层打分阶段采样条数。",
    )
    parser.add_argument(
        "--important-top-k",
        type=int,
        default=None,
        help="直接选择 top-k 个重要层；若不传，则使用 important-ratio。",
    )
    parser.add_argument(
        "--important-ratio",
        type=float,
        default=0.5,
        help="当未指定 important-top-k 时，按比例选择重要层，默认 0.5。",
    )
    parser.add_argument(
        "--high-layer-count",
        type=int,
        default=DEFAULT_HIGH_LAYER_COUNT,
        help="第三条策略固定选择层号最高的多少层，默认 18。",
    )
    parser.add_argument(
        "--important-module-types",
        type=str,
        default=",".join(ALL_LORA_MODULE_TYPES),
        help=(
            "important_rank4 分支允许注入 LoRA 的模块类型，逗号分隔。"
            "默认保持原实验行为：q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj。"
        ),
    )
    parser.add_argument(
        "--high-layer-module-types",
        type=str,
        default=",".join(ALL_LORA_MODULE_TYPES),
        help=(
            "high18_rank8 分支允许注入 LoRA 的模块类型，逗号分隔。"
            "默认保持原实验行为：q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj。"
        ),
    )
    parser.add_argument(
        "--enable-high18-rank8-2d",
        action="store_true",
        help="额外新增一个 high18_rank8_2d 分支，采用“筛层 + 筛模块”的二维过滤，不覆盖原 high18_rank8。",
    )
    parser.add_argument(
        "--high-layer-2d-module-types",
        type=str,
        default=",".join(DEFAULT_TWO_DIMENSIONAL_MODULE_TYPES),
        help=(
            "high18_rank8_2d 分支的模块白名单，逗号分隔。"
            "默认 q_proj,k_proj,v_proj,o_proj,down_proj。"
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="只完成训练与层打分，不执行最终评测。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将执行的命令，不真正运行。",
    )
    return parser.parse_args()


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
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点必须是字典: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return data


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON 根节点必须是数组: {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"JSON 数组元素必须全部为对象: {path}")
    return data


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
            raise RuntimeError(
                f"阶段 {stage_name} 执行失败，退出码 {return_code}。请查看日志: {log_path}"
            )


def resolve_train_file(base_config_path: Path) -> Path:
    config = load_yaml(base_config_path)
    train_section = config.get("finetune", {}).get("train", {})
    train_file = train_section.get("train_file")
    if not train_file:
        return DEFAULT_LAYER_DATA_PATH

    resolved = Path(train_file)
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
    total_rows = len(rows)
    if total_rows < subset_size:
        raise ValueError(
            f"训练集总条数只有 {total_rows}，无法截取固定 {subset_size} 条样本。"
        )

    subset = rows[:subset_size]
    write_json(TRAIN_SUBSET_PATH, subset)
    print(
        f"[pipeline] 已生成固定训练子集: {TRAIN_SUBSET_PATH} "
        f"(source={source_train_path}, size={subset_size}/{total_rows})",
        flush=True,
    )
    return TRAIN_SUBSET_PATH


def resolve_existing_model_path(model_path: Path) -> Path:
    resolved = model_path.expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"baseline 模型不存在: {resolved}")
    return resolved


def prepare_train_override(
    config_path: Path,
    override_path: Path,
    gpus: str,
    *,
    train_file: Path | None = None,
) -> Path:
    train_section: dict[str, Any] = {
        "config": repo_rel(config_path),
        "gpus": gpus,
        "dry_run": False,
        "finetune_method": "lora",
    }
    if train_file is not None:
        train_section["train_file"] = repo_rel(train_file)

    override = {
        "finetune": {
            "train": train_section
        }
    }
    write_yaml(override_path, override)
    return override_path


def prepare_eval_override(
    model_path: Path,
    report_path: Path,
    override_path: Path,
) -> Path:
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
    if isinstance(raw, str):
        if raw.startswith("layer_"):
            suffix = raw.split("_", 1)[1]
            return int(suffix)
        return int(raw)
    raise ValueError(f"无法解析层编号: {raw!r}")


def load_layer_ranking(report_path: Path) -> list[dict[str, Any]]:
    payload = load_json(report_path)
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"层打分报告缺少 ranking 或 ranking 为空: {report_path}")
    return [item for item in ranking if isinstance(item, dict)]


def select_important_layers(
    ranking: list[dict[str, Any]],
    *,
    important_top_k: int | None,
    important_ratio: float,
) -> list[int]:
    if not ranking:
        raise ValueError("ranking 为空，无法选择重要层。")

    total_layers = len(ranking)
    if important_top_k is not None:
        if important_top_k <= 0:
            raise ValueError("important_top_k 必须大于 0。")
        selected_count = min(important_top_k, total_layers)
        strategy = f"top_k={important_top_k}"
    else:
        if not (0.0 < important_ratio <= 1.0):
            raise ValueError("important_ratio 必须在 (0, 1] 区间内。")
        selected_count = max(1, math.ceil(total_layers * important_ratio))
        strategy = f"ratio={important_ratio:.4f}"

    selected = [_extract_layer_index(item["layer"]) for item in ranking[:selected_count]]
    selected = sorted(set(selected))
    print(f"[pipeline] 重要层选择策略: {strategy}，共选中 {len(selected)}/{total_layers} 层。", flush=True)
    return selected


def preview_important_layers(
    model_path: Path,
    *,
    important_top_k: int | None,
    important_ratio: float,
) -> list[int]:
    layer_count = infer_num_hidden_layers(model_path)
    if layer_count is None:
        layer_count = 32
        print(
            "[pipeline] dry-run 未发现现成 layer_scores.json，且无法从模型配置推断层数；"
            "将按 32 层模型生成预览配置。",
            flush=True,
        )

    if important_top_k is not None:
        selected_count = min(max(1, important_top_k), layer_count)
    else:
        if not (0.0 < important_ratio <= 1.0):
            raise ValueError("important_ratio 必须在 (0, 1] 区间内。")
        selected_count = max(1, math.ceil(layer_count * important_ratio))

    selected = list(range(layer_count - selected_count, layer_count))
    print(
        f"[pipeline] dry-run 预览重要层: 使用最高层 {selected_count} 层作为占位选择 {selected}",
        flush=True,
    )
    return selected


def build_lora_target_modules(layer_indices: list[int]) -> list[str]:
    return build_lora_target_modules_for_types(layer_indices, ALL_LORA_MODULE_TYPES)


def normalize_lora_module_types(module_types: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(module_types, str):
        raw_items = module_types.split(",")
    else:
        raw_items = list(module_types)

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
        raise ValueError(
            "存在不支持的模块类型: "
            f"{', '.join(unknown)}；当前仅支持: {supported}"
        )
    if not normalized:
        raise ValueError("至少需要一个有效模块类型，才能构造 lora_target。")
    return normalized


def build_lora_target_modules_for_types(
    layer_indices: list[int],
    module_types: str | list[str] | tuple[str, ...],
) -> list[str]:
    if not layer_indices:
        raise ValueError("至少需要一个重要层，才能构造 lora_target。")

    normalized_module_types = normalize_lora_module_types(module_types)
    target_modules: list[str] = []
    for layer_idx in sorted(set(layer_indices)):
        for module_type in normalized_module_types:
            target_modules.append(
                f"model.layers.{layer_idx}.{LORA_MODULE_PATHS[module_type]}"
            )
    return target_modules


def prepare_layer_subset_config(
    template_config_path: Path,
    output_config_path: Path,
    *,
    output_dir: Path,
    target_layers: list[int],
    lora_rank: int,
    module_types: str | list[str] | tuple[str, ...],
) -> tuple[Path, list[str], list[str]]:
    config = load_yaml(template_config_path)
    normalized_module_types = normalize_lora_module_types(module_types)
    target_modules = build_lora_target_modules_for_types(target_layers, normalized_module_types)
    config["finetuning_type"] = "lora"
    config["lora_rank"] = lora_rank
    config["lora_alpha"] = DEFAULT_LORA_ALPHA
    config["lora_target"] = ",".join(target_modules)
    config["output_dir"] = repo_rel(output_dir)
    write_yaml(output_config_path, config)
    return output_config_path, target_modules, normalized_module_types


def write_important_layer_selection(
    *,
    path: Path,
    important_layers: list[int],
    lora_target_modules: list[str],
    module_types: list[str],
    important_top_k: int | None,
    important_ratio: float,
    score_report_path: Path | None,
    preview_only: bool,
    train_subset_path: Path,
    train_subset_size: int,
    baseline_model_path: Path,
) -> None:
    payload = {
        "experiment": "exp11_exp7_adarank",
        "baseline_model_path": repo_rel(baseline_model_path),
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": train_subset_size,
        },
        "selection_strategy": {
            "important_top_k": important_top_k,
            "important_ratio": important_ratio,
            "preview_only": preview_only,
            "module_types": module_types,
        },
        "layer_score_report": repo_rel(score_report_path) if score_report_path is not None else None,
        "important_layers": important_layers,
        "module_types": module_types,
        "lora_target_modules": lora_target_modules,
        "num_important_layers": len(important_layers),
    }
    write_json(path, payload)


def infer_total_layers_for_high_layer_strategy(
    *,
    layer_model_path: Path,
    score_report_path: Path | None,
) -> int:
    if score_report_path is not None and score_report_path.exists():
        payload = load_json(score_report_path)
        summary = payload.get("summary", {})
        num_layers_scored = summary.get("num_layers_scored")
        if isinstance(num_layers_scored, int) and num_layers_scored > 0:
            return num_layers_scored

        ranking = payload.get("ranking")
        if isinstance(ranking, list) and ranking:
            return len(ranking)

    inferred = infer_num_hidden_layers(layer_model_path)
    if inferred is not None:
        return inferred

    return 32


def select_high_layers(total_layers: int, high_layer_count: int) -> list[int]:
    if total_layers <= 0:
        raise ValueError("total_layers 必须大于 0。")
    if high_layer_count <= 0:
        raise ValueError("high_layer_count 必须大于 0。")

    selected_count = min(high_layer_count, total_layers)
    start = total_layers - selected_count
    return list(range(start, total_layers))


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
    }


def build_comparison_summary(
    *,
    full_report_path: Path,
    important_report_path: Path,
    high18_report_path: Path,
    important_layers: list[int],
    lora_target_modules: list[str],
    important_module_types: list[str],
    high18_layers: list[int],
    high18_target_modules: list[str],
    high18_module_types: list[str],
    train_subset_path: Path,
    train_subset_size: int,
    baseline_model_path: Path,
) -> dict[str, Any]:
    full_metrics = extract_eval_metrics(full_report_path)
    important_metrics = extract_eval_metrics(important_report_path)
    high18_metrics = extract_eval_metrics(high18_report_path)

    candidates = {
        "full_rank4": {
            "name": "full_rank4",
            "report_path": repo_rel(full_report_path),
            **full_metrics,
        },
        "important_rank4": {
            "name": "important_rank4",
            "report_path": repo_rel(important_report_path),
            "important_layers": important_layers,
            "num_important_layers": len(important_layers),
            "module_types": important_module_types,
            "lora_target_modules": lora_target_modules,
            **important_metrics,
        },
        "high18_rank8": {
            "name": "high18_rank8",
            "report_path": repo_rel(high18_report_path),
            "high_layers": high18_layers,
            "num_high_layers": len(high18_layers),
            "module_types": high18_module_types,
            "lora_target_modules": high18_target_modules,
            **high18_metrics,
        },
    }

    def build_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
        return {
            "parse_ok_rate_delta": float(current["parse_ok_rate"]) - float(baseline["parse_ok_rate"]),
            "exact_match_rate_delta": float(current["exact_match_rate"]) - float(baseline["exact_match_rate"]),
            "action_match_rate_delta": float(current["action_match_rate"]) - float(baseline["action_match_rate"]),
            "avg_latency_sec_delta": float(current["avg_latency_sec"]) - float(baseline["avg_latency_sec"]),
            "avg_throughput_tps_delta": float(current["avg_throughput_tps"]) - float(baseline["avg_throughput_tps"]),
            "avg_peak_vram_mb_delta": float(current["avg_peak_vram_mb"]) - float(baseline["avg_peak_vram_mb"]),
        }

    winner = max(
        candidates.items(),
        key=lambda item: (
            float(item[1]["exact_match_rate"]),
            float(item[1]["action_match_rate"]),
            float(item[1]["parse_ok_rate"]),
        ),
    )[0]

    return {
        "experiment": "exp11_exp7_adarank",
        "baseline_model_path": repo_rel(baseline_model_path),
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": train_subset_size,
        },
        "baseline": candidates["full_rank4"],
        "important_rank4": candidates["important_rank4"],
        "high18_rank8": candidates["high18_rank8"],
        "delta_important_vs_full": build_delta(candidates["important_rank4"], candidates["full_rank4"]),
        "delta_high18_vs_full": build_delta(candidates["high18_rank8"], candidates["full_rank4"]),
        "delta_high18_vs_important": build_delta(candidates["high18_rank8"], candidates["important_rank4"]),
        "winner_by_accuracy": winner,
    }


def build_comparison_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    important = summary["important_rank4"]
    high18 = summary["high18_rank8"]
    delta_important = summary["delta_important_vs_full"]
    delta_high18 = summary["delta_high18_vs_full"]
    delta_high18_vs_important = summary["delta_high18_vs_important"]
    important_layers = ", ".join(str(idx) for idx in important["important_layers"])
    high_layers = ", ".join(str(idx) for idx in high18["high_layers"])
    train_subset = summary["train_subset"]
    baseline_model_path = summary["baseline_model_path"]
    target_preview = ", ".join(important["lora_target_modules"][:6])
    if len(important["lora_target_modules"]) > 6:
        target_preview += ", ..."
    high18_target_preview = ", ".join(high18["lora_target_modules"][:6])
    if len(high18["lora_target_modules"]) > 6:
        high18_target_preview += ", ..."
    important_module_types = ", ".join(important["module_types"])
    high18_module_types = ", ".join(high18["module_types"])

    return "\n".join(
        [
            "# 实验 11_exp7_adarank 对比结果",
            "",
            f"- 胜者（按 exact_match_rate > action_match_rate > parse_ok_rate 排序）：`{summary['winner_by_accuracy']}`",
            f"- baseline 模型：`{baseline_model_path}`（直接复用现有 rank4 模型，不在本轮重新训练）",
            f"- 固定训练集：`{train_subset['size']}` 条（`{train_subset['path']}`）",
            f"- 重要层数量：`{important['num_important_layers']}`",
            f"- 重要层编号：`{important_layers}`",
            f"- important_rank4 模块白名单：`{important_module_types}`",
            f"- high18_rank8 模块白名单：`{high18_module_types}`",
            "",
            "## 指标对比",
            "",
            "| 模型 | parse_ok_rate | exact_match_rate | action_match_rate | avg_latency_sec | avg_throughput_tps | avg_peak_vram_mb |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| full_rank4 | {baseline['parse_ok_rate']:.4f} | {baseline['exact_match_rate']:.4f} | "
                f"{baseline['action_match_rate']:.4f} | {baseline['avg_latency_sec']:.4f} | "
                f"{baseline['avg_throughput_tps']:.4f} | {baseline['avg_peak_vram_mb']:.1f} |"
            ),
            (
                f"| important_rank4 | {important['parse_ok_rate']:.4f} | {important['exact_match_rate']:.4f} | "
                f"{important['action_match_rate']:.4f} | {important['avg_latency_sec']:.4f} | "
                f"{important['avg_throughput_tps']:.4f} | {important['avg_peak_vram_mb']:.1f} |"
            ),
            (
                f"| high18_rank8 | {high18['parse_ok_rate']:.4f} | {high18['exact_match_rate']:.4f} | "
                f"{high18['action_match_rate']:.4f} | {high18['avg_latency_sec']:.4f} | "
                f"{high18['avg_throughput_tps']:.4f} | {high18['avg_peak_vram_mb']:.1f} |"
            ),
            "",
            "## Important vs Full 差值",
            "",
            f"- `parse_ok_rate_delta`: {delta_important['parse_ok_rate_delta']:+.4f}",
            f"- `exact_match_rate_delta`: {delta_important['exact_match_rate_delta']:+.4f}",
            f"- `action_match_rate_delta`: {delta_important['action_match_rate_delta']:+.4f}",
            f"- `avg_latency_sec_delta`: {delta_important['avg_latency_sec_delta']:+.4f}",
            f"- `avg_throughput_tps_delta`: {delta_important['avg_throughput_tps_delta']:+.4f}",
            f"- `avg_peak_vram_mb_delta`: {delta_important['avg_peak_vram_mb_delta']:+.1f}",
            "",
            "## High18 Rank8 vs Full 差值",
            "",
            f"- `parse_ok_rate_delta`: {delta_high18['parse_ok_rate_delta']:+.4f}",
            f"- `exact_match_rate_delta`: {delta_high18['exact_match_rate_delta']:+.4f}",
            f"- `action_match_rate_delta`: {delta_high18['action_match_rate_delta']:+.4f}",
            f"- `avg_latency_sec_delta`: {delta_high18['avg_latency_sec_delta']:+.4f}",
            f"- `avg_throughput_tps_delta`: {delta_high18['avg_throughput_tps_delta']:+.4f}",
            f"- `avg_peak_vram_mb_delta`: {delta_high18['avg_peak_vram_mb_delta']:+.1f}",
            "",
            "## High18 Rank8 vs Important Rank4 差值",
            "",
            f"- `parse_ok_rate_delta`: {delta_high18_vs_important['parse_ok_rate_delta']:+.4f}",
            f"- `exact_match_rate_delta`: {delta_high18_vs_important['exact_match_rate_delta']:+.4f}",
            f"- `action_match_rate_delta`: {delta_high18_vs_important['action_match_rate_delta']:+.4f}",
            f"- `avg_latency_sec_delta`: {delta_high18_vs_important['avg_latency_sec_delta']:+.4f}",
            f"- `avg_throughput_tps_delta`: {delta_high18_vs_important['avg_throughput_tps_delta']:+.4f}",
            f"- `avg_peak_vram_mb_delta`: {delta_high18_vs_important['avg_peak_vram_mb_delta']:+.1f}",
            "",
            "## 重要层 LoRA Target",
            "",
            f"`{target_preview}`",
            "",
            "## layer_scores Top-18 Rank8 LoRA",
            "",
            f"- Top-18 层编号：`{high_layers}`",
            f"- Target 预览：`{high18_target_preview}`",
            "",
        ]
    )


def build_two_dimensional_summary(
    *,
    full_report_path: Path,
    high18_report_path: Path,
    high18_2d_report_path: Path,
    high18_layers: list[int],
    high18_target_modules: list[str],
    high18_module_types: list[str],
    high18_2d_layers: list[int],
    high18_2d_target_modules: list[str],
    high18_2d_module_types: list[str],
    train_subset_path: Path,
    train_subset_size: int,
    baseline_model_path: Path,
) -> dict[str, Any]:
    full_metrics = extract_eval_metrics(full_report_path)
    high18_metrics = extract_eval_metrics(high18_report_path)
    high18_2d_metrics = extract_eval_metrics(high18_2d_report_path)

    baseline = {
        "name": "full_rank4",
        "report_path": repo_rel(full_report_path),
        **full_metrics,
    }
    high18 = {
        "name": "high18_rank8",
        "report_path": repo_rel(high18_report_path),
        "high_layers": high18_layers,
        "num_high_layers": len(high18_layers),
        "module_types": high18_module_types,
        "lora_target_modules": high18_target_modules,
        **high18_metrics,
    }
    high18_2d = {
        "name": "high18_rank8_2d",
        "report_path": repo_rel(high18_2d_report_path),
        "high_layers": high18_2d_layers,
        "num_high_layers": len(high18_2d_layers),
        "module_types": high18_2d_module_types,
        "lora_target_modules": high18_2d_target_modules,
        **high18_2d_metrics,
    }

    def build_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
        return {
            "parse_ok_rate_delta": float(current["parse_ok_rate"]) - float(reference["parse_ok_rate"]),
            "exact_match_rate_delta": float(current["exact_match_rate"]) - float(reference["exact_match_rate"]),
            "action_match_rate_delta": float(current["action_match_rate"]) - float(reference["action_match_rate"]),
            "avg_latency_sec_delta": float(current["avg_latency_sec"]) - float(reference["avg_latency_sec"]),
            "avg_throughput_tps_delta": float(current["avg_throughput_tps"]) - float(reference["avg_throughput_tps"]),
            "avg_peak_vram_mb_delta": float(current["avg_peak_vram_mb"]) - float(reference["avg_peak_vram_mb"]),
        }

    winner = max(
        {"high18_rank8": high18, "high18_rank8_2d": high18_2d}.items(),
        key=lambda item: (
            float(item[1]["exact_match_rate"]),
            float(item[1]["action_match_rate"]),
            float(item[1]["parse_ok_rate"]),
        ),
    )[0]

    return {
        "experiment": "exp11_exp7_adarank_2d_extension",
        "baseline_model_path": repo_rel(baseline_model_path),
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": train_subset_size,
        },
        "baseline": baseline,
        "high18_rank8": high18,
        "high18_rank8_2d": high18_2d,
        "delta_high18_vs_full": build_delta(high18, baseline),
        "delta_high18_2d_vs_full": build_delta(high18_2d, baseline),
        "delta_high18_2d_vs_high18": build_delta(high18_2d, high18),
        "winner_by_accuracy": winner,
    }


def build_two_dimensional_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    high18 = summary["high18_rank8"]
    high18_2d = summary["high18_rank8_2d"]
    delta_vs_full = summary["delta_high18_2d_vs_full"]
    delta_vs_high18 = summary["delta_high18_2d_vs_high18"]

    high18_layers = ", ".join(str(idx) for idx in high18["high_layers"])
    high18_module_types = ", ".join(high18["module_types"])
    high18_2d_module_types = ", ".join(high18_2d["module_types"])
    high18_preview = ", ".join(high18["lora_target_modules"][:6])
    if len(high18["lora_target_modules"]) > 6:
        high18_preview += ", ..."
    high18_2d_preview = ", ".join(high18_2d["lora_target_modules"][:6])
    if len(high18_2d["lora_target_modules"]) > 6:
        high18_2d_preview += ", ..."

    return "\n".join(
        [
            "# 实验 11_exp7_adarank 二维过滤补充结果",
            "",
            f"- 胜者（仅比较 `high18_rank8` 与 `high18_rank8_2d`）：`{summary['winner_by_accuracy']}`",
            f"- baseline 模型：`{summary['baseline_model_path']}`",
            f"- 固定训练集：`{summary['train_subset']['size']}` 条（`{summary['train_subset']['path']}`）",
            f"- Top-18 层编号：`{high18_layers}`",
            f"- 原始 high18_rank8 模块：`{high18_module_types}`",
            f"- 二维过滤 high18_rank8_2d 模块：`{high18_2d_module_types}`",
            "",
            "## 指标对比",
            "",
            "| 模型 | parse_ok_rate | exact_match_rate | action_match_rate | avg_latency_sec | avg_throughput_tps | avg_peak_vram_mb |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| full_rank4 | {baseline['parse_ok_rate']:.4f} | {baseline['exact_match_rate']:.4f} | "
                f"{baseline['action_match_rate']:.4f} | {baseline['avg_latency_sec']:.4f} | "
                f"{baseline['avg_throughput_tps']:.4f} | {baseline['avg_peak_vram_mb']:.1f} |"
            ),
            (
                f"| high18_rank8 | {high18['parse_ok_rate']:.4f} | {high18['exact_match_rate']:.4f} | "
                f"{high18['action_match_rate']:.4f} | {high18['avg_latency_sec']:.4f} | "
                f"{high18['avg_throughput_tps']:.4f} | {high18['avg_peak_vram_mb']:.1f} |"
            ),
            (
                f"| high18_rank8_2d | {high18_2d['parse_ok_rate']:.4f} | {high18_2d['exact_match_rate']:.4f} | "
                f"{high18_2d['action_match_rate']:.4f} | {high18_2d['avg_latency_sec']:.4f} | "
                f"{high18_2d['avg_throughput_tps']:.4f} | {high18_2d['avg_peak_vram_mb']:.1f} |"
            ),
            "",
            "## high18_rank8_2d vs Full 差值",
            "",
            f"- `parse_ok_rate_delta`: {delta_vs_full['parse_ok_rate_delta']:+.4f}",
            f"- `exact_match_rate_delta`: {delta_vs_full['exact_match_rate_delta']:+.4f}",
            f"- `action_match_rate_delta`: {delta_vs_full['action_match_rate_delta']:+.4f}",
            f"- `avg_latency_sec_delta`: {delta_vs_full['avg_latency_sec_delta']:+.4f}",
            f"- `avg_throughput_tps_delta`: {delta_vs_full['avg_throughput_tps_delta']:+.4f}",
            f"- `avg_peak_vram_mb_delta`: {delta_vs_full['avg_peak_vram_mb_delta']:+.1f}",
            "",
            "## high18_rank8_2d vs high18_rank8 差值",
            "",
            f"- `parse_ok_rate_delta`: {delta_vs_high18['parse_ok_rate_delta']:+.4f}",
            f"- `exact_match_rate_delta`: {delta_vs_high18['exact_match_rate_delta']:+.4f}",
            f"- `action_match_rate_delta`: {delta_vs_high18['action_match_rate_delta']:+.4f}",
            f"- `avg_latency_sec_delta`: {delta_vs_high18['avg_latency_sec_delta']:+.4f}",
            f"- `avg_throughput_tps_delta`: {delta_vs_high18['avg_throughput_tps_delta']:+.4f}",
            f"- `avg_peak_vram_mb_delta`: {delta_vs_high18['avg_peak_vram_mb_delta']:+.1f}",
            "",
            "## Target 预览",
            "",
            f"- `high18_rank8`: `{high18_preview}`",
            f"- `high18_rank8_2d`: `{high18_2d_preview}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    baseline_model_path = resolve_existing_model_path(args.baseline_model_path)
    train_subset_path = prepare_train_subset(args.base_config, TRAIN_SUBSET_SIZE)
    layer_data_path = args.layer_data_path.resolve()
    if layer_data_path == DEFAULT_LAYER_DATA_PATH.resolve():
        layer_data_path = train_subset_path
    print(
        "[pipeline] "
        f"baseline 模型: {baseline_model_path}，固定训练集大小: {TRAIN_SUBSET_SIZE}，"
        f"层打分数据集: {layer_data_path}",
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

    if args.dry_run:
        score_report_path = LAYER_SCORE_REPORT_PATH if LAYER_SCORE_REPORT_PATH.exists() else None
        if score_report_path is not None:
            important_layers = select_important_layers(
                load_layer_ranking(score_report_path),
                important_top_k=args.important_top_k,
                important_ratio=args.important_ratio,
            )
        else:
            important_layers = preview_important_layers(
                args.layer_model_path,
                important_top_k=args.important_top_k,
                important_ratio=args.important_ratio,
            )
    else:
        important_layers = select_important_layers(
            load_layer_ranking(LAYER_SCORE_REPORT_PATH),
            important_top_k=args.important_top_k,
            important_ratio=args.important_ratio,
        )
        score_report_path = LAYER_SCORE_REPORT_PATH

    _, lora_target_modules, important_module_types = prepare_layer_subset_config(
        IMPORTANT_TEMPLATE_CONFIG_PATH,
        GENERATED_IMPORTANT_CONFIG_PATH,
        output_dir=IMPORTANT_RANK4_OUTPUT_DIR,
        target_layers=important_layers,
        lora_rank=4,
        module_types=args.important_module_types,
    )
    write_important_layer_selection(
        path=IMPORTANT_LAYER_SELECTION_PATH,
        important_layers=important_layers,
        lora_target_modules=lora_target_modules,
        module_types=important_module_types,
        important_top_k=args.important_top_k,
        important_ratio=args.important_ratio,
        score_report_path=score_report_path,
        preview_only=args.dry_run,
        train_subset_path=train_subset_path,
        train_subset_size=TRAIN_SUBSET_SIZE,
        baseline_model_path=baseline_model_path,
    )
    print(f"[pipeline] 重要层配置已生成: {GENERATED_IMPORTANT_CONFIG_PATH}", flush=True)
    print(f"[pipeline] 重要层报告已写入: {IMPORTANT_LAYER_SELECTION_PATH}", flush=True)

    if score_report_path is not None:
        high18_layers = select_important_layers(
            load_layer_ranking(score_report_path),
            important_top_k=args.high_layer_count,
            important_ratio=1.0,
        )
    else:
        total_layers_for_high18 = infer_total_layers_for_high_layer_strategy(
            layer_model_path=args.layer_model_path,
            score_report_path=score_report_path,
        )
        high18_layers = select_high_layers(total_layers_for_high18, args.high_layer_count)
    _, high18_target_modules, high18_module_types = prepare_layer_subset_config(
        HIGH18_RANK8_TEMPLATE_CONFIG_PATH,
        GENERATED_HIGH18_RANK8_CONFIG_PATH,
        output_dir=HIGH18_RANK8_OUTPUT_DIR,
        target_layers=high18_layers,
        lora_rank=8,
        module_types=args.high_layer_module_types,
    )
    write_important_layer_selection(
        path=HIGH18_LAYER_SELECTION_PATH,
        important_layers=high18_layers,
        lora_target_modules=high18_target_modules,
        module_types=high18_module_types,
        important_top_k=args.high_layer_count,
        important_ratio=0.0,
        score_report_path=score_report_path,
        preview_only=args.dry_run,
        train_subset_path=train_subset_path,
        train_subset_size=TRAIN_SUBSET_SIZE,
        baseline_model_path=baseline_model_path,
    )
    print(f"[pipeline] Top-18 层 rank8 配置已生成: {GENERATED_HIGH18_RANK8_CONFIG_PATH}", flush=True)
    print(f"[pipeline] Top-18 层报告已写入: {HIGH18_LAYER_SELECTION_PATH}", flush=True)

    high18_2d_target_modules: list[str] = []
    high18_2d_module_types: list[str] = []
    if args.enable_high18_rank8_2d:
        _, high18_2d_target_modules, high18_2d_module_types = prepare_layer_subset_config(
            HIGH18_RANK8_TEMPLATE_CONFIG_PATH,
            GENERATED_HIGH18_RANK8_2D_CONFIG_PATH,
            output_dir=HIGH18_RANK8_2D_OUTPUT_DIR,
            target_layers=high18_layers,
            lora_rank=8,
            module_types=args.high_layer_2d_module_types,
        )
        write_important_layer_selection(
            path=HIGH18_2D_LAYER_SELECTION_PATH,
            important_layers=high18_layers,
            lora_target_modules=high18_2d_target_modules,
            module_types=high18_2d_module_types,
            important_top_k=args.high_layer_count,
            important_ratio=0.0,
            score_report_path=score_report_path,
            preview_only=args.dry_run,
            train_subset_path=train_subset_path,
            train_subset_size=TRAIN_SUBSET_SIZE,
            baseline_model_path=baseline_model_path,
        )
        print(
            f"[pipeline] Top-18 层二维过滤 rank8 配置已生成: {GENERATED_HIGH18_RANK8_2D_CONFIG_PATH}",
            flush=True,
        )
        print(f"[pipeline] Top-18 层二维过滤报告已写入: {HIGH18_2D_LAYER_SELECTION_PATH}", flush=True)

    prepare_train_override(
        config_path=GENERATED_IMPORTANT_CONFIG_PATH,
        override_path=IMPORTANT_RANK4_TRAIN_OVERRIDE,
        gpus=args.gpus,
        train_file=train_subset_path,
    )
    prepare_train_override(
        config_path=GENERATED_HIGH18_RANK8_CONFIG_PATH,
        override_path=HIGH18_RANK8_TRAIN_OVERRIDE,
        gpus=args.gpus,
        train_file=train_subset_path,
    )
    if args.enable_high18_rank8_2d:
        prepare_train_override(
            config_path=GENERATED_HIGH18_RANK8_2D_CONFIG_PATH,
            override_path=HIGH18_RANK8_2D_TRAIN_OVERRIDE,
            gpus=args.gpus,
            train_file=train_subset_path,
        )

    run_stage(
        "train_important_rank4",
        [
            sys.executable,
            str(TRAIN_RUNNER),
            "--base-config",
            str(args.base_config),
            "--config",
            str(IMPORTANT_RANK4_TRAIN_OVERRIDE),
            *(["--dry-run"] if args.dry_run else []),
            "output_dir=" + str(IMPORTANT_RANK4_OUTPUT_DIR),
        ],
        LOGS_DIR / "train_important_rank4.log",
        dry_run=args.dry_run,
    )
    run_stage(
        "train_high18_rank8",
        [
            sys.executable,
            str(TRAIN_RUNNER),
            "--base-config",
            str(args.base_config),
            "--config",
            str(HIGH18_RANK8_TRAIN_OVERRIDE),
            *(["--dry-run"] if args.dry_run else []),
            "output_dir=" + str(HIGH18_RANK8_OUTPUT_DIR),
        ],
        LOGS_DIR / "train_high18_rank8.log",
        dry_run=args.dry_run,
    )
    if args.enable_high18_rank8_2d:
        run_stage(
            "train_high18_rank8_2d",
            [
                sys.executable,
                str(TRAIN_RUNNER),
                "--base-config",
                str(args.base_config),
                "--config",
                str(HIGH18_RANK8_2D_TRAIN_OVERRIDE),
                *(["--dry-run"] if args.dry_run else []),
                "output_dir=" + str(HIGH18_RANK8_2D_OUTPUT_DIR),
            ],
            LOGS_DIR / "train_high18_rank8_2d.log",
            dry_run=args.dry_run,
        )

    if args.skip_eval:
        print("[pipeline] 已跳过评测阶段。", flush=True)
        return

    prepare_eval_override(
        model_path=baseline_model_path,
        report_path=REPORTS_DIR / "accuracy_report_full_rank4.json",
        override_path=FULL_RANK4_EVAL_OVERRIDE,
    )
    prepare_eval_override(
        model_path=IMPORTANT_RANK4_OUTPUT_DIR,
        report_path=REPORTS_DIR / "accuracy_report_important_rank4.json",
        override_path=IMPORTANT_RANK4_EVAL_OVERRIDE,
    )
    prepare_eval_override(
        model_path=HIGH18_RANK8_OUTPUT_DIR,
        report_path=REPORTS_DIR / "accuracy_report_high18_rank8.json",
        override_path=HIGH18_RANK8_EVAL_OVERRIDE,
    )
    if args.enable_high18_rank8_2d:
        prepare_eval_override(
            model_path=HIGH18_RANK8_2D_OUTPUT_DIR,
            report_path=REPORTS_DIR / "accuracy_report_high18_rank8_2d.json",
            override_path=HIGH18_RANK8_2D_EVAL_OVERRIDE,
        )

    run_stage(
        "eval_full_rank4",
        [
            sys.executable,
            str(EVAL_RUNNER),
            "--base-config",
            str(args.base_config),
            "--config",
            str(FULL_RANK4_EVAL_OVERRIDE),
        ],
        LOGS_DIR / "eval_full_rank4.log",
        dry_run=args.dry_run,
    )

    run_stage(
        "eval_important_rank4",
        [
            sys.executable,
            str(EVAL_RUNNER),
            "--base-config",
            str(args.base_config),
            "--config",
            str(IMPORTANT_RANK4_EVAL_OVERRIDE),
        ],
        LOGS_DIR / "eval_important_rank4.log",
        dry_run=args.dry_run,
    )
    run_stage(
        "eval_high18_rank8",
        [
            sys.executable,
            str(EVAL_RUNNER),
            "--base-config",
            str(args.base_config),
            "--config",
            str(HIGH18_RANK8_EVAL_OVERRIDE),
        ],
        LOGS_DIR / "eval_high18_rank8.log",
        dry_run=args.dry_run,
    )
    if args.enable_high18_rank8_2d:
        run_stage(
            "eval_high18_rank8_2d",
            [
                sys.executable,
                str(EVAL_RUNNER),
                "--base-config",
                str(args.base_config),
                "--config",
                str(HIGH18_RANK8_2D_EVAL_OVERRIDE),
            ],
            LOGS_DIR / "eval_high18_rank8_2d.log",
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[pipeline] dry-run 已完成，评测对比摘要将在真实运行后生成。", flush=True)
        return

    summary = build_comparison_summary(
        full_report_path=REPORTS_DIR / "accuracy_report_full_rank4.json",
        important_report_path=REPORTS_DIR / "accuracy_report_important_rank4.json",
        high18_report_path=REPORTS_DIR / "accuracy_report_high18_rank8.json",
        important_layers=important_layers,
        lora_target_modules=lora_target_modules,
        important_module_types=important_module_types,
        high18_layers=high18_layers,
        high18_target_modules=high18_target_modules,
        high18_module_types=high18_module_types,
        train_subset_path=train_subset_path,
        train_subset_size=TRAIN_SUBSET_SIZE,
        baseline_model_path=baseline_model_path,
    )
    write_json(COMPARISON_SUMMARY_JSON_PATH, summary)
    COMPARISON_SUMMARY_MD_PATH.write_text(
        build_comparison_markdown(summary),
        encoding="utf-8",
    )
    if args.enable_high18_rank8_2d:
        summary_2d = build_two_dimensional_summary(
            full_report_path=REPORTS_DIR / "accuracy_report_full_rank4.json",
            high18_report_path=REPORTS_DIR / "accuracy_report_high18_rank8.json",
            high18_2d_report_path=REPORTS_DIR / "accuracy_report_high18_rank8_2d.json",
            high18_layers=high18_layers,
            high18_target_modules=high18_target_modules,
            high18_module_types=high18_module_types,
            high18_2d_layers=high18_layers,
            high18_2d_target_modules=high18_2d_target_modules,
            high18_2d_module_types=high18_2d_module_types,
            train_subset_path=train_subset_path,
            train_subset_size=TRAIN_SUBSET_SIZE,
            baseline_model_path=baseline_model_path,
        )
        write_json(COMPARISON_SUMMARY_2D_JSON_PATH, summary_2d)
        COMPARISON_SUMMARY_2D_MD_PATH.write_text(
            build_two_dimensional_markdown(summary_2d),
            encoding="utf-8",
        )

    print("[pipeline] 实验流程已完成。", flush=True)
    print(f"[pipeline] 对比摘要 JSON: {COMPARISON_SUMMARY_JSON_PATH}", flush=True)
    print(f"[pipeline] 对比摘要 MD  : {COMPARISON_SUMMARY_MD_PATH}", flush=True)
    if args.enable_high18_rank8_2d:
        print(f"[pipeline] 二维补充摘要 JSON: {COMPARISON_SUMMARY_2D_JSON_PATH}", flush=True)
        print(f"[pipeline] 二维补充摘要 MD  : {COMPARISON_SUMMARY_2D_MD_PATH}", flush=True)
    print(f"[pipeline] 准确率优胜者  : {summary['winner_by_accuracy']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[pipeline] 失败: {exc}", file=sys.stderr)
        raise
