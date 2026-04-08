#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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

RANK8_TEMPLATE_CONFIG_PATH = CONFIG_DIR / "train_rank8_subset_template.yaml"
DEFAULT_BASELINE_MODEL_PATH = REPO_ROOT / "model" / "qwen2.5-3b-genesis-lora-rank-4"
DEFAULT_LAYER_DATA_PATH = REPO_ROOT / "data_prepare" / "splits" / "train.json"
DEFAULT_LAYER_MODEL_PATH = REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct"

TRAIN_SUBSET_SIZE = 600
TRAIN_SUBSET_PATH = TEMP_DIR / f"train_subset_{TRAIN_SUBSET_SIZE}.json"
LAYER_SCORE_REPORT_PATH = REPORTS_DIR / "layer_scores.json"
BASELINE_REPORT_PATH = REPORTS_DIR / "accuracy_report_full_rank4.json"
BASELINE_EVAL_OVERRIDE_PATH = TEMP_DIR / "eval_full_rank4_override.yaml"

DEFAULT_SELECTED_LAYER_COUNT = 18
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "实验 exp11：比较全层 rank4 baseline 与五个 18 层 rank8 消融分支，"
            "验证收益是否来自重要层选择。"
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
        help="现成的全层 rank4 baseline 模型路径，只做评测，不重复训练。",
    )
    parser.add_argument(
        "--layer-model-path",
        type=Path,
        default=DEFAULT_LAYER_MODEL_PATH,
        help="层打分使用的基础模型路径。",
    )
    parser.add_argument(
        "--layer-data-path",
        type=Path,
        default=DEFAULT_LAYER_DATA_PATH,
        help="层打分使用的数据集；若保持默认，则自动替换为固定训练子集。",
    )
    parser.add_argument(
        "--layer-sample-size",
        type=int,
        default=100,
        help="层打分采样条数。",
    )
    parser.add_argument(
        "--selected-layer-count",
        type=int,
        default=DEFAULT_SELECTED_LAYER_COUNT,
        help="所有 rank8 分支统一选择的层数，默认 18。",
    )
    parser.add_argument(
        "--random-layer-seed",
        type=int,
        default=42,
        help="随机 18 层分支的随机种子。",
    )
    parser.add_argument(
        "--rank8-module-types",
        type=str,
        default=",".join(ALL_LORA_MODULE_TYPES),
        help="所有 rank8 分支统一使用的模块类型白名单，逗号分隔。",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="传给训练脚本的 GPU 编号字符串，例如 0 或 0,1。",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="只执行层打分和训练，不做评测。",
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
        raise ValueError(f"训练集总条数只有 {total_rows}，无法截取固定 {subset_size} 条样本。")

    subset = rows[:subset_size]
    write_json(TRAIN_SUBSET_PATH, subset)
    print(
        f"[exp11] 已生成固定训练子集: {TRAIN_SUBSET_PATH} "
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
    train_file: Path,
) -> Path:
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
            return int(raw.split("_", 1)[1])
        return int(raw)
    raise ValueError(f"无法解析层编号: {raw!r}")


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


def build_output_dir(variant_name: str) -> Path:
    return REPO_ROOT / "output" / "exp13_exp11_ablation" / variant_name


def build_generated_config_path(variant_name: str) -> Path:
    return TEMP_DIR / f"train_{variant_name}_generated.yaml"


def build_train_override_path(variant_name: str) -> Path:
    return TEMP_DIR / f"train_{variant_name}_override.yaml"


def build_eval_override_path(variant_name: str) -> Path:
    return TEMP_DIR / f"eval_{variant_name}_override.yaml"


def build_eval_report_path(variant_name: str) -> Path:
    return REPORTS_DIR / f"accuracy_report_{variant_name}.json"


def build_selection_report_path(variant_name: str) -> Path:
    return REPORTS_DIR / f"{variant_name}_layers.json"


def prepare_rank8_config(
    *,
    variant_name: str,
    target_layers: list[int],
    module_types: list[str],
) -> tuple[Path, list[str]]:
    config = load_yaml(RANK8_TEMPLATE_CONFIG_PATH)
    target_modules = build_lora_target_modules(target_layers, module_types)
    config["lora_rank"] = 8
    config["lora_alpha"] = DEFAULT_LORA_ALPHA
    config["lora_target"] = ",".join(target_modules)
    config["output_dir"] = repo_rel(build_output_dir(variant_name))

    generated_config_path = build_generated_config_path(variant_name)
    write_yaml(generated_config_path, config)
    return generated_config_path, target_modules


def select_top_layers_by_score(ranking: list[dict[str, Any]], selected_count: int) -> list[int]:
    return sorted({_extract_layer_index(item["layer"]) for item in ranking[:selected_count]})


def select_random_layers(total_layers: int, selected_count: int, seed: int) -> list[int]:
    actual_count = min(selected_count, total_layers)
    rng = random.Random(seed)
    return sorted(rng.sample(list(range(total_layers)), actual_count))


def select_low_layers(total_layers: int, selected_count: int) -> list[int]:
    actual_count = min(selected_count, total_layers)
    return list(range(actual_count))


def select_high_layers(total_layers: int, selected_count: int) -> list[int]:
    actual_count = min(selected_count, total_layers)
    return list(range(total_layers - actual_count, total_layers))


def select_middle_layers(total_layers: int, selected_count: int) -> list[int]:
    actual_count = min(selected_count, total_layers)
    start = max(0, (total_layers - actual_count) // 2)
    return list(range(start, start + actual_count))


def build_variant_specs(
    *,
    total_layers: int,
    selected_layer_count: int,
    ranking: list[dict[str, Any]] | None,
    random_layer_seed: int,
) -> list[dict[str, Any]]:
    actual_count = min(selected_layer_count, total_layers)
    if actual_count <= 0:
        raise ValueError("selected_layer_count 必须大于 0。")

    if ranking is None:
        top_layers = select_high_layers(total_layers, actual_count)
        top_label = f"按打分选 Top-{actual_count} 层（dry-run 占位）"
    else:
        top_layers = select_top_layers_by_score(ranking, actual_count)
        top_label = f"按打分选 Top-{actual_count} 层"

    return [
        {
            "name": f"random{actual_count}_rank8",
            "selection_label": f"不打分，随机选 {actual_count} 层",
            "layers": select_random_layers(total_layers, actual_count, random_layer_seed),
            "score_based": False,
            "extra": {"random_seed": random_layer_seed},
        },
        {
            "name": f"top{actual_count}_rank8",
            "selection_label": top_label,
            "layers": top_layers,
            "score_based": True,
            "extra": None,
        },
        {
            "name": f"high{actual_count}_rank8",
            "selection_label": f"只选高层 {actual_count} 层",
            "layers": select_high_layers(total_layers, actual_count),
            "score_based": False,
            "extra": None,
        },
        {
            "name": f"mid{actual_count}_rank8",
            "selection_label": f"只选中层 {actual_count} 层",
            "layers": select_middle_layers(total_layers, actual_count),
            "score_based": False,
            "extra": None,
        },
        {
            "name": f"low{actual_count}_rank8",
            "selection_label": f"只选低层 {actual_count} 层",
            "layers": select_low_layers(total_layers, actual_count),
            "score_based": False,
            "extra": None,
        },
    ]


def write_selection_report(
    *,
    variant: dict[str, Any],
    selected_layer_count: int,
    total_layers: int,
    module_types: list[str],
    target_modules: list[str],
    train_subset_path: Path,
    baseline_model_path: Path,
    score_report_path: Path | None,
    preview_only: bool,
) -> None:
    payload: dict[str, Any] = {
        "experiment": "exp11_layer_selection_ablation",
        "variant_name": variant["name"],
        "selection_label": variant["selection_label"],
        "baseline_model_path": repo_rel(baseline_model_path),
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": TRAIN_SUBSET_SIZE,
        },
        "selection_strategy": {
            "selected_layer_count": selected_layer_count,
            "total_layers": total_layers,
            "score_based": variant["score_based"],
            "preview_only": preview_only,
        },
        "layer_score_report": repo_rel(score_report_path) if score_report_path is not None else None,
        "target_layers": variant["layers"],
        "num_target_layers": len(variant["layers"]),
        "module_types": module_types,
        "lora_target_modules": target_modules,
    }
    if variant.get("extra") is not None:
        payload["extra"] = variant["extra"]
    write_json(build_selection_report_path(variant["name"]), payload)


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


def build_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "parse_ok_rate_delta": float(current["parse_ok_rate"]) - float(baseline["parse_ok_rate"]),
        "exact_match_rate_delta": float(current["exact_match_rate"]) - float(baseline["exact_match_rate"]),
        "action_match_rate_delta": float(current["action_match_rate"]) - float(baseline["action_match_rate"]),
        "avg_latency_sec_delta": float(current["avg_latency_sec"]) - float(baseline["avg_latency_sec"]),
        "avg_throughput_tps_delta": float(current["avg_throughput_tps"]) - float(baseline["avg_throughput_tps"]),
        "avg_peak_vram_mb_delta": float(current["avg_peak_vram_mb"]) - float(baseline["avg_peak_vram_mb"]),
    }


def build_comparison_summary(
    *,
    baseline_model_path: Path,
    train_subset_path: Path,
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = {
        "name": "full_rank4",
        "model_path": repo_rel(baseline_model_path),
        "report_path": repo_rel(BASELINE_REPORT_PATH),
        **extract_eval_metrics(BASELINE_REPORT_PATH),
    }

    rank8_variants: list[dict[str, Any]] = []
    for variant in variants:
        result = {
            "name": variant["name"],
            "selection_label": variant["selection_label"],
            "report_path": repo_rel(build_eval_report_path(variant["name"])),
            "output_dir": repo_rel(build_output_dir(variant["name"])),
            "target_layers": variant["layers"],
            "num_target_layers": len(variant["layers"]),
            "module_types": variant["module_types"],
            **extract_eval_metrics(build_eval_report_path(variant["name"])),
        }
        if variant.get("extra") is not None:
            result["extra"] = variant["extra"]
        result["delta_vs_full_rank4"] = build_delta(result, baseline)
        rank8_variants.append(result)

    winner = max(
        rank8_variants,
        key=lambda item: (
            float(item["exact_match_rate"]),
            float(item["action_match_rate"]),
            float(item["parse_ok_rate"]),
        ),
    )["name"]

    return {
        "experiment": "exp11_layer_selection_ablation",
        "question": "收益是否来自重要层选择，而不是少量层 + 更高 rank 的偶然组合。",
        "baseline": baseline,
        "train_subset": {
            "path": repo_rel(train_subset_path),
            "size": TRAIN_SUBSET_SIZE,
        },
        "rank8_variants": rank8_variants,
        "winner_among_rank8_variants": winner,
    }


def build_comparison_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    lines = [
        "# exp11 层选择消融结果",
        "",
        f"- 实验问题：{summary['question']}",
        f"- baseline：`full_rank4`，模型路径 `{baseline['model_path']}`",
        (
            f"- 固定训练集：`{summary['train_subset']['size']}` 条，"
            f"路径 `{summary['train_subset']['path']}`"
        ),
        f"- rank8 分支优胜者：`{summary['winner_among_rank8_variants']}`",
        "",
        "## 指标总表",
        "",
        "| 模型 | 选层方式 | parse_ok_rate | exact_match_rate | action_match_rate | avg_latency_sec | avg_throughput_tps | avg_peak_vram_mb |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| full_rank4 | 全层 rank4 baseline | {baseline['parse_ok_rate']:.4f} | "
            f"{baseline['exact_match_rate']:.4f} | {baseline['action_match_rate']:.4f} | "
            f"{baseline['avg_latency_sec']:.4f} | {baseline['avg_throughput_tps']:.4f} | "
            f"{baseline['avg_peak_vram_mb']:.1f} |"
        ),
    ]

    for item in summary["rank8_variants"]:
        lines.append(
            (
                f"| {item['name']} | {item['selection_label']} | {item['parse_ok_rate']:.4f} | "
                f"{item['exact_match_rate']:.4f} | {item['action_match_rate']:.4f} | "
                f"{item['avg_latency_sec']:.4f} | {item['avg_throughput_tps']:.4f} | "
                f"{item['avg_peak_vram_mb']:.1f} |"
            )
        )

    lines.extend(["", "## 各 rank8 分支相对 full_rank4 的差值", ""])
    for item in summary["rank8_variants"]:
        delta = item["delta_vs_full_rank4"]
        layers_text = ", ".join(str(idx) for idx in item["target_layers"])
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- 选层方式：`{item['selection_label']}`",
                f"- 层编号：`{layers_text}`",
                f"- `parse_ok_rate_delta`: {delta['parse_ok_rate_delta']:+.4f}",
                f"- `exact_match_rate_delta`: {delta['exact_match_rate_delta']:+.4f}",
                f"- `action_match_rate_delta`: {delta['action_match_rate_delta']:+.4f}",
                f"- `avg_latency_sec_delta`: {delta['avg_latency_sec_delta']:+.4f}",
                f"- `avg_throughput_tps_delta`: {delta['avg_throughput_tps_delta']:+.4f}",
                f"- `avg_peak_vram_mb_delta`: {delta['avg_peak_vram_mb_delta']:+.1f}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    baseline_model_path = resolve_existing_model_path(args.baseline_model_path)
    train_subset_path = prepare_train_subset(args.base_config, TRAIN_SUBSET_SIZE)

    layer_data_path = args.layer_data_path.resolve()
    if layer_data_path == DEFAULT_LAYER_DATA_PATH.resolve():
        layer_data_path = train_subset_path

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

    module_types = normalize_lora_module_types(args.rank8_module_types)
    total_layers = infer_num_hidden_layers(args.layer_model_path)
    if total_layers is None:
        total_layers = 32
        print("[exp11] 未能从模型配置推断层数，默认按 32 层处理。", flush=True)

    if args.dry_run and not LAYER_SCORE_REPORT_PATH.exists():
        ranking = None
        print("[exp11] dry-run 下未发现现成 layer_scores.json，Top-18 分支会先用高层占位。", flush=True)
    else:
        ranking = load_layer_ranking(LAYER_SCORE_REPORT_PATH)
        total_layers = max(total_layers, len(ranking))

    variants = build_variant_specs(
        total_layers=total_layers,
        selected_layer_count=args.selected_layer_count,
        ranking=ranking,
        random_layer_seed=args.random_layer_seed,
    )

    for variant in variants:
        generated_config_path, target_modules = prepare_rank8_config(
            variant_name=variant["name"],
            target_layers=variant["layers"],
            module_types=module_types,
        )
        variant["generated_config_path"] = generated_config_path
        variant["module_types"] = module_types

        write_selection_report(
            variant=variant,
            selected_layer_count=args.selected_layer_count,
            total_layers=total_layers,
            module_types=module_types,
            target_modules=target_modules,
            train_subset_path=train_subset_path,
            baseline_model_path=baseline_model_path,
            score_report_path=LAYER_SCORE_REPORT_PATH if variant["score_based"] and ranking is not None else None,
            preview_only=args.dry_run,
        )

        prepare_train_override(
            config_path=generated_config_path,
            override_path=build_train_override_path(variant["name"]),
            gpus=args.gpus,
            train_file=train_subset_path,
        )

    for variant in variants:
        run_stage(
            f"train_{variant['name']}",
            [
                sys.executable,
                str(TRAIN_RUNNER),
                "--base-config",
                str(args.base_config),
                "--config",
                str(build_train_override_path(variant["name"])),
                *(["--dry-run"] if args.dry_run else []),
                "output_dir=" + str(build_output_dir(variant["name"])),
            ],
            LOGS_DIR / f"train_{variant['name']}.log",
            dry_run=args.dry_run,
        )

    if args.skip_eval:
        print("[exp11] 已跳过评测阶段。", flush=True)
        return

    prepare_eval_override(
        model_path=baseline_model_path,
        report_path=BASELINE_REPORT_PATH,
        override_path=BASELINE_EVAL_OVERRIDE_PATH,
    )
    for variant in variants:
        prepare_eval_override(
            model_path=build_output_dir(variant["name"]),
            report_path=build_eval_report_path(variant["name"]),
            override_path=build_eval_override_path(variant["name"]),
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
    for variant in variants:
        run_stage(
            f"eval_{variant['name']}",
            [
                sys.executable,
                str(EVAL_RUNNER),
                "--base-config",
                str(args.base_config),
                "--config",
                str(build_eval_override_path(variant["name"])),
            ],
            LOGS_DIR / f"eval_{variant['name']}.log",
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[exp11] dry-run 已完成，真实运行后会生成 comparison_summary.*。", flush=True)
        return

    summary = build_comparison_summary(
        baseline_model_path=baseline_model_path,
        train_subset_path=train_subset_path,
        variants=variants,
    )
    write_json(REPORTS_DIR / "comparison_summary.json", summary)
    (REPORTS_DIR / "comparison_summary.md").write_text(
        build_comparison_markdown(summary),
        encoding="utf-8",
    )

    print("[exp11] 实验流程已完成。", flush=True)
    print(f"[exp11] 对比摘要 JSON: {REPORTS_DIR / 'comparison_summary.json'}", flush=True)
    print(f"[exp11] 对比摘要 MD  : {REPORTS_DIR / 'comparison_summary.md'}", flush=True)
    print(f"[exp11] rank8 优胜者  : {summary['winner_among_rank8_variants']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[exp11] 失败: {exc}", file=sys.stderr)
        raise
