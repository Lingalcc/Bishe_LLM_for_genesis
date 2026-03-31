#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("缺少 matplotlib，请先安装：pip install matplotlib") from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("缺少 pandas，请先安装：pip install pandas") from exc

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError as exc:
    raise SystemExit("缺少 ruamel.yaml，请先安装：pip install ruamel.yaml") from exc

from src.utils.plotting import configure_report_matplotlib

configure_report_matplotlib(matplotlib)


# =========================
# Exp3 方法对比配置
# =========================
METHODS = ["lora", "qlora", "dora", "galore"]
BASE_CONFIG_PATH = REPO_ROOT / "configs/base.yaml"
TRAIN_CONFIG_DIR = REPO_ROOT / "experiments/02_finetune_exp/configs"
CLI_PATH = REPO_ROOT / "cli.py"

REPORTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TEMP_DIR = EXPERIMENT_DIR / ".cache"

RESULTS_CSV_PATH = REPORTS_DIR / "exp3_methods_comparison.csv"
DUAL_AXIS_FIG_PATH = REPORTS_DIR / "exp3_methods_comparison_dual_axis.png"
RADAR_FIG_PATH = REPORTS_DIR / "exp3_methods_comparison_radar.png"
OUTPUT_MODEL_ROOT = REPO_ROOT / "output/exp3_methods"

GPU_ID = 0
SLEEP_SECONDS = 15
EXPERIMENT_MAX_SAMPLES = 625

GLOBAL_TRAIN_OVERRIDES: dict[str, Any] = {
    "max_samples": EXPERIMENT_MAX_SAMPLES,
    "num_train_epochs": 3,
    "lr_scheduler_type": "cosine",
    "overwrite_output_dir": True,
}

# 说明：
# - 为了保证对比公平，统一使用相同的样本规模和有效 batch。
# - GaLore 保持更小的微批尺寸以降低 OOM 风险，但通过更高的梯度累积维持同等有效 batch。
METHOD_TRAIN_OVERRIDES: dict[str, dict[str, Any]] = {
    "lora": {
        "lora_rank": 4,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "lora_target": "all",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
    "qlora": {
        "lora_rank": 4,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "lora_target": "all",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
    "dora": {
        "lora_rank": 4,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "lora_target": "all",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
    "galore": {
        "cutoff_len": 512,
        "learning_rate": 2e-5,
        "optim": "adamw_torch",
        "pure_bf16": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
}

GALORE_UNUSED_KEYS = ("lora_rank", "lora_alpha", "lora_target")

TRAIN_SUMMARY_TIME_RE = re.compile(r"\[finetune\]\s+time \(sec\)\s*:\s*(?P<value>[0-9.]+)")
TRAIN_SUMMARY_FINAL_LOSS_RE = re.compile(r"\[finetune\]\s+final loss\s*:\s*(?P<value>[0-9.]+)")
TRAIN_SUMMARY_MIN_LOSS_RE = re.compile(
    r"\[finetune\]\s+min loss\s*:\s*(?P<loss>[0-9.]+)\s+\(step\s+(?P<step>\d+)\)"
)
TRAIN_SUMMARY_PEAK_VRAM_RE = re.compile(r"\[finetune\]\s+peak VRAM\s*:\s*(?P<value>[0-9.]+)\s+MB")
TRAIN_SUMMARY_PEAK_DELTA_VRAM_RE = re.compile(
    r"\[finetune\]\s+peak ΔVRAM\s*:\s*(?P<value>[0-9.]+)\s+MB"
)


def ensure_runtime_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def build_yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    yaml = build_yaml()
    with path.open("r", encoding="utf-8") as file:
        data = yaml.load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点必须是映射类型: {path}")
    return data


def dump_yaml_mapping(path: Path, data: dict[str, Any]) -> None:
    yaml = build_yaml()
    with path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file)


def repo_relative_str(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def get_train_config_path(method: str) -> Path:
    return TRAIN_CONFIG_DIR / f"llamafactory_train_{method}_sft.yaml"


def get_private_train_config_path(method: str) -> Path:
    return TEMP_DIR / f"{method}_train_private.yaml"


def get_train_override_path(method: str) -> Path:
    return TEMP_DIR / f"{method}_train_override.yaml"


def get_eval_override_path(method: str) -> Path:
    return TEMP_DIR / f"{method}_eval_override.yaml"


def get_method_report_path(method: str) -> Path:
    return REPORTS_DIR / f"accuracy_report_{method}.json"


def get_method_train_log_path(method: str) -> Path:
    return LOGS_DIR / f"{method}_train.log"


def get_method_eval_log_path(method: str) -> Path:
    return LOGS_DIR / f"{method}_eval.log"


def get_method_output_dir(method: str) -> Path:
    return OUTPUT_MODEL_ROOT / f"qwen2.5-3b-genesis-{method}"


def build_private_train_config(method: str) -> Path:
    base_config_path = get_train_config_path(method)
    data = load_yaml_mapping(base_config_path)

    for key, value in GLOBAL_TRAIN_OVERRIDES.items():
        data[key] = value

    for key, value in METHOD_TRAIN_OVERRIDES[method].items():
        data[key] = value

    data["output_dir"] = repo_relative_str(get_method_output_dir(method))

    if method == "galore":
        for key in GALORE_UNUSED_KEYS:
            data.pop(key, None)
    else:
        data.pop("optim", None)

    private_config_path = get_private_train_config_path(method)
    dump_yaml_mapping(private_config_path, data)
    print(f"[config:{method}] 已生成私有训练配置: {private_config_path}", flush=True)
    return private_config_path


def write_train_override(method: str, train_config_path: Path) -> Path:
    override_path = get_train_override_path(method)
    override = {
        "finetune": {
            "train": {
                "config": repo_relative_str(train_config_path),
                "finetune_method": method,
                "gpus": str(GPU_ID),
                "dry_run": False,
            }
        }
    }
    dump_yaml_mapping(override_path, override)
    return override_path


def write_eval_override(method: str, model_path: Path, report_path: Path) -> Path:
    override_path = get_eval_override_path(method)
    override = {
        "test": {
            "accuracy_eval": {
                "mode": "local",
                "report_file": repo_relative_str(report_path),
                "model_path": repo_relative_str(model_path),
                "backend": "transformers",
                "temperature": 0.0,
                "trust_remote_code": True,
            }
        }
    }
    dump_yaml_mapping(override_path, override)
    return override_path


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if any(token in lowered for token in ("cuda out of memory", "out of memory", "cublas_status_alloc_failed")):
        return "oom"
    if any(
        token in lowered
        for token in (
            "missing required package",
            "modulenotfounderror",
            "importerror",
            "bitsandbytes",
            "huggingface_hub",
            "galore_torch",
            "distribution was not found",
        )
    ):
        return "dependency"
    if any(
        token in lowered
        for token in (
            "unsupported",
            "valueerror",
            "config file not found",
            "missing required split files",
            "yaml",
            "file not found",
        )
    ):
        return "config"
    return "runtime"


def stream_subprocess(command: list[str], *, stage: str, log_path: Path) -> None:
    print(f"[{stage}] 执行命令: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"stage: {stage}\n")
        log_file.write(f"cwd: {REPO_ROOT}\n")
        log_file.write(f"command: {' '.join(command)}\n\n")
        log_file.flush()

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
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def extract_train_metrics(log_path: Path) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "train_time_sec": None,
        "final_loss": None,
        "min_loss": None,
        "min_loss_step": None,
        "train_peak_vram_mb": None,
        "train_peak_delta_vram_mb": None,
    }
    if not log_path.exists():
        return metrics

    for line in log_path.read_text(encoding="utf-8").splitlines():
        matched_time = TRAIN_SUMMARY_TIME_RE.search(line)
        if matched_time:
            metrics["train_time_sec"] = float(matched_time.group("value"))
            continue

        matched_final_loss = TRAIN_SUMMARY_FINAL_LOSS_RE.search(line)
        if matched_final_loss:
            metrics["final_loss"] = float(matched_final_loss.group("value"))
            continue

        matched_min_loss = TRAIN_SUMMARY_MIN_LOSS_RE.search(line)
        if matched_min_loss:
            metrics["min_loss"] = float(matched_min_loss.group("loss"))
            metrics["min_loss_step"] = int(matched_min_loss.group("step"))
            continue

        matched_peak_vram = TRAIN_SUMMARY_PEAK_VRAM_RE.search(line)
        if matched_peak_vram:
            metrics["train_peak_vram_mb"] = float(matched_peak_vram.group("value"))
            continue

        matched_peak_delta_vram = TRAIN_SUMMARY_PEAK_DELTA_VRAM_RE.search(line)
        if matched_peak_delta_vram:
            metrics["train_peak_delta_vram_mb"] = float(matched_peak_delta_vram.group("value"))

    return metrics


def load_accuracy_metrics(report_path: Path) -> tuple[float, float, float]:
    if not report_path.exists():
        raise FileNotFoundError(f"评估报告不存在: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    parse_ok = float(report.get("parse_ok_rate", 0.0))
    exact_match = float(report.get("exact_match_rate", 0.0))
    action_match = float(report.get("action_match_rate", 0.0))
    return parse_ok, exact_match, action_match


def safe_sleep(stage: str, seconds: int = SLEEP_SECONDS) -> None:
    print(f"[{stage}] 等待 {seconds}s，确保 GPU/CPU 资源充分回收。", flush=True)
    time.sleep(seconds)


def run_single_method(method: str) -> dict[str, Any]:
    print(f"\n{'=' * 24} 开始实验: {method} {'=' * 24}", flush=True)

    train_config_path = build_private_train_config(method)
    train_override_path = write_train_override(method, train_config_path)
    model_output_dir = get_method_output_dir(method)
    report_path = get_method_report_path(method)
    eval_override_path = write_eval_override(method, model_output_dir, report_path)
    train_log_path = get_method_train_log_path(method)
    eval_log_path = get_method_eval_log_path(method)

    train_status = "success"
    eval_status = "skipped"
    failure_reason = ""
    parse_ok = 0.0
    exact_match = 0.0
    action_match = 0.0
    train_started_at = time.time()

    try:
        stream_subprocess(
            [
                sys.executable,
                str(CLI_PATH),
                "finetune",
                "start",
                "--base-config",
                str(BASE_CONFIG_PATH),
                "--config",
                str(train_override_path),
                "--finetune-method",
                method,
            ],
            stage=f"train:{method}",
            log_path=train_log_path,
        )
    except Exception as exc:
        train_status = classify_failure(read_text_if_exists(train_log_path))
        failure_reason = f"{type(exc).__name__}: {exc}"
        print(f"[warning] {method} 训练失败，分类={train_status}，原因={failure_reason}", flush=True)

    train_metrics = extract_train_metrics(train_log_path)
    if train_metrics["train_time_sec"] is None:
        train_metrics["train_time_sec"] = round(time.time() - train_started_at, 2)

    print(
        "[train:{method}] 训练结束，状态={status}，峰值显存={peak} MB，峰值增量显存={delta} MB，训练耗时={time_sec}s".format(
            method=method,
            status=train_status,
            peak=train_metrics["train_peak_vram_mb"],
            delta=train_metrics["train_peak_delta_vram_mb"],
            time_sec=train_metrics["train_time_sec"],
        ),
        flush=True,
    )
    safe_sleep(stage=f"post-train:{method}")

    if train_status == "success":
        try:
            stream_subprocess(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "eval",
                    "accuracy",
                    "--base-config",
                    str(BASE_CONFIG_PATH),
                    "--config",
                    str(eval_override_path),
                ],
                stage=f"eval:{method}",
                log_path=eval_log_path,
            )
            parse_ok, exact_match, action_match = load_accuracy_metrics(report_path)
            eval_status = "success"
        except Exception as exc:
            eval_status = classify_failure(read_text_if_exists(eval_log_path))
            failure_reason = f"{type(exc).__name__}: {exc}"
            print(f"[warning] {method} 评估失败，分类={eval_status}，原因={failure_reason}", flush=True)
    else:
        print(f"[eval:{method}] 因训练未成功，跳过评估。", flush=True)

    safe_sleep(stage=f"post-eval:{method}")

    return {
        "Method": method,
        "训练状态": train_status,
        "评估状态": eval_status,
        "失败原因": failure_reason,
        "峰值显存(MB)": train_metrics["train_peak_vram_mb"],
        "峰值增量显存(MB)": train_metrics["train_peak_delta_vram_mb"],
        "训练耗时(s)": train_metrics["train_time_sec"],
        "Final Loss": train_metrics["final_loss"],
        "Min Loss": train_metrics["min_loss"],
        "Min Loss Step": train_metrics["min_loss_step"],
        "Parse OK": parse_ok,
        "Exact Match": exact_match,
        "Action Match": action_match,
        "模型输出目录": repo_relative_str(model_output_dir),
        "训练日志": repo_relative_str(train_log_path),
        "评估日志": repo_relative_str(eval_log_path),
        "评估报告": repo_relative_str(report_path),
        "训练配置": repo_relative_str(train_config_path),
    }


def normalize_series(series: pd.Series, *, higher_is_better: bool, fill_value: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([fill_value] * len(series), index=series.index, dtype=float)

    min_value = float(valid.min())
    max_value = float(valid.max())
    if math.isclose(min_value, max_value):
        normalized = pd.Series([1.0] * len(series), index=series.index, dtype=float)
        normalized[numeric.isna()] = fill_value
        return normalized

    if higher_is_better:
        normalized = (numeric - min_value) / (max_value - min_value)
    else:
        normalized = (max_value - numeric) / (max_value - min_value)
    return normalized.fillna(fill_value).clip(lower=0.0, upper=1.0)


def choose_memory_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    delta_series = pd.to_numeric(df["峰值增量显存(MB)"], errors="coerce")
    if delta_series.notna().any():
        return delta_series, "Peak Delta VRAM (MB)"
    return pd.to_numeric(df["峰值显存(MB)"], errors="coerce"), "Peak VRAM (MB)"


def plot_dual_axis_chart(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    memory_series, memory_label = choose_memory_series(df)
    action_match = pd.to_numeric(df["Action Match"], errors="coerce").fillna(0.0)

    valid_memory = memory_series.dropna()
    fallback_height = float(valid_memory.max()) * 1.1 if not valid_memory.empty else 1.0
    memory_for_plot = memory_series.fillna(fallback_height)
    bar_colors = ["#4C78A8" if pd.notna(value) else "#E45756" for value in memory_series]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    x = range(len(df))
    bars = ax1.bar(x, memory_for_plot, color=bar_colors, width=0.6, label=memory_label)
    ax1.set_xlabel("PEFT Method")
    ax1.set_ylabel(memory_label, color="#2F4B7C")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["Method"].tolist())
    ax1.tick_params(axis="y", labelcolor="#2F4B7C")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        list(x),
        action_match,
        color="#F28E2B",
        marker="o",
        linewidth=2.0,
        label="Action Match",
    )
    ax2.set_ylabel("Action Match", color="#A0512D")
    ax2.set_ylim(0, max(1.0, float(action_match.max()) * 1.1 if not action_match.empty else 1.0))
    ax2.tick_params(axis="y", labelcolor="#A0512D")

    for idx, bar in enumerate(bars):
        label = "Failed" if pd.isna(memory_series.iloc[idx]) else f"{int(memory_for_plot.iloc[idx])}"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for idx, value in enumerate(action_match):
        ax2.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#A0512D")

    plt.title("Exp3: PEFT Training Resource vs Action Match")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radar_chart(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    memory_series, _ = choose_memory_series(df)
    training_time = pd.to_numeric(df["训练耗时(s)"], errors="coerce")
    exact_match = pd.to_numeric(df["Exact Match"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    action_match = pd.to_numeric(df["Action Match"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    training_speed = training_time.apply(
        lambda value: 0.0 if pd.isna(value) or value <= 0 else 1.0 / float(value)
    )
    training_speed[memory_series.isna()] = 0.0

    memory_efficiency = normalize_series(memory_series, higher_is_better=False, fill_value=0.0)
    speed_score = normalize_series(training_speed, higher_is_better=True, fill_value=0.0)

    radar_df = pd.DataFrame(
        {
            "Method": df["Method"],
            "Memory Efficiency": memory_efficiency,
            "Training Speed": speed_score,
            "Exact Match": exact_match,
            "Action Match": action_match,
        }
    )

    categories = ["Memory Efficiency", "Training Speed", "Exact Match", "Action Match"]
    angles = [index / float(len(categories)) * 2 * math.pi for index in range(len(categories))]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.subplot(111, polar=True)

    color_cycle = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for idx, row in radar_df.iterrows():
        values = [float(row[category]) for category in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Method"], color=color_cycle[idx % len(color_cycle)])
        ax.fill(angles, values, alpha=0.12, color=color_cycle[idx % len(color_cycle)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title("Exp3: PEFT Overall Capability Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_figures(csv_path: Path) -> None:
    plot_dual_axis_chart(csv_path, DUAL_AXIS_FIG_PATH)
    plot_radar_chart(csv_path, RADAR_FIG_PATH)
    print(f"[plot] 已生成图表: {DUAL_AXIS_FIG_PATH}", flush=True)
    print(f"[plot] 已生成图表: {RADAR_FIG_PATH}", flush=True)


def main() -> None:
    ensure_runtime_dirs()

    results: list[dict[str, Any]] = []
    for method in METHODS:
        result = run_single_method(method)
        results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
        print(f"[result] 当前结果已保存到: {RESULTS_CSV_PATH}", flush=True)

    final_df = pd.read_csv(RESULTS_CSV_PATH)
    print("\n实验汇总结果：", flush=True)
    print(final_df.to_string(index=False), flush=True)

    plot_all_figures(RESULTS_CSV_PATH)


if __name__ == "__main__":
    main()
