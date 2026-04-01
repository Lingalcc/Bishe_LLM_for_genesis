#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.data_core.split_dataset import SplitDatasetConfig, run_split_dataset
from src.utils.config import load_merged_config
from src.utils.plotting import configure_report_matplotlib

configure_report_matplotlib(matplotlib)


def generate_exp_num_arr(minimum: int, maximum: int, step: int) -> list[int]:
    values: list[int] = []
    current = minimum
    while current <= maximum:
        values.append(current)
        current += step
    return values


DATA_SIZES = generate_exp_num_arr(1800, 3600, 200)
SLEEP_SECONDS = 10
FINETUNE_METHOD = "qlora"
LORA_RANK = 4
LORA_ALPHA = 32
BOUNDARY_GAIN_THRESHOLD = 0.01
BOUNDARY_CONSECUTIVE_POINTS = 2
EXPECTED_SOURCE_SIZE = 4000

BASE_CONFIG_PATH = REPO_ROOT / "configs/base.yaml"
SOURCE_DATASET_PATH = REPO_ROOT / "data_prepare/genesis_franka_toolcall_alpaca.json"
SPLIT_DIR = REPO_ROOT / "data_prepare/splits_4000_exp1_qlora_rank4"
TRAIN_JSON_PATH = SPLIT_DIR / "train.json"
VAL_JSON_PATH = SPLIT_DIR / "val.json"
TEST_JSON_PATH = SPLIT_DIR / "test.json"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"
DEFAULT_RUNNER_PYTHON = Path("/home/lin/miniconda3/envs/llm_genesis/bin/python")
DEFAULT_CONDA_ENV_NAME = "llm_genesis"

BASE_MODEL_PATH = REPO_ROOT / "model/Qwen_Qwen2.5-3B-Instruct"
OUTPUT_ROOT = REPO_ROOT / "output/exp1_data_scale_qlora_rank4"

TEMP_DIR = EXPERIMENT_DIR / ".cache/qlora_rank4"
TEMP_TRAIN_CONFIG_PATH = TEMP_DIR / "train_override.yaml"
TEMP_EVAL_CONFIG_PATH = TEMP_DIR / "eval_override.yaml"
FULL_TRAIN_SNAPSHOT_PATH = TEMP_DIR / "full_train_snapshot.json"
HF_CACHE_ROOT = REPO_ROOT / ".cache/huggingface"
HF_HOME_PATH = HF_CACHE_ROOT / "home"
HF_HUB_CACHE_PATH = HF_CACHE_ROOT / "hub"
HF_DATASETS_CACHE_PATH = HF_CACHE_ROOT / "datasets"
TRANSFORMERS_CACHE_PATH = HF_CACHE_ROOT / "transformers"

REPORTS_DIR = EXPERIMENT_DIR / "reports/qlora_rank4"
LOGS_DIR = EXPERIMENT_DIR / "logs/qlora_rank4"
RESULTS_CSV_PATH = REPORTS_DIR / "exp1_data_scale_qlora_rank4_results.csv"
CHART_PATH = REPORTS_DIR / "exp1_data_scale_qlora_rank4_dashboard.png"
BOUNDARY_SUMMARY_PATH = REPORTS_DIR / "exp1_data_scale_qlora_rank4_boundary.json"
PROGRESS_STATE_PATH = REPORTS_DIR / "progress_state.json"

TRAIN_TOTAL_STEPS_RE = re.compile(r"Total optimization steps = (?P<steps>\d+)")
TRAIN_LOSS_RE = re.compile(r"\{'loss': (?P<loss>[0-9.]+).*?'epoch': (?P<epoch>[0-9.]+)\}")
TRAIN_RUNTIME_RE = re.compile(r"'train_runtime': (?P<runtime>[0-9.]+)")
TRAIN_SUMMARY_TIME_RE = re.compile(r"\[finetune\]\s+time \(sec\)\s*:\s*(?P<value>[0-9.]+)")
TRAIN_SUMMARY_PEAK_VRAM_RE = re.compile(r"\[finetune\]\s+peak VRAM\s*:\s*(?P<value>[0-9.]+)\s+MB")
TRAIN_SUMMARY_PEAK_DELTA_VRAM_RE = re.compile(
    r"\[finetune\]\s+peak ΔVRAM\s*:\s*(?P<value>[0-9.]+)\s+MB"
)
EVAL_PROGRESS_RE = re.compile(
    r"Progress:\s*(?P<done>\d+)/(?P<total>\d+)\s+parse_ok=(?P<parse_ok>[0-9.]+)%\s+exact_match=(?P<exact>[0-9.]+)%\s+action_match=(?P<action>[0-9.]+)%"
)
CLI_METRIC_RE = re.compile(r"\[ok\]\s+(?P<name>exact match|action match)\s*:\s*(?P<count>\d+)\s+\((?P<rate>[0-9.]+)\)")
GENERIC_PERCENT_RE = re.compile(r"(?P<current>\d+)%\|")
TQDM_PROGRESS_RE = re.compile(
    r"(?P<percent>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[^<]+)<(?P<remaining>[^,]+),\s*(?P<speed>[^\]]+)\]"
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _path_to_repo_str(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_runner_python() -> Path:
    env_override = os.environ.get("LLM_GENESIS_RUN_PYTHON", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    if DEFAULT_RUNNER_PYTHON.exists():
        return DEFAULT_RUNNER_PYTHON
    return Path(sys.executable).resolve()


def _resolve_runner_prefix() -> list[str]:
    env_command = os.environ.get("LLM_GENESIS_RUN_COMMAND", "").strip()
    if env_command:
        return env_command.split()

    conda_bin = shutil.which("conda")
    if conda_bin:
        return [conda_bin, "run", "-n", DEFAULT_CONDA_ENV_NAME, "python"]

    return [str(_resolve_runner_python())]


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}, but got {type(rows).__name__}.")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _build_subprocess_env() -> dict[str, str]:
    for cache_dir in (
        HF_CACHE_ROOT,
        HF_HOME_PATH,
        HF_HUB_CACHE_PATH,
        HF_DATASETS_CACHE_PATH,
        TRANSFORMERS_CACHE_PATH,
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(HF_HOME_PATH)
    env["HF_HUB_CACHE"] = str(HF_HUB_CACHE_PATH)
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE_PATH)
    env["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE_PATH)
    env["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE_PATH)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache/matplotlib").resolve()))
    return env


def _ensure_split(force_resplit: bool = False) -> dict[str, Any]:
    if not SOURCE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_DATASET_PATH}")

    if (
        not force_resplit
        and TRAIN_JSON_PATH.exists()
        and VAL_JSON_PATH.exists()
        and TEST_JSON_PATH.exists()
        and SPLIT_METADATA_PATH.exists()
    ):
        return json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))

    metadata = run_split_dataset(
        SplitDatasetConfig(
            input_file=SOURCE_DATASET_PATH,
            out_dir=SPLIT_DIR,
            train_ratio=0.9,
            val_ratio=0.05,
            test_ratio=0.05,
            seed=42,
            train_name=TRAIN_JSON_PATH.name,
            val_name=VAL_JSON_PATH.name,
            test_name=TEST_JSON_PATH.name,
            metadata_name=SPLIT_METADATA_PATH.name,
            preserve_existing_splits=False,
        )
    )
    return metadata


def _prepare_train_config() -> Path:
    train_override = {
        "finetune": {
            "train": {
                "dry_run": False,
                "train_file": _path_to_repo_str(TRAIN_JSON_PATH),
                "val_file": _path_to_repo_str(VAL_JSON_PATH),
                "gpus": "",
            }
        }
    }
    _write_yaml(TEMP_TRAIN_CONFIG_PATH, train_override)
    return TEMP_TRAIN_CONFIG_PATH


def _prepare_eval_config(model_path: Path, report_path: Path) -> Path:
    eval_override = {
        "test": {
            "accuracy_eval": {
                "mode": "local",
                "test_file": _path_to_repo_str(TEST_JSON_PATH),
                "report_file": _path_to_repo_str(report_path),
                "model_path": _path_to_repo_str(model_path),
                "backend": "transformers",
                "temperature": 0.0,
                "trust_remote_code": True,
            }
        }
    }
    _write_yaml(TEMP_EVAL_CONFIG_PATH, eval_override)
    return TEMP_EVAL_CONFIG_PATH


def _resolve_effective_paths(
    train_config_path: Path,
    eval_config_path: Path,
) -> tuple[Path, Path, Path, Path]:
    merged_train_cfg = load_merged_config(
        base_config_path=BASE_CONFIG_PATH,
        override_config_path=train_config_path,
    )
    merged_eval_cfg = load_merged_config(
        base_config_path=BASE_CONFIG_PATH,
        override_config_path=eval_config_path,
    )

    train_file = Path(merged_train_cfg["finetune"]["train"]["train_file"])
    val_file = Path(merged_train_cfg["finetune"]["train"]["val_file"])
    test_file = Path(merged_eval_cfg["test"]["accuracy_eval"]["test_file"])
    report_file = Path(merged_eval_cfg["test"]["accuracy_eval"]["report_file"])

    resolved = []
    for path in (train_file, val_file, test_file, report_file):
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved.append(path)
    return tuple(resolved)  # type: ignore[return-value]


def _load_or_create_full_train_snapshot(train_file: Path) -> list[dict[str, Any]]:
    if FULL_TRAIN_SNAPSHOT_PATH.exists():
        return _load_json_list(FULL_TRAIN_SNAPSHOT_PATH)

    full_train_data = _load_json_list(train_file)
    _write_json(FULL_TRAIN_SNAPSHOT_PATH, full_train_data)
    return full_train_data


def _print_stage_message(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def _stream_subprocess(
    command: list[str],
    *,
    cwd: Path,
    stage: str,
    log_path: Path,
) -> None:
    _ensure_parent(log_path)
    _print_stage_message(stage, f"starting, full log -> {log_path}")
    _print_stage_message(stage, f"command: {' '.join(command)}")

    start_time = time.time()
    total_steps: int | None = None
    last_eval_progress: tuple[int, int] | None = None
    last_percent: int | None = None
    last_tqdm_step: tuple[int, int] | None = None
    last_status_ts = 0.0

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {_timestamp()} | stage={stage} =====\n")
        log_file.write(f"cwd: {cwd}\n")
        log_file.write(f"command: {' '.join(command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=_build_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                log_file.write(raw_line)
                log_file.flush()

                line = raw_line.strip()
                if not line:
                    continue

                matched_total_steps = TRAIN_TOTAL_STEPS_RE.search(line)
                if matched_total_steps:
                    total_steps = int(matched_total_steps.group("steps"))
                    _print_stage_message(stage, f"training plan: total_steps={total_steps}")
                    continue

                matched_loss = TRAIN_LOSS_RE.search(line)
                if matched_loss:
                    loss = float(matched_loss.group("loss"))
                    epoch = float(matched_loss.group("epoch"))
                    if total_steps is not None:
                        _print_stage_message(stage, f"update: loss={loss:.4f}, epoch={epoch:.2f}, planned_steps={total_steps}")
                    else:
                        _print_stage_message(stage, f"update: loss={loss:.4f}, epoch={epoch:.2f}")
                    continue

                matched_runtime = TRAIN_RUNTIME_RE.search(line)
                if matched_runtime:
                    runtime = float(matched_runtime.group("runtime"))
                    _print_stage_message(stage, f"finished training in {runtime:.1f}s")
                    continue

                matched_eval_progress = EVAL_PROGRESS_RE.search(line)
                if matched_eval_progress:
                    done = int(matched_eval_progress.group("done"))
                    total = int(matched_eval_progress.group("total"))
                    progress_key = (done, total)
                    if progress_key != last_eval_progress:
                        last_eval_progress = progress_key
                        _print_stage_message(
                            stage,
                            "progress: "
                            f"{done}/{total}, "
                            f"parse_ok={matched_eval_progress.group('parse_ok')}%, "
                            f"exact_match={matched_eval_progress.group('exact')}%, "
                            f"action_match={matched_eval_progress.group('action')}%",
                        )
                    continue

                matched_cli_metric = CLI_METRIC_RE.search(line)
                if matched_cli_metric:
                    metric_name = matched_cli_metric.group("name")
                    rate = float(matched_cli_metric.group("rate"))
                    _print_stage_message(stage, f"{metric_name}={rate:.4f}")
                    continue

                matched_tqdm = TQDM_PROGRESS_RE.search(line)
                if matched_tqdm:
                    step = int(matched_tqdm.group("step"))
                    total = int(matched_tqdm.group("total"))
                    progress_key = (step, total)
                    if progress_key != last_tqdm_step:
                        last_tqdm_step = progress_key
                        _print_stage_message(
                            stage,
                            "tqdm: "
                            f"{step}/{total} "
                            f"({matched_tqdm.group('percent')}%), "
                            f"elapsed={matched_tqdm.group('elapsed').strip()}, "
                            f"eta={matched_tqdm.group('remaining').strip()}, "
                            f"speed={matched_tqdm.group('speed').strip()}",
                        )
                    continue

                matched_percent = GENERIC_PERCENT_RE.search(line)
                if matched_percent:
                    current_percent = int(matched_percent.group("current"))
                    now = time.time()
                    if current_percent != last_percent and (current_percent % 10 == 0 or current_percent == 100):
                        last_percent = current_percent
                        if now - last_status_ts >= 1.0:
                            _print_stage_message(stage, f"progress bar: {current_percent}%")
                            last_status_ts = now
                    continue

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        except KeyboardInterrupt:
            _print_stage_message(stage, "interrupted, terminating subprocess...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _print_stage_message(stage, "subprocess did not exit in time, killing it...")
                process.kill()
                process.wait()
            raise
        finally:
            elapsed = time.time() - start_time
            log_file.write(f"\n===== {_timestamp()} | stage={stage} finished in {elapsed:.1f}s =====\n")
            log_file.flush()


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    stage: str,
    log_path: Path,
) -> None:
    try:
        _stream_subprocess(command, cwd=cwd, stage=stage, log_path=log_path)
    finally:
        print(f"[sleep] Waiting {SLEEP_SECONDS}s for GPU/CPU resources to settle...", flush=True)
        time.sleep(SLEEP_SECONDS)


def _extract_eval_metrics(report_path: Path) -> dict[str, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "parse_ok_rate": float(report.get("parse_ok_rate", 0.0)),
        "exact_match_rate": float(report.get("exact_match_rate", 0.0)),
        "action_match_rate": float(report.get("action_match_rate", 0.0)),
        "avg_latency_sec": float(report.get("avg_latency_sec", 0.0)),
        "avg_throughput_tps": float(report.get("avg_throughput_tps", 0.0)),
        "avg_peak_vram_mb": float(report.get("avg_peak_vram_mb", 0.0)),
        "max_peak_vram_mb": float(report.get("max_peak_vram_mb", 0.0)),
    }


def _extract_train_metrics(log_path: Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "train_time_sec": 0.0,
        "train_time_min": 0.0,
        "final_loss": 0.0,
        "min_loss": 0.0,
        "min_loss_epoch": 0.0,
        "train_peak_vram_mb": 0.0,
        "train_peak_delta_vram_mb": 0.0,
    }
    if not log_path.exists():
        return metrics

    losses: list[tuple[float, float]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        matched_loss = TRAIN_LOSS_RE.search(line)
        if matched_loss:
            losses.append(
                (
                    float(matched_loss.group("loss")),
                    float(matched_loss.group("epoch")),
                )
            )
            continue

        matched_time = TRAIN_SUMMARY_TIME_RE.search(line)
        if matched_time:
            metrics["train_time_sec"] = float(matched_time.group("value"))
            metrics["train_time_min"] = float(matched_time.group("value")) / 60.0
            continue

        matched_peak_vram = TRAIN_SUMMARY_PEAK_VRAM_RE.search(line)
        if matched_peak_vram:
            metrics["train_peak_vram_mb"] = float(matched_peak_vram.group("value"))
            continue

        matched_peak_delta_vram = TRAIN_SUMMARY_PEAK_DELTA_VRAM_RE.search(line)
        if matched_peak_delta_vram:
            metrics["train_peak_delta_vram_mb"] = float(matched_peak_delta_vram.group("value"))
            continue

    if losses:
        metrics["final_loss"] = losses[-1][0]
        best_loss, best_epoch = min(losses, key=lambda item: item[0])
        metrics["min_loss"] = best_loss
        metrics["min_loss_epoch"] = best_epoch

    return metrics


def _save_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    _ensure_parent(csv_path)
    fieldnames = [
        "data_size",
        "parse_ok_rate",
        "exact_match_rate",
        "action_match_rate",
        "avg_latency_sec",
        "avg_throughput_tps",
        "avg_peak_vram_mb",
        "max_peak_vram_mb",
        "train_time_sec",
        "train_time_min",
        "final_loss",
        "min_loss",
        "min_loss_epoch",
        "train_peak_vram_mb",
        "train_peak_delta_vram_mb",
        "model_output_dir",
        "eval_report_path",
        "train_log_path",
        "eval_log_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _load_progress_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "completed_sizes": [],
            "results": [],
            "last_started_size": None,
            "last_status": "not_started",
            "last_error": None,
            "updated_at": None,
        }

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"Expected a JSON object in {state_path}, but got {type(state).__name__}.")
    state.setdefault("completed_sizes", [])
    state.setdefault("results", [])
    state.setdefault("last_started_size", None)
    state.setdefault("last_status", "unknown")
    state.setdefault("last_error", None)
    state.setdefault("updated_at", None)
    return state


def _save_progress_state(state: dict[str, Any], state_path: Path) -> None:
    state["updated_at"] = _timestamp()
    _write_json(state_path, state)


def _upsert_result(results: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    data_size = int(row["data_size"])
    filtered = [item for item in results if int(item["data_size"]) != data_size]
    filtered.append(row)
    filtered.sort(key=lambda item: int(item["data_size"]))
    return filtered


def _find_plateau_size(
    results: list[dict[str, Any]],
    metric_key: str,
    *,
    threshold: float,
    consecutive_points: int,
) -> int | None:
    if len(results) < consecutive_points + 1:
        return None

    small_gain_count = 0
    previous_value = float(results[0][metric_key])
    for row in results[1:]:
        current_value = float(row[metric_key])
        gain = current_value - previous_value
        if gain <= threshold:
            small_gain_count += 1
            if small_gain_count >= consecutive_points:
                return int(row["data_size"])
        else:
            small_gain_count = 0
        previous_value = current_value
    return None


def _build_boundary_summary(
    results: list[dict[str, Any]],
    *,
    requested_sizes: list[int],
    max_train_size_available: int,
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not results:
        return {
            "method": FINETUNE_METHOD,
            "lora_rank": LORA_RANK,
            "requested_sizes": requested_sizes,
            "evaluated_sizes": [],
            "max_train_size_available": max_train_size_available,
            "split_dir": str(SPLIT_DIR),
            "split_metadata": split_metadata,
            "boundary_note": "尚无已完成结果，无法计算边界摘要。",
        }

    best_exact = max(results, key=lambda row: float(row["exact_match_rate"]))
    best_action = max(results, key=lambda row: float(row["action_match_rate"]))
    best_parse = max(results, key=lambda row: float(row["parse_ok_rate"]))

    summary = {
        "method": FINETUNE_METHOD,
        "lora_rank": LORA_RANK,
        "requested_sizes": requested_sizes,
        "evaluated_sizes": [int(row["data_size"]) for row in results],
        "max_train_size_available": max_train_size_available,
        "split_dir": str(SPLIT_DIR),
        "split_metadata": split_metadata,
        "best_exact_match": {
            "data_size": int(best_exact["data_size"]),
            "rate": float(best_exact["exact_match_rate"]),
        },
        "best_action_match": {
            "data_size": int(best_action["data_size"]),
            "rate": float(best_action["action_match_rate"]),
        },
        "best_parse_ok": {
            "data_size": int(best_parse["data_size"]),
            "rate": float(best_parse["parse_ok_rate"]),
        },
        "exact_match_plateau_size": _find_plateau_size(
            results,
            "exact_match_rate",
            threshold=BOUNDARY_GAIN_THRESHOLD,
            consecutive_points=BOUNDARY_CONSECUTIVE_POINTS,
        ),
        "action_match_plateau_size": _find_plateau_size(
            results,
            "action_match_rate",
            threshold=BOUNDARY_GAIN_THRESHOLD,
            consecutive_points=BOUNDARY_CONSECUTIVE_POINTS,
        ),
        "boundary_rule": {
            "gain_threshold": BOUNDARY_GAIN_THRESHOLD,
            "consecutive_points": BOUNDARY_CONSECUTIVE_POINTS,
            "description": "若相邻数据规模的准确率提升连续两次不超过 0.01，则记为进入平台边界。",
        },
    }
    return summary


def _save_dashboard(results: list[dict[str, Any]], chart_path: Path) -> None:
    _ensure_parent(chart_path)
    sizes = [int(row["data_size"]) for row in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Exp1 扩展：QLoRA Rank=4 在不同数据规模下的效果与资源变化", fontsize=16, y=0.98)

    def _plot(ax: Any, key: str, title: str, ylabel: str, *, ylim: tuple[float, float] | None = None) -> None:
        values = [float(row[key]) for row in results]
        ax.plot(sizes, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("训练样本数")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sizes)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.35)

    def _plot_multi(
        ax: Any,
        series: list[tuple[str, str, str]],
        title: str,
        ylabel: str,
        *,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        for key, label, marker in series:
            values = [float(row[key]) for row in results]
            ax.plot(sizes, values, marker=marker, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("训练样本数")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sizes)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    _plot(axes[0, 0], "parse_ok_rate", "Parse OK Rate", "比例", ylim=(0.0, 1.0))
    _plot(axes[0, 1], "exact_match_rate", "Exact Match Rate", "比例", ylim=(0.0, 1.0))
    _plot(axes[0, 2], "action_match_rate", "Action Match Rate", "比例", ylim=(0.0, 1.0))
    _plot_multi(
        axes[1, 0],
        [
            ("avg_peak_vram_mb", "评测平均峰值显存", "o"),
            ("max_peak_vram_mb", "评测最大峰值显存", "s"),
        ],
        "评测显存",
        "MB",
    )
    _plot_multi(
        axes[1, 1],
        [
            ("train_time_min", "训练时长", "o"),
            ("avg_latency_sec", "评测平均延迟", "s"),
        ],
        "时延与训练耗时",
        "分钟 / 秒",
    )
    _plot_multi(
        axes[1, 2],
        [
            ("final_loss", "最终 loss", "o"),
            ("min_loss", "最小 loss", "s"),
        ],
        "训练损失",
        "Loss",
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(chart_path, dpi=220)
    plt.close(fig)


def _build_model_output_dir(data_size: int) -> Path:
    return OUTPUT_ROOT / f"size_{data_size}"


def _build_eval_report_path(data_size: int) -> Path:
    return REPORTS_DIR / f"accuracy_report_size_{data_size}.json"


def main(force_resplit: bool = False) -> None:
    split_metadata = _ensure_split(force_resplit=force_resplit)
    runner_prefix = _resolve_runner_prefix()
    source_num_samples = int(split_metadata.get("source", {}).get("num_samples", 0))
    if source_num_samples != EXPECTED_SOURCE_SIZE:
        print(
            f"[warn] Source dataset size is {source_num_samples}, expected {EXPECTED_SOURCE_SIZE}. "
            "The experiment will continue with the current dataset snapshot.",
            flush=True,
        )

    train_config_path = _prepare_train_config()
    bootstrap_eval_config_path = _prepare_eval_config(_build_model_output_dir(DATA_SIZES[0]), _build_eval_report_path(DATA_SIZES[0]))
    effective_train_file, effective_val_file, effective_test_file, _ = _resolve_effective_paths(
        train_config_path,
        bootstrap_eval_config_path,
    )

    for required_path in (effective_train_file, effective_val_file, effective_test_file):
        if not required_path.exists():
            raise FileNotFoundError(f"Required split file not found: {required_path}")

    original_train_text = effective_train_file.read_text(encoding="utf-8")
    full_train_data = _load_or_create_full_train_snapshot(effective_train_file)
    total_train_samples = len(full_train_data)
    effective_sizes = [size for size in DATA_SIZES if size <= total_train_samples]
    skipped_sizes = [size for size in DATA_SIZES if size > total_train_samples]

    if not effective_sizes:
        raise ValueError(
            f"Configured sizes {DATA_SIZES} exceed available train samples {total_train_samples}."
        )

    print(f"[info] Experiment dir: {EXPERIMENT_DIR}", flush=True)
    print(f"[info] Source dataset path: {SOURCE_DATASET_PATH}", flush=True)
    print(f"[info] Source dataset size: {source_num_samples}", flush=True)
    print(f"[info] Split dir: {SPLIT_DIR}", flush=True)
    print(f"[info] Train/val/test: {effective_train_file} | {effective_val_file} | {effective_test_file}", flush=True)
    print(f"[info] Train set size after split: {total_train_samples}", flush=True)
    print(f"[info] Running data scales: {effective_sizes}", flush=True)
    if skipped_sizes:
        print(f"[warn] Skipped data scales beyond train size: {skipped_sizes}", flush=True)
    print(
        f"[info] Finetune method: {FINETUNE_METHOD}, lora_rank={LORA_RANK}, lora_alpha={LORA_ALPHA}",
        flush=True,
    )
    print(f"[info] Base model path: {BASE_MODEL_PATH}", flush=True)
    print(f"[info] Runner command: {' '.join(runner_prefix)}", flush=True)
    print(f"[info] Output root: {OUTPUT_ROOT}", flush=True)
    print(f"[info] Reports dir: {REPORTS_DIR}", flush=True)
    print(f"[info] Logs dir: {LOGS_DIR}", flush=True)
    print(f"[info] Progress state: {PROGRESS_STATE_PATH}", flush=True)

    state = _load_progress_state(PROGRESS_STATE_PATH)
    results: list[dict[str, Any]] = sorted(
        state.get("results", []),
        key=lambda item: int(item["data_size"]),
    )
    completed_sizes = {int(size) for size in state.get("completed_sizes", [])}
    remaining_sizes = [size for size in effective_sizes if size not in completed_sizes]

    if completed_sizes:
        print(f"[resume] Completed sizes found: {sorted(completed_sizes)}", flush=True)
    if state.get("last_status") in {"interrupted", "failed"} and state.get("last_started_size") is not None:
        print(
            "[resume] "
            f"Last run ended with status={state['last_status']} at size={state['last_started_size']}. "
            "The script will continue from the first unfinished size.",
            flush=True,
        )
        if state.get("last_error"):
            print(f"[resume] Last error: {state['last_error']}", flush=True)
    if not remaining_sizes:
        print("[resume] All configured data scales are already completed.", flush=True)
        if results:
            _save_csv(results, RESULTS_CSV_PATH)
            _save_dashboard(results, CHART_PATH)
            boundary_summary = _build_boundary_summary(
                results,
                requested_sizes=DATA_SIZES,
                max_train_size_available=total_train_samples,
                split_metadata=split_metadata,
            )
            _write_json(BOUNDARY_SUMMARY_PATH, boundary_summary)
        return

    try:
        for data_size in remaining_sizes:
            run_tag = f"size_{data_size}"
            model_output_dir = _build_model_output_dir(data_size)
            eval_report_path = _build_eval_report_path(data_size)
            train_log_path = LOGS_DIR / f"{run_tag}_finetune.log"
            eval_log_path = LOGS_DIR / f"{run_tag}_eval.log"

            state["last_started_size"] = data_size
            state["last_status"] = "running"
            state["last_error"] = None
            _save_progress_state(state, PROGRESS_STATE_PATH)

            subset = full_train_data[:data_size]
            print(f"\n[dataset] Writing first {data_size} samples to {effective_train_file}", flush=True)
            _write_json(effective_train_file, subset)

            if model_output_dir.exists():
                print(f"[cleanup] Removing previous output for {run_tag}: {model_output_dir}", flush=True)
                shutil.rmtree(model_output_dir)

            eval_config_path = _prepare_eval_config(model_output_dir, eval_report_path)

            finetune_cmd = [
                *runner_prefix,
                "cli.py",
                "finetune",
                "start",
                "--base-config",
                str(BASE_CONFIG_PATH),
                "--config",
                str(train_config_path),
                "--finetune-method",
                FINETUNE_METHOD,
                f"model_name_or_path={BASE_MODEL_PATH}",
                f"output_dir={model_output_dir}",
                f"lora_rank={LORA_RANK}",
                f"lora_alpha={LORA_ALPHA}",
                "preprocessing_num_workers=1",
                "dataloader_num_workers=0",
            ]
            _run_subprocess(
                finetune_cmd,
                cwd=REPO_ROOT,
                stage=f"train/{run_tag}",
                log_path=train_log_path,
            )

            eval_cmd = [
                *runner_prefix,
                "cli.py",
                "eval",
                "accuracy",
                "--base-config",
                str(BASE_CONFIG_PATH),
                "--config",
                str(eval_config_path),
            ]
            _run_subprocess(
                eval_cmd,
                cwd=REPO_ROOT,
                stage=f"eval/{run_tag}",
                log_path=eval_log_path,
            )

            eval_metrics = _extract_eval_metrics(eval_report_path)
            train_metrics = _extract_train_metrics(train_log_path)
            row = {
                "data_size": data_size,
                "model_output_dir": str(model_output_dir),
                "eval_report_path": str(eval_report_path),
                "train_log_path": str(train_log_path),
                "eval_log_path": str(eval_log_path),
                **eval_metrics,
                **train_metrics,
            }
            results = _upsert_result(results, row)
            completed_sizes.add(data_size)
            state["results"] = results
            state["completed_sizes"] = sorted(completed_sizes)
            state["last_status"] = "completed"
            state["last_error"] = None
            _save_progress_state(state, PROGRESS_STATE_PATH)
            _save_csv(results, RESULTS_CSV_PATH)
            _save_dashboard(results, CHART_PATH)
            boundary_summary = _build_boundary_summary(
                results,
                requested_sizes=DATA_SIZES,
                max_train_size_available=total_train_samples,
                split_metadata=split_metadata,
            )
            _write_json(BOUNDARY_SUMMARY_PATH, boundary_summary)
            print(
                "[result] "
                f"size={data_size}, "
                f"exact_match_rate={row['exact_match_rate']:.4f}, "
                f"action_match_rate={row['action_match_rate']:.4f}, "
                f"train_log={train_log_path.name}, "
                f"eval_log={eval_log_path.name}",
                flush=True,
            )

        print(f"\n[done] CSV saved to: {RESULTS_CSV_PATH}", flush=True)
        print(f"[done] Dashboard saved to: {CHART_PATH}", flush=True)
        print(f"[done] Boundary summary saved to: {BOUNDARY_SUMMARY_PATH}", flush=True)
    except KeyboardInterrupt:
        state["results"] = results
        state["completed_sizes"] = sorted(completed_sizes)
        state["last_status"] = "interrupted"
        state["last_error"] = "KeyboardInterrupt"
        _save_progress_state(state, PROGRESS_STATE_PATH)
        print(
            "\n[interrupt] Experiment interrupted. Progress has been saved and can be resumed "
            "by rerunning the same script.",
            flush=True,
        )
    except Exception as exc:
        state["results"] = results
        state["completed_sizes"] = sorted(completed_sizes)
        state["last_status"] = "failed"
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        _save_progress_state(state, PROGRESS_STATE_PATH)
        print(
            "\n[error] Experiment stopped due to an exception. Progress has been saved and "
            "the next run will continue from the first unfinished size.",
            flush=True,
        )
        raise
    finally:
        effective_train_file.write_text(original_train_text, encoding="utf-8")
        print(f"\n[restore] Restored original train split snapshot: {effective_train_file}", flush=True)


if __name__ == "__main__":
    main(force_resplit="--force-resplit" in sys.argv)
