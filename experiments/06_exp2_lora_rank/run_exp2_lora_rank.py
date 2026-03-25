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

from src.utils.config import load_merged_config

LORA_RANKS = [4, 8, 16, 32, 64]
TRAIN_SUBSET_SIZE = 600
SLEEP_SECONDS = 10
FINETUNE_METHOD = "lora"

BASE_CONFIG_PATH = REPO_ROOT / "configs/base.yaml"
TRAIN_JSON_PATH = REPO_ROOT / "data_prepare/splits/train.json"
BASE_MODEL_PATH = REPO_ROOT / "model/Qwen_Qwen2.5-3B-Instruct"
OUTPUT_DIR_ROOT = REPO_ROOT / "output"

TEMP_DIR = EXPERIMENT_DIR / ".cache"
TEMP_TRAIN_CONFIG_PATH = TEMP_DIR / "train_override.yaml"
TEMP_EVAL_CONFIG_PATH = TEMP_DIR / "eval_override.yaml"
FULL_TRAIN_SNAPSHOT_PATH = TEMP_DIR / "full_train_snapshot.json"

REPORTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
RESULTS_CSV_PATH = REPORTS_DIR / "exp2_lora_rank_results.csv"
CHART_PATH = REPORTS_DIR / "exp2_lora_rank_dashboard.png"
PROGRESS_STATE_PATH = REPORTS_DIR / "progress_state.json"

TRAIN_TOTAL_STEPS_RE = re.compile(r"Total optimization steps = (?P<steps>\d+)")
TRAIN_LOSS_RE = re.compile(r"\{'loss': (?P<loss>[0-9.]+).*?'epoch': (?P<epoch>[0-9.]+)\}")
TRAIN_RUNTIME_RE = re.compile(r"'train_runtime': (?P<runtime>[0-9.]+)")
TRAIN_SUMMARY_TIME_RE = re.compile(r"\[finetune\]\s+time \(sec\)\s*:\s*(?P<value>[0-9.]+)")
TRAIN_SUMMARY_FINAL_LOSS_RE = re.compile(r"\[finetune\]\s+final loss\s*:\s*(?P<value>[0-9.]+)")
TRAIN_SUMMARY_MIN_LOSS_RE = re.compile(
    r"\[finetune\]\s+min loss\s*:\s*(?P<loss>[0-9.]+)\s+\(step\s+(?P<step>\d+)\)"
)
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


def _prepare_train_config() -> Path:
    train_override = {
        "finetune": {
            "train": {
                "dry_run": False,
                "train_file": _path_to_repo_str(TRAIN_JSON_PATH),
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


def _resolve_effective_train_file(train_config_path: Path) -> Path:
    merged_train_cfg = load_merged_config(
        base_config_path=BASE_CONFIG_PATH,
        override_config_path=train_config_path,
    )
    train_file = Path(merged_train_cfg["finetune"]["train"]["train_file"])
    if not train_file.is_absolute():
        train_file = REPO_ROOT / train_file
    return train_file


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
    last_loss_line: str | None = None
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
                    last_loss_line = f"loss={loss:.4f}, epoch={epoch:.2f}"
                    if total_steps is not None:
                        _print_stage_message(stage, f"update: {last_loss_line}, planned_steps={total_steps}")
                    else:
                        _print_stage_message(stage, f"update: {last_loss_line}")
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
        "min_loss_step": 0,
        "train_peak_vram_mb": 0.0,
        "train_peak_delta_vram_mb": 0.0,
    }
    if not log_path.exists():
        return metrics

    for line in log_path.read_text(encoding="utf-8").splitlines():
        matched_time = TRAIN_SUMMARY_TIME_RE.search(line)
        if matched_time:
            metrics["train_time_sec"] = float(matched_time.group("value"))
            metrics["train_time_min"] = float(matched_time.group("value")) / 60.0
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
            continue

    return metrics


def _save_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    _ensure_parent(csv_path)
    fieldnames = [
        "rank",
        "train_samples",
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
        "min_loss_step",
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
            "completed_ranks": [],
            "results": [],
            "last_started_rank": None,
            "last_status": "not_started",
            "last_error": None,
            "updated_at": None,
        }

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"Expected a JSON object in {state_path}, but got {type(state).__name__}.")
    state.setdefault("completed_ranks", [])
    state.setdefault("results", [])
    state.setdefault("last_started_rank", None)
    state.setdefault("last_status", "unknown")
    state.setdefault("last_error", None)
    state.setdefault("updated_at", None)
    return state


def _save_progress_state(state: dict[str, Any], state_path: Path) -> None:
    state["updated_at"] = _timestamp()
    _write_json(state_path, state)


def _upsert_result(results: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    rank = int(row["rank"])
    filtered = [item for item in results if int(item["rank"]) != rank]
    filtered.append(row)
    filtered.sort(key=lambda item: int(item["rank"]))
    return filtered


def _save_dashboard(results: list[dict[str, Any]], chart_path: Path) -> None:
    _ensure_parent(chart_path)
    ranks = [int(row["rank"]) for row in results]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        "Exp2: Impact of LoRA Rank on Qwen2.5-3B (Train Size = 600)",
        fontsize=16,
        y=0.98,
    )

    def _plot(ax: Any, key: str, title: str, ylabel: str, *, ylim: tuple[float, float] | None = None) -> None:
        values = [float(row[key]) for row in results]
        ax.plot(ranks, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ranks)
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
            ax.plot(ranks, values, marker=marker, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ranks)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    _plot(axes[0, 0], "parse_ok_rate", "Parse OK Rate", "Rate", ylim=(0.0, 1.0))
    _plot(axes[0, 1], "exact_match_rate", "Exact Match Rate", "Rate", ylim=(0.0, 1.0))
    _plot(axes[0, 2], "action_match_rate", "Action Match Rate", "Rate", ylim=(0.0, 1.0))
    _plot(axes[1, 0], "avg_latency_sec", "Average Latency", "Seconds")
    _plot(axes[1, 1], "avg_throughput_tps", "Average Throughput", "Tokens / Second")
    _plot_multi(
        axes[1, 2],
        [
            ("avg_peak_vram_mb", "Avg Peak VRAM", "o"),
            ("max_peak_vram_mb", "Max Peak VRAM", "s"),
        ],
        "Evaluation VRAM",
        "MB",
    )
    _plot_multi(
        axes[2, 0],
        [
            ("final_loss", "Final Loss", "o"),
            ("min_loss", "Min Loss", "s"),
        ],
        "Training Loss",
        "Loss",
    )
    _plot(axes[2, 1], "train_time_min", "Training Time", "Minutes")
    _plot_multi(
        axes[2, 2],
        [
            ("train_peak_vram_mb", "Peak VRAM", "o"),
            ("train_peak_delta_vram_mb", "Peak Delta VRAM", "s"),
        ],
        "Training VRAM",
        "MB",
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(chart_path, dpi=220)
    plt.close(fig)


def _build_model_output_dir(rank: int) -> Path:
    return OUTPUT_DIR_ROOT / f"qwen2.5-3b-genesis-lora-rank-{rank}"


def _build_eval_report_path(rank: int) -> Path:
    return REPORTS_DIR / f"accuracy_report_rank_{rank}.json"


def main() -> None:
    train_config_path = _prepare_train_config()
    effective_train_file = _resolve_effective_train_file(train_config_path)

    if not effective_train_file.exists():
        raise FileNotFoundError(
            f"Training split not found: {effective_train_file}. "
            "Please ensure train.json has been generated before running this script."
        )

    original_train_text = effective_train_file.read_text(encoding="utf-8")
    full_train_data = _load_or_create_full_train_snapshot(effective_train_file)
    total_samples = len(full_train_data)
    if total_samples < TRAIN_SUBSET_SIZE:
        raise ValueError(
            f"Configured train subset size is {TRAIN_SUBSET_SIZE}, "
            f"but snapshot only contains {total_samples} rows."
        )

    subset = full_train_data[:TRAIN_SUBSET_SIZE]
    print(f"[info] Experiment dir: {EXPERIMENT_DIR}", flush=True)
    print(f"[info] Full train set size: {total_samples}", flush=True)
    print(f"[info] Fixed train subset size: {TRAIN_SUBSET_SIZE}", flush=True)
    print(f"[info] Running LoRA ranks: {LORA_RANKS}", flush=True)
    print(f"[info] Base model path: {BASE_MODEL_PATH}", flush=True)
    print(f"[info] Logs dir: {LOGS_DIR}", flush=True)
    print(f"[info] Progress state: {PROGRESS_STATE_PATH}", flush=True)

    state = _load_progress_state(PROGRESS_STATE_PATH)
    results: list[dict[str, Any]] = sorted(
        state.get("results", []),
        key=lambda item: int(item["rank"]),
    )
    completed_ranks = {int(rank) for rank in state.get("completed_ranks", [])}
    remaining_ranks = [rank for rank in LORA_RANKS if rank not in completed_ranks]

    if completed_ranks:
        print(f"[resume] Completed ranks found: {sorted(completed_ranks)}", flush=True)
    if state.get("last_status") in {"interrupted", "failed"} and state.get("last_started_rank") is not None:
        print(
            "[resume] "
            f"Last run ended with status={state['last_status']} at rank={state['last_started_rank']}. "
            "The script will continue from the first unfinished rank.",
            flush=True,
        )
        if state.get("last_error"):
            print(f"[resume] Last error: {state['last_error']}", flush=True)
    if not remaining_ranks:
        print("[resume] All configured LoRA ranks are already completed.", flush=True)
        if results:
            _save_csv(results, RESULTS_CSV_PATH)
            _save_dashboard(results, CHART_PATH)
        return

    try:
        for rank in remaining_ranks:
            run_tag = f"rank_{rank}"
            train_log_path = LOGS_DIR / f"{run_tag}_finetune.log"
            eval_log_path = LOGS_DIR / f"{run_tag}_eval.log"
            model_output_dir = _build_model_output_dir(rank)
            eval_report_path = _build_eval_report_path(rank)
            eval_config_path = _prepare_eval_config(model_output_dir, eval_report_path)

            state["last_started_rank"] = rank
            state["last_status"] = "running"
            state["last_error"] = None
            _save_progress_state(state, PROGRESS_STATE_PATH)

            print(
                f"\n[dataset] Writing first {TRAIN_SUBSET_SIZE} samples to {effective_train_file}",
                flush=True,
            )
            _write_json(effective_train_file, subset)

            if model_output_dir.exists():
                print(f"[cleanup] Removing previous finetuned output: {model_output_dir}", flush=True)
                if model_output_dir.is_dir():
                    shutil.rmtree(model_output_dir)
                else:
                    model_output_dir.unlink()

            finetune_cmd = [
                sys.executable,
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
                f"lora_rank={rank}",
            ]
            _run_subprocess(
                finetune_cmd,
                cwd=REPO_ROOT,
                stage=f"train/{run_tag}",
                log_path=train_log_path,
            )

            eval_cmd = [
                sys.executable,
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

            train_metrics = _extract_train_metrics(train_log_path)
            eval_metrics = _extract_eval_metrics(eval_report_path)
            row = {
                "rank": rank,
                "train_samples": TRAIN_SUBSET_SIZE,
                "parse_ok_rate": eval_metrics["parse_ok_rate"],
                "exact_match_rate": eval_metrics["exact_match_rate"],
                "action_match_rate": eval_metrics["action_match_rate"],
                "avg_latency_sec": eval_metrics["avg_latency_sec"],
                "avg_throughput_tps": eval_metrics["avg_throughput_tps"],
                "avg_peak_vram_mb": eval_metrics["avg_peak_vram_mb"],
                "max_peak_vram_mb": eval_metrics["max_peak_vram_mb"],
                "train_time_sec": float(train_metrics["train_time_sec"]),
                "train_time_min": float(train_metrics["train_time_min"]),
                "final_loss": float(train_metrics["final_loss"]),
                "min_loss": float(train_metrics["min_loss"]),
                "min_loss_step": int(train_metrics["min_loss_step"]),
                "train_peak_vram_mb": float(train_metrics["train_peak_vram_mb"]),
                "train_peak_delta_vram_mb": float(train_metrics["train_peak_delta_vram_mb"]),
                "model_output_dir": _path_to_repo_str(model_output_dir),
                "eval_report_path": _path_to_repo_str(eval_report_path),
                "train_log_path": _path_to_repo_str(train_log_path),
                "eval_log_path": _path_to_repo_str(eval_log_path),
            }
            results = _upsert_result(results, row)
            completed_ranks.add(rank)
            state["results"] = results
            state["completed_ranks"] = sorted(completed_ranks)
            state["last_status"] = "completed"
            state["last_error"] = None
            _save_progress_state(state, PROGRESS_STATE_PATH)
            _save_csv(results, RESULTS_CSV_PATH)
            _save_dashboard(results, CHART_PATH)
            print(
                "[result] "
                f"rank={rank}, "
                f"parse_ok_rate={row['parse_ok_rate']:.4f}, "
                f"exact_match_rate={row['exact_match_rate']:.4f}, "
                f"action_match_rate={row['action_match_rate']:.4f}, "
                f"avg_latency_sec={row['avg_latency_sec']:.3f}, "
                f"avg_throughput_tps={row['avg_throughput_tps']:.2f}, "
                f"train_time_min={row['train_time_min']:.2f}, "
                f"train_log={train_log_path.name}, "
                f"eval_log={eval_log_path.name}",
                flush=True,
            )

        print(f"\n[done] CSV saved to: {RESULTS_CSV_PATH}", flush=True)
        print(f"[done] Dashboard saved to: {CHART_PATH}", flush=True)
    except KeyboardInterrupt:
        state["results"] = results
        state["completed_ranks"] = sorted(completed_ranks)
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
        state["completed_ranks"] = sorted(completed_ranks)
        state["last_status"] = "failed"
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        _save_progress_state(state, PROGRESS_STATE_PATH)
        print(
            "\n[error] Experiment stopped due to an exception. Progress has been saved and "
            "the next run will continue from the first unfinished rank.",
            flush=True,
        )
        raise
    finally:
        effective_train_file.write_text(original_train_text, encoding="utf-8")
        print(f"\n[restore] Restored original train split: {effective_train_file}", flush=True)


if __name__ == "__main__":
    main()
