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

# =========================
# High-Extensibility Config
# =========================
def generate_exp_num_arr(min,max,step):
    arr = []
    num = min
    while num <= max:
        arr.append(num)
        num += step
    return arr
DATA_SIZES = generate_exp_num_arr(200, 1600, 200)
SLEEP_SECONDS = 10
FINETUNE_METHOD = "lora"

BASE_CONFIG_PATH = REPO_ROOT / "configs/base.yaml"
TRAIN_JSON_PATH = REPO_ROOT / "data_prepare/splits/train.json"
BASE_MODEL_PATH = REPO_ROOT / "model/Qwen_Qwen2.5-3B-Instruct"
FINETUNED_MODEL_PATH = REPO_ROOT / "output/qwen2.5-3b-genesis-lora"

TEMP_DIR = EXPERIMENT_DIR / ".cache"
TEMP_TRAIN_CONFIG_PATH = TEMP_DIR / "train_override.yaml"
TEMP_EVAL_CONFIG_PATH = TEMP_DIR / "eval_override.yaml"

REPORTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
ACCURACY_REPORT_PATH = REPORTS_DIR / "accuracy_report_exp1_data_scale.json"
RESULTS_CSV_PATH = REPORTS_DIR / "exp1_data_scale_results.csv"
CHART_PATH = REPORTS_DIR / "exp1_data_scale_chart.png"
PROGRESS_STATE_PATH = REPORTS_DIR / "progress_state.json"

TRAIN_TOTAL_STEPS_RE = re.compile(r"Total optimization steps = (?P<steps>\d+)")
TRAIN_LOSS_RE = re.compile(
    r"\{'loss': (?P<loss>[0-9.]+).*?'epoch': (?P<epoch>[0-9.]+)\}"
)
TRAIN_RUNTIME_RE = re.compile(r"'train_runtime': (?P<runtime>[0-9.]+)")
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


def _prepare_temp_configs() -> tuple[Path, Path]:
    train_override = {
        "finetune": {
            "train": {
                "dry_run": False,
                "train_file": str(TRAIN_JSON_PATH.relative_to(REPO_ROOT)),
            }
        }
    }
    eval_override = {
        "test": {
            "accuracy_eval": {
                "mode": "local",
                "report_file": str(ACCURACY_REPORT_PATH.relative_to(REPO_ROOT)),
                "model_path": str(FINETUNED_MODEL_PATH.relative_to(REPO_ROOT)),
                "backend": "transformers",
                "temperature": 0.0,
                "trust_remote_code": True,
            }
        }
    }
    _write_yaml(TEMP_TRAIN_CONFIG_PATH, train_override)
    _write_yaml(TEMP_EVAL_CONFIG_PATH, eval_override)
    return TEMP_TRAIN_CONFIG_PATH, TEMP_EVAL_CONFIG_PATH


def _resolve_effective_paths(
    train_config_path: Path,
    eval_config_path: Path,
) -> tuple[Path, Path]:
    merged_train_cfg = load_merged_config(
        base_config_path=BASE_CONFIG_PATH,
        override_config_path=train_config_path,
    )
    merged_eval_cfg = load_merged_config(
        base_config_path=BASE_CONFIG_PATH,
        override_config_path=eval_config_path,
    )

    train_file = Path(merged_train_cfg["finetune"]["train"]["train_file"])
    report_file = Path(merged_eval_cfg["test"]["accuracy_eval"]["report_file"])

    if not train_file.is_absolute():
        train_file = REPO_ROOT / train_file
    if not report_file.is_absolute():
        report_file = REPO_ROOT / report_file
    return train_file, report_file


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


def _extract_metrics(report_path: Path) -> dict[str, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "exact_match_rate": float(report.get("exact_match_rate", 0.0)),
        "action_match_rate": float(report.get("action_match_rate", 0.0)),
    }


def _save_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    _ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["data_size", "exact_match_rate", "action_match_rate"],
        )
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


def _save_chart(results: list[dict[str, Any]], chart_path: Path) -> None:
    _ensure_parent(chart_path)
    sizes = [int(row["data_size"]) for row in results]
    exact_rates = [float(row["exact_match_rate"]) for row in results]
    action_rates = [float(row["action_match_rate"]) for row in results]

    plt.figure(figsize=(9, 5.5))
    plt.plot(sizes, exact_rates, marker="o", linewidth=2, label="Exact Match Rate")
    plt.plot(sizes, action_rates, marker="s", linewidth=2, label="Action Match Rate")
    plt.xlabel("Training Data Size")
    plt.ylabel("Accuracy")
    plt.title("Exp1: Impact of Training Data Scale on Fine-Tuning Accuracy")
    plt.xticks(sizes)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()


def main() -> None:
    train_config_path, eval_config_path = _prepare_temp_configs()
    effective_train_file, effective_report_file = _resolve_effective_paths(
        train_config_path,
        eval_config_path,
    )

    if not effective_train_file.exists():
        raise FileNotFoundError(
            f"Training split not found: {effective_train_file}. "
            "Please ensure the full train.json has been generated before running this script."
        )

    original_train_text = effective_train_file.read_text(encoding="utf-8")
    full_train_data = _load_json_list(effective_train_file)
    total_samples = len(full_train_data)
    effective_sizes = [size for size in DATA_SIZES if size <= total_samples]
    if not effective_sizes:
        effective_sizes = [total_samples]

    print(f"[info] Experiment dir: {EXPERIMENT_DIR}", flush=True)
    print(f"[info] Full train set size: {total_samples}", flush=True)
    print(f"[info] Running data scales: {effective_sizes}", flush=True)
    print(f"[info] Base model path: {BASE_MODEL_PATH}", flush=True)
    print(f"[info] Finetuned output path: {FINETUNED_MODEL_PATH}", flush=True)
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
            _save_chart(results, CHART_PATH)
        return

    try:
        for data_size in remaining_sizes:
            run_tag = f"size_{data_size}"
            train_log_path = LOGS_DIR / f"{run_tag}_finetune.log"
            eval_log_path = LOGS_DIR / f"{run_tag}_eval.log"

            state["last_started_size"] = data_size
            state["last_status"] = "running"
            state["last_error"] = None
            _save_progress_state(state, PROGRESS_STATE_PATH)

            subset = full_train_data[:data_size]
            print(f"\n[dataset] Writing first {data_size} samples to {effective_train_file}", flush=True)
            _write_json(effective_train_file, subset)

            if FINETUNED_MODEL_PATH.exists():
                print(f"[cleanup] Removing previous finetuned output: {FINETUNED_MODEL_PATH}", flush=True)
                if FINETUNED_MODEL_PATH.is_dir():
                    shutil.rmtree(FINETUNED_MODEL_PATH)
                else:
                    FINETUNED_MODEL_PATH.unlink()

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
                f"output_dir={FINETUNED_MODEL_PATH}",
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

            metrics = _extract_metrics(effective_report_file)
            row = {
                "data_size": data_size,
                "exact_match_rate": metrics["exact_match_rate"],
                "action_match_rate": metrics["action_match_rate"],
            }
            results = _upsert_result(results, row)
            completed_sizes.add(data_size)
            state["results"] = results
            state["completed_sizes"] = sorted(completed_sizes)
            state["last_status"] = "completed"
            state["last_error"] = None
            _save_progress_state(state, PROGRESS_STATE_PATH)
            _save_csv(results, RESULTS_CSV_PATH)
            _save_chart(results, CHART_PATH)
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
        print(f"[done] Chart saved to: {CHART_PATH}", flush=True)
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
        print(f"\n[restore] Restored original train split: {effective_train_file}", flush=True)


if __name__ == "__main__":
    main()
