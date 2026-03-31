#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TEMP_DIR = EXPERIMENT_DIR / ".cache"
DEFAULT_BENCHMARK_PROMPTS = EXPERIMENT_DIR / "prompts" / "default_prompts.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import configure_report_matplotlib, pick_plot_text
from src.utils.run_meta import record_run_meta


BATCH_SIZE = 1
NUM_SAMPLES = 200
MAX_NEW_TOKENS = 128
MAX_MODEL_LEN = 2048
POST_RUN_SLEEP_SECONDS = 15
VRAM_POLL_INTERVAL_SEC = 0.2
DEFAULT_MEMORY_BUDGETS_GB = (8.0, 6.0, 4.0, 2.0)
GPU_MEMORY_UTILIZATION_CAP = 0.99
GPU_MEMORY_UTILIZATION_FLOOR = 0.05

MODEL_ARTIFACT: dict[str, Any] = {
    "model_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
    "tokenizer_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
    "hf_repo_id": "Qwen/Qwen2.5-3B-Instruct",
    "allow_patterns": None,
}

BACKEND_DEPENDENCIES: list[dict[str, str]] = [
    {"import_name": "vllm", "pip_name": "vllm"},
    {"import_name": "torch", "pip_name": "torch"},
    {"import_name": "bitsandbytes", "pip_name": "bitsandbytes"},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验11 Exp7 补充：vLLM 在不同显存预算下的推理性能基准。")
    parser.add_argument("--gpu-id", type=int, default=None, help="nvidia-smi 监控的物理 GPU 编号，默认自动推断。")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_ARTIFACT["model_path"]))
    parser.add_argument("--tokenizer-path", type=Path, default=Path(MODEL_ARTIFACT["tokenizer_path"]))
    parser.add_argument("--hf-repo-id", type=str, default=str(MODEL_ARTIFACT.get("hf_repo_id") or ""))
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--memory-budgets-gb",
        type=str,
        default=",".join(str(int(v)) for v in DEFAULT_MEMORY_BUDGETS_GB),
        help="逗号分隔的显存预算列表，单位 GB，例如 8,6,4,2。",
    )
    parser.add_argument(
        "--total-gpu-memory-mb",
        type=int,
        default=None,
        help="手动指定可见 GPU 总显存（MB）。默认自动探测，用于把显存预算换算成 gpu_memory_utilization。",
    )
    parser.add_argument("--benchmark-prompts-file", type=Path, default=DEFAULT_BENCHMARK_PROMPTS)
    parser.add_argument("--sleep-seconds", type=int, default=POST_RUN_SLEEP_SECONDS)
    parser.add_argument("--vram-poll-interval", type=float, default=VRAM_POLL_INTERVAL_SEC)
    parser.add_argument("--auto-install-deps", action="store_true", help="缺少依赖时自动执行 pip install。")
    parser.add_argument("--auto-download-missing-models", action="store_true", help="缺少模型资产时尝试自动下载。")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "")
    return parser.parse_args(argv)


def print_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def print_warning(message: str) -> None:
    print(f"\033[1;33m[WARN]\033[0m {message}", flush=True)


def print_error(message: str) -> None:
    print(f"\033[1;31m[ERROR]\033[0m {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def install_python_package(package_name: str) -> None:
    print_info(f"检测到缺失依赖，准备安装：{package_name}")
    subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True, text=True)


def ensure_backend_dependencies(*, auto_install: bool) -> tuple[bool, str | None]:
    for dep in BACKEND_DEPENDENCIES:
        if maybe_import(dep["import_name"]):
            continue
        if not auto_install:
            return False, f"缺少依赖 {dep['import_name']}，可使用 --auto-install-deps 自动安装。"
        try:
            install_python_package(dep["pip_name"])
        except subprocess.CalledProcessError as exc:
            return False, f"安装依赖 {dep['pip_name']} 失败：{exc}"
        if not maybe_import(dep["import_name"]):
            return False, f"依赖 {dep['import_name']} 安装后仍不可用。"
    return True, None


def download_model_from_hf(
    *,
    repo_id: str,
    target_dir: Path,
    hf_token: str,
    allow_patterns: list[str] | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    ensure_dir(target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=hf_token or None,
        allow_patterns=allow_patterns,
    )


def ensure_model_artifact(
    artifact: dict[str, Any],
    *,
    auto_download: bool,
    hf_token: str,
) -> tuple[bool, str | None]:
    model_path = Path(artifact["model_path"])
    if model_path.exists():
        return True, None

    repo_id = str(artifact.get("hf_repo_id") or "").strip()
    if not auto_download:
        return False, f"模型缺失：{model_path}。可使用 --auto-download-missing-models 自动下载。"
    if not repo_id:
        return False, f"模型缺失：{model_path}。当前未提供可下载的 Hugging Face Repo ID。"

    try:
        target_dir = model_path if model_path.suffix == "" else model_path.parent
        print_info(f"模型不存在，开始从 Hugging Face 下载：{repo_id} -> {target_dir}")
        download_model_from_hf(
            repo_id=repo_id,
            target_dir=target_dir,
            hf_token=hf_token,
            allow_patterns=artifact.get("allow_patterns"),
        )
    except Exception as exc:
        return False, f"下载模型失败：{repo_id} -> {model_path}，错误：{exc}"

    if not model_path.exists():
        return False, f"模型下载完成后仍未找到目标路径：{model_path}"
    return True, None


def infer_gpu_id(explicit_gpu_id: int | None) -> int:
    if explicit_gpu_id is not None:
        return int(explicit_gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


def query_total_gpu_memory_mb(gpu_id: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if values and values[0].replace(".", "", 1).isdigit():
            return int(float(values[0]))
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            device_index = 0
            props = torch.cuda.get_device_properties(device_index)
            return int(props.total_memory // (1024 * 1024))
    except Exception:
        pass
    return None


def monitor_vram(stop_event: threading.Event) -> None:
    peak_vram_mb = 0.0
    samples: list[float] = []
    gpu_id = int(getattr(stop_event, "gpu_id", 0))
    poll_interval_sec = float(getattr(stop_event, "poll_interval_sec", 0.5))
    query_cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]

    while not stop_event.is_set():
        try:
            result = subprocess.run(query_cmd, check=True, capture_output=True, text=True)
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if values:
                current = max(values)
                peak_vram_mb = max(peak_vram_mb, current)
                samples.append(current)
        except Exception as exc:
            stop_event.monitor_error = str(exc)
            break
        stop_event.wait(poll_interval_sec)

    stop_event.peak_vram_mb = peak_vram_mb
    stop_event.samples = samples


def parse_memory_budgets(raw_text: str) -> list[float]:
    budgets: list[float] = []
    seen: set[float] = set()
    for chunk in str(raw_text).split(","):
        text = chunk.strip()
        if not text:
            continue
        value = float(text)
        if value <= 0:
            raise ValueError("显存预算必须为正数。")
        if value in seen:
            continue
        seen.add(value)
        budgets.append(value)
    if not budgets:
        raise ValueError("至少需要提供一个显存预算。")
    return budgets


def budget_gb_to_mb(budget_gb: float) -> int:
    return int(round(float(budget_gb) * 1024.0))


def compute_gpu_memory_utilization(
    requested_budget_mb: int,
    total_gpu_memory_mb: int,
    *,
    floor: float = GPU_MEMORY_UTILIZATION_FLOOR,
    cap: float = GPU_MEMORY_UTILIZATION_CAP,
) -> float:
    if total_gpu_memory_mb <= 0:
        raise ValueError("total_gpu_memory_mb 必须大于 0。")
    raw_ratio = float(requested_budget_mb) / float(total_gpu_memory_mb)
    return round(min(cap, max(floor, raw_ratio)), 6)


def _format_budget_label(budget_gb: float) -> str:
    if float(budget_gb).is_integer():
        return f"{int(budget_gb)}GB"
    return f"{budget_gb:g}GB"


def build_budget_cases(memory_budgets_gb: list[float], *, total_gpu_memory_mb: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for budget_gb in memory_budgets_gb:
        requested_budget_mb = budget_gb_to_mb(budget_gb)
        utilization = compute_gpu_memory_utilization(requested_budget_mb, total_gpu_memory_mb)
        effective_budget_mb = round(float(total_gpu_memory_mb) * utilization, 2)
        clamped = effective_budget_mb + 1e-9 < float(requested_budget_mb)
        budget_label = _format_budget_label(budget_gb)
        note = (
            f"目标预算 {requested_budget_mb} MB -> gpu_memory_utilization={utilization:.4f}"
            f"（按可见 GPU 总显存 {total_gpu_memory_mb} MB 换算）"
        )
        if clamped:
            note += f"，由于 vLLM 预算比例上限为 {GPU_MEMORY_UTILIZATION_CAP:.2f}，实际近似预算约 {effective_budget_mb:.2f} MB。"

        cases.append(
            {
                "name": f"vLLM_BNB_4bit_{budget_label}",
                "budget_label": budget_label,
                "backend": "vllm",
                "quant": "4bit",
                "stack_label": f"vLLM + bitsandbytes 4bit @ {budget_label}",
                "requested_budget_gb": float(budget_gb),
                "requested_budget_mb": requested_budget_mb,
                "effective_budget_mb": effective_budget_mb,
                "total_gpu_memory_mb": int(total_gpu_memory_mb),
                "gpu_memory_utilization": utilization,
                "budget_mapping_note": note,
            }
        )
    return cases


def build_artifact_from_args(args: argparse.Namespace) -> dict[str, Any]:
    default_model_path = Path(MODEL_ARTIFACT["model_path"]).resolve()
    current_model_path = Path(args.model_path).resolve()
    hf_repo_id = str(args.hf_repo_id or "").strip()
    if not hf_repo_id and current_model_path == default_model_path:
        hf_repo_id = str(MODEL_ARTIFACT.get("hf_repo_id") or "")

    return {
        "model_path": current_model_path,
        "tokenizer_path": Path(args.tokenizer_path).resolve(),
        "hf_repo_id": hf_repo_id or None,
        "allow_patterns": MODEL_ARTIFACT.get("allow_patterns"),
    }


def build_benchmark_command(
    cfg: dict[str, Any],
    artifact: dict[str, Any],
    *,
    args: argparse.Namespace,
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "cli.py"),
        "eval",
        "benchmark",
        "--backend",
        "vllm",
        "--model-path",
        str(Path(artifact["model_path"]).resolve()),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(cfg["gpu_memory_utilization"]),
        "--output-json",
        str(output_json),
        "--require-gpu",
        "--use-chat",
        "--quantization",
        str(cfg["quant"]),
    ]
    tokenizer_path = artifact.get("tokenizer_path")
    if tokenizer_path:
        command.extend(["--tokenizer-path", str(Path(tokenizer_path).resolve())])

    prompts_file = Path(args.benchmark_prompts_file)
    if prompts_file.exists():
        command.extend(["--prompts-file", str(prompts_file.resolve())])
    return command


def build_runtime_env(*, gpu_id: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def run_command(command: list[str], *, log_path: Path, gpu_id: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=build_runtime_env(gpu_id=gpu_id),
    )
    ensure_dir(log_path.parent)
    log_path.write_text(
        f"$ {' '.join(command)}\n\n[stdout]\n{result.stdout}\n\n[stderr]\n{result.stderr}\n",
        encoding="utf-8",
    )
    return result


def persist_failure_log(command: list[str], exc: subprocess.CalledProcessError, *, log_path: Path) -> None:
    ensure_dir(log_path.parent)
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    log_path.write_text(
        f"$ {' '.join(command)}\n\n[returncode]\n{exc.returncode}\n\n[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n",
        encoding="utf-8",
    )


def cleanup_runtime_state() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def looks_like_oom(exc: subprocess.CalledProcessError | None) -> bool:
    if exc is None:
        return False
    text = "\n".join(
        [
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
        ]
    ).lower()
    return any(
        token in text
        for token in [
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "less than desired gpu memory utilization",
        ]
    )


def classify_failure(exc: subprocess.CalledProcessError | None) -> str:
    if exc is None:
        return "failed"
    text = "\n".join(
        [
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
        ]
    ).lower()
    if looks_like_oom(exc):
        return "oom"
    if "no module named" in text or "缺少" in text:
        return "dependency_missing"
    return "failed"


def to_peak_vram_display(value: float | str | None, *, oom: bool) -> float | str:
    if oom:
        return "OOM"
    if isinstance(value, (int, float)) and float(value) > 0:
        return round(float(value), 2)
    return math.nan


def peak_vram_for_plot(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def _get_float(report: dict[str, Any], key: str) -> float:
    value = report.get(key, math.nan)
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def _fmt_metric(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    text = str(value).strip()
    return text or "-"


def summarize_case_result(
    cfg: dict[str, Any],
    benchmark_report: dict[str, Any],
    *,
    batch_size: int,
    num_samples: int,
    peak_vram_mb: float | str | None,
    status: str,
    benchmark_ok: bool,
    benchmark_status: str,
    benchmark_path: Path,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "Name": cfg["name"],
        "Budget Label": cfg["budget_label"],
        "Requested Budget (GB)": cfg["requested_budget_gb"],
        "Requested Budget (MB)": cfg["requested_budget_mb"],
        "Effective Budget (MB)": cfg["effective_budget_mb"],
        "Total GPU Memory (MB)": cfg["total_gpu_memory_mb"],
        "GPU Memory Utilization": cfg["gpu_memory_utilization"],
        "Budget Mapping Note": cfg["budget_mapping_note"],
        "Stack Label": cfg["stack_label"],
        "Backend": cfg["backend"],
        "Quantization": cfg["quant"],
        "Batch Size": batch_size,
        "Num Samples": num_samples,
        "Avg Latency (s)": _get_float(benchmark_report, "avg_latency") if benchmark_ok else math.nan,
        "P50 Latency (s)": _get_float(benchmark_report, "p50_latency") if benchmark_ok else math.nan,
        "P95 Latency (s)": _get_float(benchmark_report, "p95_latency") if benchmark_ok else math.nan,
        "Sample Throughput (samples/s)": _get_float(benchmark_report, "sample_throughput_sps") if benchmark_ok else math.nan,
        "Token Throughput (tokens/s)": _get_float(benchmark_report, "token_throughput_tps") if benchmark_ok else math.nan,
        "Avg TTFT (s)": _get_float(benchmark_report, "avg_ttft_sec") if benchmark_ok else math.nan,
        "Avg Decode TPS": _get_float(benchmark_report, "avg_decode_tps") if benchmark_ok else math.nan,
        "Peak VRAM (MB)": peak_vram_mb,
        "Avg Process RSS (MB)": _get_float(benchmark_report, "avg_process_rss_mb") if benchmark_ok else math.nan,
        "Max Process RSS (MB)": _get_float(benchmark_report, "max_process_rss_mb") if benchmark_ok else math.nan,
        "Status": status,
        "Benchmark Status": benchmark_status,
        "Benchmark Report": str(benchmark_path.resolve()),
        "Model Path": str(Path(artifact["model_path"]).resolve()),
        "Tokenizer Path": str(Path(artifact["tokenizer_path"]).resolve()) if artifact.get("tokenizer_path") else "",
    }


def draw_figures(df: pd.DataFrame, *, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((TEMP_DIR / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_report_matplotlib(matplotlib)

    try:
        import seaborn as sns
    except Exception:
        sns = None

    ensure_dir(output_dir)
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    plot_df = df.copy()
    plot_df["Peak_VRAM_Plot_MB"] = plot_df["Peak VRAM (MB)"].apply(peak_vram_for_plot)
    labels = plot_df["Budget Label"].tolist()
    x = list(range(len(plot_df)))

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    width = 0.25
    avg_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in plot_df["Avg Latency (s)"].tolist()]
    p50_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in plot_df["P50 Latency (s)"].tolist()]
    p95_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in plot_df["P95 Latency (s)"].tolist()]
    ax1.bar([idx - width for idx in x], avg_values, width=width, color="#4c78a8", label="Avg Latency")
    ax1.bar(x, p50_values, width=width, color="#72b7b2", label="P50 Latency")
    ax1.bar([idx + width for idx in x], p95_values, width=width, color="#f58518", label="P95 Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(pick_plot_text("图1：vLLM 不同显存预算下的延迟对比", "Figure 1: vLLM Latency under Different Memory Budgets"))
    ax1.set_xlabel(pick_plot_text("显存预算", "Memory Budget"))
    ax1.set_ylabel("Seconds")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(output_dir / "exp7_vllm_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    throughput_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0
        for v in plot_df["Sample Throughput (samples/s)"].tolist()
    ]
    token_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0
        for v in plot_df["Token Throughput (tokens/s)"].tolist()
    ]
    ax2.bar(labels, throughput_values, color="#54a24b", label="Samples/s")
    ax2.set_xlabel(pick_plot_text("显存预算", "Memory Budget"))
    ax2.set_ylabel("Samples / Second", color="#54a24b")
    ax2.tick_params(axis="y", labelcolor="#54a24b")
    ax2.tick_params(axis="x", rotation=0)
    ax2.set_title(pick_plot_text("图2：vLLM 不同显存预算下的吞吐对比", "Figure 2: vLLM Throughput under Different Memory Budgets"))
    ax2_twin = ax2.twinx()
    ax2_twin.plot(labels, token_values, color="#e45756", marker="o", linewidth=2.0, label="Tokens/s")
    ax2_twin.set_ylabel("Tokens / Second", color="#e45756")
    ax2_twin.tick_params(axis="y", labelcolor="#e45756")
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp7_vllm_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(14, 8))
    requested_values = [float(v) for v in plot_df["Requested Budget (MB)"].tolist()]
    peak_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0
        for v in plot_df["Peak_VRAM_Plot_MB"].tolist()
    ]
    rss_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0
        for v in plot_df["Avg Process RSS (MB)"].tolist()
    ]
    ax3.bar([idx - 0.25 for idx in x], requested_values, width=0.25, color="#bab0ab", label="Requested Budget")
    ax3.bar(x, peak_values, width=0.25, color="#e45756", label="Observed Peak VRAM")
    ax3.bar([idx + 0.25 for idx in x], rss_values, width=0.25, color="#b279a2", label="Avg RSS")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title(pick_plot_text("图3：显存预算与实际占用对比", "Figure 3: Budget vs Observed Memory Usage"))
    ax3.set_xlabel(pick_plot_text("显存预算", "Memory Budget"))
    ax3.set_ylabel("MB")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp7_vllm_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)


def write_markdown_report(df: pd.DataFrame, *, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    lines: list[str] = [
        "# Exp7 vLLM 显存预算补充实验报告",
        "",
        "## 实验目标",
        "",
        "- 固定 `vLLM + bitsandbytes 4bit` 这一路部署栈，只改变显存预算，观察推理延迟、吞吐与资源占用如何变化。",
        "- 当前默认对比 `8GB / 6GB / 4GB / 2GB` 四档预算。",
        "- 结果用于回答“同一套 vLLM 部署在不同显存预算下还能跑多快、何时开始明显退化或直接 OOM”。",
        "",
        "## 预算换算口径",
        "",
        "- vLLM 本身接收的是 `gpu_memory_utilization`，不是直接的“显存上限 MB”。",
        "- 本实验按 `requested_budget_mb / total_gpu_memory_mb` 把目标预算换算成 `gpu_memory_utilization`，并限制在 `0.05 ~ 0.99` 区间。",
        "- 因此这里的 `8GB / 6GB / 4GB / 2GB` 是“近似预算档位”，不是 CUDA 层面的绝对硬上限。",
        "- 若预算过低，常见现象包括初始化阶段 OOM、KV Cache 过小导致吞吐下降，或尾延迟明显变差。",
        "",
        "## 当前结果",
        "",
        "| 预算 | Utilization | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {budget} | {util} | {avg} | {p50} | {p95} | {throughput} | {token_tps} | {peak} | {status} |".format(
                budget=row["Budget Label"],
                util=_fmt_metric(row["GPU Memory Utilization"], digits=4),
                avg=_fmt_metric(row["Avg Latency (s)"]),
                p50=_fmt_metric(row["P50 Latency (s)"]),
                p95=_fmt_metric(row["P95 Latency (s)"]),
                throughput=_fmt_metric(row["Sample Throughput (samples/s)"]),
                token_tps=_fmt_metric(row["Token Throughput (tokens/s)"]),
                peak=_fmt_metric(row["Peak VRAM (MB)"], digits=2),
                status=row["Status"],
            )
        )

    success_df = df[df["Status"] == "success"].copy()
    if not success_df.empty:
        success_df = success_df.sort_values("Avg Latency (s)", ascending=True)
        best_row = success_df.iloc[0]
        lines.extend(
            [
                "",
                "## 结果分析",
                "",
                f"- 当前成功运行的预算档位共有 `{len(success_df)}` 个，其中表现最优的是 `{best_row['Budget Label']}`。",
            ]
        )

        if len(success_df) >= 2:
            worst_row = success_df.iloc[-1]
            best_latency = float(best_row["Avg Latency (s)"])
            worst_latency = float(worst_row["Avg Latency (s)"])
            best_throughput = float(best_row["Sample Throughput (samples/s)"])
            worst_throughput = float(worst_row["Sample Throughput (samples/s)"])
            latency_gain_pct = ((worst_latency - best_latency) / worst_latency * 100.0) if worst_latency > 0 else 0.0
            throughput_gain_pct = ((best_throughput - worst_throughput) / worst_throughput * 100.0) if worst_throughput > 0 else 0.0
            lines.extend(
                [
                    (
                        f"- 最优档 `{best_row['Budget Label']}` 的平均延迟为 `{best_latency:.4f}s`，"
                        f"相对当前最慢的成功档 `{worst_row['Budget Label']}` 下降约 `{latency_gain_pct:.2f}%`。"
                    ),
                    (
                        f"- 在吞吐上，`{best_row['Budget Label']}` 达到 `{best_throughput:.4f} samples/s`，"
                        f"相对 `{worst_row['Budget Label']}` 提升约 `{throughput_gain_pct:.2f}%`。"
                    ),
                ]
            )

    failed_rows = df[df["Status"] != "success"]
    if not failed_rows.empty:
        if "## 结果分析" not in lines:
            lines.extend(["", "## 结果分析", ""])
        for _, row in failed_rows.iterrows():
            if row["Status"] == "oom":
                lines.append(
                    f"- `{row['Budget Label']}` 档在初始化阶段触发 OOM，说明当前可用空闲显存不足以满足该预算对应的 vLLM 启动需求。"
                )
            else:
                lines.append(
                    f"- `{row['Budget Label']}` 档未成功完成 benchmark，当前结果表明该预算已逼近或低于可运行边界。"
                )

    lines.extend(
        [
            "",
            "## 解读建议",
            "",
            "- 如果关注交互体验，优先比较 `Avg Latency` 和 `P95 Latency`，因为显存预算收紧时，尾延迟通常先恶化。",
            "- 如果关注部署下限，优先看 `Status` 是否出现 `oom`，以及 `Peak VRAM` 是否已经逼近目标预算。",
            "- 如果 `2GB` 或 `4GB` 档直接失败，这并不等价于“vLLM 不适合该模型”，更常见的解释是模型权重、运行时编译和 KV Cache 预算已无法同时容纳。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_single_case(
    cfg: dict[str, Any],
    artifact: dict[str, Any],
    *,
    args: argparse.Namespace,
    results_dir: Path,
    gpu_id: int,
) -> dict[str, Any]:
    name = str(cfg["name"])
    benchmark_json = results_dir / f"{name}_benchmark.json"
    benchmark_log = LOGS_DIR / f"{name}_benchmark.log"

    benchmark_ok = False
    benchmark_status = "pending"

    stop_event = threading.Event()
    stop_event.gpu_id = gpu_id
    stop_event.poll_interval_sec = max(0.05, float(args.vram_poll_interval))
    monitor_thread = threading.Thread(target=monitor_vram, args=(stop_event,), daemon=True)
    monitor_thread.start()

    try:
        benchmark_command = build_benchmark_command(cfg, artifact, args=args, output_json=benchmark_json)
        try:
            print_info(
                f"[{name}] 开始执行 benchmark：预算 {cfg['requested_budget_mb']} MB，"
                f"gpu_memory_utilization={cfg['gpu_memory_utilization']:.4f}"
            )
            run_command(benchmark_command, log_path=benchmark_log, gpu_id=gpu_id)
            benchmark_ok = True
            benchmark_status = "success"
        except subprocess.CalledProcessError as exc:
            persist_failure_log(benchmark_command, exc, log_path=benchmark_log)
            benchmark_status = classify_failure(exc)
            print_error(f"[{name}] benchmark 执行失败，已记录日志：{benchmark_log}")
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)
        peak_vram_mb = float(getattr(stop_event, "peak_vram_mb", 0.0) or 0.0)
        monitor_error = getattr(stop_event, "monitor_error", None)
        if monitor_error:
            print_warning(f"[{name}] nvidia-smi 监控异常：{monitor_error}")

        cleanup_runtime_state()
        print_info(f"[{name}] 冷却 {args.sleep_seconds} 秒，等待显存彻底释放")
        time.sleep(args.sleep_seconds)

    oom_flag = benchmark_status == "oom"
    benchmark_report = load_json_if_exists(benchmark_json) if benchmark_ok else {}
    fallback_peak = float(benchmark_report.get("peak_memory", 0.0) or 0.0)
    peak_display = to_peak_vram_display(max(peak_vram_mb, fallback_peak), oom=oom_flag)

    return summarize_case_result(
        cfg,
        benchmark_report,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        peak_vram_mb=peak_display,
        status=benchmark_status,
        benchmark_ok=benchmark_ok,
        benchmark_status=benchmark_status,
        benchmark_path=benchmark_json,
        artifact=artifact,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.results_dir)
    ensure_dir(LOGS_DIR)
    ensure_dir(TEMP_DIR)

    memory_budgets_gb = parse_memory_budgets(args.memory_budgets_gb)
    gpu_id = infer_gpu_id(args.gpu_id)
    total_gpu_memory_mb = int(args.total_gpu_memory_mb) if args.total_gpu_memory_mb else query_total_gpu_memory_mb(gpu_id)
    if not total_gpu_memory_mb or total_gpu_memory_mb <= 0:
        print_error("无法探测 GPU 总显存，请通过 --total-gpu-memory-mb 手动指定。")
        return 1

    print_info(
        f"实验11补充开始，目标 GPU = {gpu_id}，总显存约 {total_gpu_memory_mb} MB，"
        f"预算档位 = {', '.join(_format_budget_label(v) for v in memory_budgets_gb)}"
    )

    ok, reason = ensure_backend_dependencies(auto_install=args.auto_install_deps)
    if not ok:
        print_error(reason or "依赖检查失败。")
        return 1

    artifact = build_artifact_from_args(args)
    ok, reason = ensure_model_artifact(
        artifact,
        auto_download=args.auto_download_missing_models,
        hf_token=args.hf_token,
    )
    if not ok:
        print_error(reason or "模型检查失败。")
        return 1

    cases = build_budget_cases(memory_budgets_gb, total_gpu_memory_mb=total_gpu_memory_mb)
    rows = [run_single_case(cfg, artifact, args=args, results_dir=args.results_dir, gpu_id=gpu_id) for cfg in cases]

    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "exp7_vllm_memory_budget_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    draw_figures(df, output_dir=args.results_dir)

    markdown_path = args.results_dir / "exp7_vllm_budget_report.md"
    write_markdown_report(df, output_path=markdown_path)

    summary_path = args.results_dir / "exp7_vllm_memory_budget_summary.json"
    summary_payload = {
        "results_csv": str(csv_path.resolve()),
        "results_markdown": str(markdown_path.resolve()),
        "gpu_id": gpu_id,
        "total_gpu_memory_mb": total_gpu_memory_mb,
        "num_cases": len(rows),
        "comparison_scope": "同一 vLLM 部署栈在不同显存预算下的端到端速度与资源对比",
        "budget_mapping_policy": {
            "formula": "gpu_memory_utilization = clamp(requested_budget_mb / total_gpu_memory_mb, 0.05, 0.99)",
            "note": "显存预算是近似档位，不是 CUDA 层面的严格硬上限。",
        },
        "fairness_notes": [
            "所有 case 统一使用同一基座模型、同一 tokenizer、同一 prompts、同一 batch size、同一 max_new_tokens。",
            "唯一核心变量是目标显存预算，以及由此换算得到的 gpu_memory_utilization。",
            "低预算下若出现 OOM，更多反映的是模型权重、运行时编译和 KV Cache 无法同时容纳。",
        ],
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = record_run_meta(
        args.results_dir,
        cli_args=vars(args),
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        data_paths=[args.benchmark_prompts_file] if Path(args.benchmark_prompts_file).exists() else None,
        extra_meta={
            "entry": "experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py",
            "stage": "vllm_memory_budget_benchmark",
            "summary_path": str(summary_path.resolve()),
            "gpu_id": gpu_id,
            "total_gpu_memory_mb": total_gpu_memory_mb,
            "memory_budgets_gb": memory_budgets_gb,
        },
    )

    print_info(f"CSV 汇总已导出：{csv_path}")
    print_info(f"Markdown 报告已导出：{markdown_path}")
    print_info(f"JSON 摘要已导出：{summary_path}")
    print_info(f"运行元数据已导出：{meta_path}")
    print_info(f"图表已导出到目录：{args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
