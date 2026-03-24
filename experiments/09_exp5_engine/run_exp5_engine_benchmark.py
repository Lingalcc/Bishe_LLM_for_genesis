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

BATCH_SIZE = 1
NUM_SAMPLES = 200
MAX_NEW_TOKENS = 128
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.9
POST_RUN_SLEEP_SECONDS = 15
VRAM_LIMIT_MB = 8192


# Exp5 修订后的实验口径：
# 1. 只评测推理速度与资源占用，不再统计准确率；
# 2. 结论解释层级限定为“本地部署栈”端到端表现，不将结果表述为纯引擎优劣；
# 3. 默认矩阵纳入当前仓库可复现的 GPU 本地部署方案，并强制 GPU-only 运行。
test_configs: list[dict[str, Any]] = [
    {
        "name": "Transformers_BNB_4bit",
        "backend": "transformers",
        "quant": "4bit",
        "artifact_key": "base_fp16",
        "stack_label": "Transformers + bitsandbytes 4bit",
        "quant_note": "运行时 4bit（bitsandbytes NF4）",
    },
    {
        "name": "LlamaCPP_GGUF_Q4_K_M",
        "backend": "llama.cpp",
        "quant": "gguf_q4_k_m",
        "artifact_key": "llamacpp_gguf_q4",
        "stack_label": "llama.cpp + GGUF Q4_K_M",
        "quant_note": "离线量化 GGUF Q4_K_M",
    },
    {
        "name": "ExLlamaV2_EXL2_LocalAsset",
        "backend": "exllamav2",
        "quant": "exl2_local_asset",
        "artifact_key": "exllamav2_local_asset",
        "stack_label": "ExLlamaV2 + EXL2 local asset",
        "quant_note": "本地 EXL2 资产 README 标注 Bits 8.0",
    },
]


MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "base_fp16": {
        "model_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
        "tokenizer_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
        "hf_repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "allow_patterns": None,
    },
    "llamacpp_gguf_q4": {
        "model_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct-GGUF" / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "tokenizer_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
        "hf_repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "allow_patterns": ["*Q4_K_M*.gguf", "*.gguf"],
        "download_hint": "若官方 GGUF 仓库失效，请在 MODEL_ARTIFACTS['llamacpp_gguf_q4'] 中替换为新的可用仓库 ID。",
    },
    "exllamav2_local_asset": {
        "model_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct-EXL2-4bpw",
        "tokenizer_path": REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
        "hf_repo_id": None,
        "allow_patterns": None,
        "download_hint": "当前仓库仅发现本地 EXL2 资产；其 README 标注 Bits 8.0，如需严格同构 4bit 对比，请替换为真实 4bpw EXL2 模型。",
    },
}


BACKEND_DEPENDENCIES: dict[str, list[dict[str, str]]] = {
    "transformers": [
        {"import_name": "transformers", "pip_name": "transformers"},
        {"import_name": "torch", "pip_name": "torch"},
    ],
    "llama.cpp": [
        {"import_name": "llama_cpp", "pip_name": "llama-cpp-python"},
    ],
    "exllamav2": [
        {"import_name": "ninja", "pip_name": "ninja"},
        {"import_name": "exllamav2", "pip_name": "exllamav2"},
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验09 Exp5：本地部署栈速度与资源基准。")
    parser.add_argument("--gpu-id", type=int, default=None, help="nvidia-smi 监控的物理 GPU 编号，默认自动推断。")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--benchmark-prompts-file", type=Path, default=DEFAULT_BENCHMARK_PROMPTS)
    parser.add_argument("--sleep-seconds", type=int, default=POST_RUN_SLEEP_SECONDS)
    parser.add_argument("--auto-install-deps", action="store_true", help="缺少后端依赖时自动执行 pip install。")
    parser.add_argument("--auto-download-missing-models", action="store_true", help="缺少模型资产时尝试从 Hugging Face 下载。")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "")
    return parser.parse_args(argv)


def print_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def print_warning(message: str) -> None:
    print(f"\033[1;33m[WARN]\033[0m {message}", flush=True)


def print_error(message: str) -> None:
    print(f"\033[1;31m[ERROR]\033[0m {message}", flush=True)


def infer_gpu_id(explicit_gpu_id: int | None) -> int:
    if explicit_gpu_id is not None:
        return int(explicit_gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


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
            result = subprocess.run(
                query_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
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
    subprocess.run(
        [sys.executable, "-m", "pip", "install", package_name],
        check=True,
        text=True,
    )


def ensure_backend_dependencies(
    backend: str,
    *,
    quantization: str,
    auto_install: bool,
) -> tuple[bool, str | None]:
    deps = list(BACKEND_DEPENDENCIES.get(backend, []))
    quant_text = quantization.strip().lower()
    if backend == "transformers" and "4bit" in quant_text:
        deps.append({"import_name": "bitsandbytes", "pip_name": "bitsandbytes"})

    for dep in deps:
        if maybe_import(dep["import_name"]):
            continue
        if not auto_install:
            return False, f"缺少依赖 {dep['import_name']}，可使用 --auto-install-deps 自动安装。"
        try:
            install_python_package(dep["pip_name"])
        except subprocess.CalledProcessError as exc:
            return False, f"安装依赖 {dep['pip_name']} 失败: {exc}"
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

    repo_id = artifact.get("hf_repo_id")
    if not auto_download:
        hint = artifact.get("download_hint")
        if hint:
            return False, f"模型缺失：{model_path}。{hint}"
        return False, f"模型缺失：{model_path}。可使用 --auto-download-missing-models 自动下载。"
    if not repo_id:
        hint = artifact.get("download_hint") or "请在配置区补充 hf_repo_id。"
        return False, f"模型缺失：{model_path}。{hint}"

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
        if model_path.suffix and model_path.parent.exists():
            exact_matches = list(model_path.parent.rglob(model_path.name))
            if exact_matches:
                artifact["model_path"] = exact_matches[0]
                return True, None
            if model_path.suffix.lower() == ".gguf":
                gguf_matches = list(model_path.parent.rglob("*.gguf"))
                if gguf_matches:
                    artifact["model_path"] = gguf_matches[0]
                    return True, None
        return False, f"模型下载完成后仍未找到目标路径：{model_path}"
    return True, None


def map_quantization_for_cli(raw_quant: str) -> str | None:
    text = raw_quant.strip().lower()
    if text in {"", "none", "null"}:
        return None
    return raw_quant


def clone_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return dict(artifact)


def make_runtime_plan(
    *,
    backend: str,
    quantization: str | None,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": backend,
        "quantization": quantization,
        "artifact": artifact,
    }


def prepare_runtime_plan(
    plan: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> tuple[bool, str, str | None]:
    quant_text = str(plan.get("quantization") or "none")
    ok, reason = ensure_backend_dependencies(
        str(plan["backend"]),
        quantization=quant_text,
        auto_install=args.auto_install_deps,
    )
    if not ok:
        return False, "dependency_missing", reason

    ok, reason = ensure_model_artifact(
        plan["artifact"],
        auto_download=args.auto_download_missing_models,
        hf_token=args.hf_token,
    )
    if not ok:
        return False, "model_missing", reason

    return True, "ready", None


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
        str(cfg["backend"]),
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
        str(args.gpu_memory_utilization),
        "--output-json",
        str(output_json),
        "--require-gpu",
        "--use-chat",
    ]
    tokenizer_path = artifact.get("tokenizer_path")
    if tokenizer_path:
        command.extend(["--tokenizer-path", str(Path(tokenizer_path).resolve())])

    prompts_file = Path(args.benchmark_prompts_file)
    if prompts_file.exists():
        command.extend(["--prompts-file", str(prompts_file.resolve())])

    quant = map_quantization_for_cli(str(cfg["quant"]))
    if quant is not None:
        command.extend(["--quantization", quant])
    return command


def build_runtime_env(*, gpu_id: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def run_command(
    command: list[str],
    *,
    log_path: Path,
    gpu_id: int,
) -> subprocess.CompletedProcess[str]:
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


def persist_failure_log(
    command: list[str],
    exc: subprocess.CalledProcessError,
    *,
    log_path: Path,
) -> None:
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


def looks_like_oom(exc: subprocess.CalledProcessError | None, *, quantization: str) -> bool:
    if exc is None:
        return False
    text = "\n".join(
        [
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
        ]
    ).lower()
    if any(token in text for token in ["out of memory", "cuda oom", "cuda error: out of memory", "cublas_status_alloc_failed"]):
        return True
    if "free memory on device" in text and "less than desired gpu memory utilization" in text:
        return True
    return False


def classify_failure(exc: subprocess.CalledProcessError | None) -> str:
    if exc is None:
        return "failed"
    text = "\n".join(
        [
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else "",
        ]
    ).lower()
    if looks_like_oom(exc, quantization="none"):
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
    text = str(value).strip().lower()
    if text == "oom":
        return float(VRAM_LIMIT_MB)
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
    benchmark_plan: dict[str, Any],
) -> dict[str, Any]:
    benchmark_artifact = benchmark_plan["artifact"]
    return {
        "Name": cfg["name"],
        "Stack Label": cfg.get("stack_label", cfg["name"]),
        "Backend": cfg["backend"],
        "Quantization": cfg["quant"],
        "Quantization Note": cfg.get("quant_note", ""),
        "Comparison Scope": "deployment_stack_only",
        "Execution Policy": "gpu_only",
        "Batch Size": batch_size,
        "Num Samples": num_samples,
        "Avg Latency (s)": _get_float(benchmark_report, "avg_latency") if benchmark_ok else math.nan,
        "P50 Latency (s)": _get_float(benchmark_report, "p50_latency") if benchmark_ok else math.nan,
        "P95 Latency (s)": _get_float(benchmark_report, "p95_latency") if benchmark_ok else math.nan,
        "Sample Throughput (samples/s)": _get_float(benchmark_report, "sample_throughput_sps") if benchmark_ok else math.nan,
        "Peak VRAM (MB)": peak_vram_mb,
        "Avg Process RSS (MB)": _get_float(benchmark_report, "avg_process_rss_mb") if benchmark_ok else math.nan,
        "Max Process RSS (MB)": _get_float(benchmark_report, "max_process_rss_mb") if benchmark_ok else math.nan,
        "Status": status,
        "Benchmark Status": benchmark_status,
        "Benchmark Report": str(benchmark_path.resolve()),
        "Model Path": str(Path(benchmark_artifact["model_path"]).resolve()),
        "Tokenizer Path": str(Path(benchmark_artifact["tokenizer_path"]).resolve()) if benchmark_artifact.get("tokenizer_path") else "",
    }


def draw_figures(df: pd.DataFrame, *, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((TEMP_DIR / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig1, ax1 = plt.subplots(figsize=(16, 8))
    x = list(range(len(plot_df)))
    width = 0.25
    avg_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in plot_df["Avg Latency (s)"].tolist()]
    p50_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in plot_df["P50 Latency (s)"].tolist()]
    p95_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in plot_df["P95 Latency (s)"].tolist()]
    ax1.bar([idx - width for idx in x], avg_values, width=width, color="#4c78a8", label="Avg Latency")
    ax1.bar(x, p50_values, width=width, color="#72b7b2", label="P50 Latency")
    ax1.bar([idx + width for idx in x], p95_values, width=width, color="#f58518", label="P95 Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["Name"].tolist(), rotation=18)
    ax1.set_title("图1：端到端延迟对比")
    ax1.set_xlabel("部署方案")
    ax1.set_ylabel("Seconds")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(output_dir / "exp5_engine_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 8))
    throughput_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0
        for v in plot_df["Sample Throughput (samples/s)"].tolist()
    ]
    ax2.bar(plot_df["Name"], throughput_values, color="#54a24b", edgecolor="#222222", linewidth=0.8)
    ax2.set_title("图2：样本吞吐对比")
    ax2.set_xlabel("部署方案")
    ax2.set_ylabel("Samples / Second")
    ax2.tick_params(axis="x", rotation=18)
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp5_engine_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(16, 8))
    rss_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0
        for v in plot_df["Avg Process RSS (MB)"].tolist()
    ]
    peak_values = [
        float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else float(VRAM_LIMIT_MB)
        for v in plot_df["Peak_VRAM_Plot_MB"].tolist()
    ]
    ax3.bar([idx - 0.2 for idx in x], peak_values, width=0.4, color="#e45756", label="Peak VRAM")
    ax3.bar([idx + 0.2 for idx in x], rss_values, width=0.4, color="#b279a2", label="Avg RSS")
    ax3.axhline(VRAM_LIMIT_MB, color="red", linestyle="--", linewidth=1.8, label="8GB VRAM Limit")
    ax3.set_xticks(x)
    ax3.set_xticklabels(plot_df["Name"].tolist(), rotation=18)
    ax3.set_title("图3：显存与进程内存占用对比")
    ax3.set_xlabel("部署方案")
    ax3.set_ylabel("MB")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp5_engine_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)


def write_markdown_report(df: pd.DataFrame, *, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    lines: list[str] = [
        "# Exp5 速度基准报告",
        "",
        "## 口径修正",
        "",
        "- 本报告只统计推理速度与资源占用，不再统计准确率。",
        "- 参与方案均为未针对当前任务微调的基座模型，因此不使用 Action Match Rate、Exact Match 等任务指标。",
        "- 当前结果仅解释为“本地部署栈”的端到端表现，不将其写成同构量化下的纯推理引擎优劣结论。",
        "- Exp5 在执行层面强制 GPU-only：子进程会绑定 `CUDA_VISIBLE_DEVICES`，并向 benchmark CLI 显式传入 `--require-gpu`。",
        "- 三组方案统一使用相同的 prompts、batch size、num samples、max_new_tokens 和 max_model_len。",
        "- `Transformers_BNB_4bit` 与 `LlamaCPP_GGUF_Q4_K_M` 同属 4bit 部署方案，但底层量化格式分别为 `bitsandbytes 4bit` 与 `GGUF Q4_K_M`，属于不同实现路径。",
        "- `ExLlamaV2_EXL2_LocalAsset` 已纳入同一套 GPU-only 基准，但当前本地 EXL2 资产 README 标注 `Bits 8.0`，因此不应把它写成严格同构 4bit 主结论。",
        "",
        "## 统计指标",
        "",
        "- `Avg Latency (s)`：平均端到端时延",
        "- `P50 Latency (s)`：中位数时延",
        "- `P95 Latency (s)`：尾部时延",
        "- `Sample Throughput (samples/s)`：样本吞吐",
        "- `Peak VRAM (MB)`：通过 `nvidia-smi` 采样得到的峰值显存",
        "- `Avg Process RSS (MB)`：进程常驻内存均值",
        "",
        "## 当前结果",
        "",
        "| 方案 | Backend | Quantization | 量化备注 | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Peak VRAM (MB) | Avg RSS (MB) | 状态 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {backend} | {quant} | {quant_note} | {avg} | {p50} | {p95} | {throughput} | {peak} | {rss} | {status} |".format(
                name=row["Name"],
                backend=row["Backend"],
                quant=row["Quantization"],
                quant_note=row["Quantization Note"],
                avg=_fmt_metric(row["Avg Latency (s)"]),
                p50=_fmt_metric(row["P50 Latency (s)"]),
                p95=_fmt_metric(row["P95 Latency (s)"]),
                throughput=_fmt_metric(row["Sample Throughput (samples/s)"]),
                peak=_fmt_metric(row["Peak VRAM (MB)"], digits=2),
                rss=_fmt_metric(row["Avg Process RSS (MB)"], digits=2),
                status=row["Status"],
            )
        )

    lines.extend(
        [
            "",
            "## 解读建议",
            "",
            "- 如果关注交互响应，优先看 `Avg Latency` 与 `P95 Latency`。",
            "- 如果关注端侧落地约束，优先看 `Peak VRAM` 是否接近 `8GB` 上限。",
            "- 如果需要进一步追究“为什么某个部署栈更慢”，应继续下钻具体运行参数，例如 `llama.cpp` 的 `n_batch`、线程数、Flash Attention、GPU offload 策略，或 ExLlamaV2 的缓存/切分策略，而不是仅凭当前报告直接归因到引擎本身。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_single_config(
    cfg: dict[str, Any],
    benchmark_plan: dict[str, Any],
    *,
    args: argparse.Namespace,
    results_dir: Path,
    gpu_id: int,
) -> dict[str, Any]:
    name = str(cfg["name"])
    benchmark_json = results_dir / f"{name}_benchmark.json"
    benchmark_log = LOGS_DIR / f"{name}_benchmark.log"

    benchmark_ok = False
    benchmark_status = str(benchmark_plan.get("status", "skipped"))

    stop_event = threading.Event()
    stop_event.gpu_id = gpu_id
    stop_event.poll_interval_sec = 0.5
    monitor_thread = threading.Thread(target=monitor_vram, args=(stop_event,), daemon=True)
    monitor_thread.start()

    try:
        if bool(benchmark_plan.get("enabled", False)):
            benchmark_command = build_benchmark_command(
                cfg,
                benchmark_plan["artifact"],
                args=args,
                output_json=benchmark_json,
            )
            try:
                print_info(f"[{name}] 开始执行 benchmark")
                run_command(benchmark_command, log_path=benchmark_log, gpu_id=gpu_id)
                benchmark_ok = True
                benchmark_status = "success"
            except subprocess.CalledProcessError as exc:
                persist_failure_log(benchmark_command, exc, log_path=benchmark_log)
                benchmark_status = classify_failure(exc)
                print_error(f"[{name}] benchmark 执行失败，已记录日志：{benchmark_log}")
        else:
            print_warning(f"[{name}] 跳过 benchmark：{benchmark_plan.get('reason')}")
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

    status = benchmark_status
    oom_flag = benchmark_status == "oom"
    if oom_flag:
        print_error(f"[{name}] benchmark 检测到疑似 OOM，显存标记为 OOM。")

    benchmark_report = load_json_if_exists(benchmark_json) if benchmark_ok else {}
    fallback_peak = float(benchmark_report.get("peak_memory", 0.0) or 0.0)
    peak_display = to_peak_vram_display(max(peak_vram_mb, fallback_peak), oom=oom_flag)

    return summarize_case_result(
        cfg,
        benchmark_report,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        peak_vram_mb=peak_display,
        status=status,
        benchmark_ok=benchmark_ok,
        benchmark_status=benchmark_status,
        benchmark_path=benchmark_json,
        benchmark_plan=benchmark_plan,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.results_dir)
    ensure_dir(LOGS_DIR)
    ensure_dir(TEMP_DIR)

    gpu_id = infer_gpu_id(args.gpu_id)
    print_info(f"实验09开始，目标 GPU = {gpu_id}，batch_size = {args.batch_size}，num_samples = {args.num_samples}")

    rows: list[dict[str, Any]] = []
    for cfg in test_configs:
        benchmark_artifact = clone_artifact(MODEL_ARTIFACTS[str(cfg["artifact_key"])])
        benchmark_plan = make_runtime_plan(
            backend=str(cfg["backend"]),
            quantization=map_quantization_for_cli(str(cfg["quant"])),
            artifact=benchmark_artifact,
        )
        benchmark_enabled, benchmark_status, benchmark_reason = prepare_runtime_plan(benchmark_plan, args=args)
        benchmark_plan["enabled"] = benchmark_enabled
        benchmark_plan["status"] = "pending" if benchmark_enabled else benchmark_status
        benchmark_plan["reason"] = benchmark_reason

        row = run_single_config(
            cfg,
            benchmark_plan,
            args=args,
            results_dir=args.results_dir,
            gpu_id=gpu_id,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "exp5_engine_speed_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    draw_figures(df, output_dir=args.results_dir)

    markdown_path = args.results_dir / "exp5_speed_report.md"
    write_markdown_report(df, output_path=markdown_path)

    summary_path = args.results_dir / "exp5_engine_speed_summary.json"
    summary_payload = {
        "results_csv": str(csv_path.resolve()),
        "results_markdown": str(markdown_path.resolve()),
        "num_cases": len(rows),
        "comparison_scope": "本地部署栈端到端速度与资源对比，不解释为纯引擎优劣",
        "execution_policy": {
            "require_gpu": True,
            "cuda_visible_devices_bound_to_requested_gpu": True,
        },
        "metric_policy": {
            "include_accuracy_metrics": False,
            "include_speed_metrics": True,
            "include_resource_metrics": True,
            "notes": [
                "不统计准确率，因为参与方案均为未针对当前任务微调的基座模型。",
                "仅保留平均/分位延迟、样本吞吐、显存与进程 RSS 等速度与资源指标。",
                "所有子进程均强制 GPU-only 执行；若无法在 GPU 上初始化，将直接失败而不是退回 CPU。",
            ],
        },
        "fairness_notes": [
            "三组方案统一使用相同的 prompts、batch size、num samples、max_new_tokens 与 max_model_len。",
            "Transformers_BNB_4bit 与 LlamaCPP_GGUF_Q4_K_M 都属于 4bit GPU 部署方案，但量化格式不同。",
            "ExLlamaV2_EXL2_LocalAsset 已纳入统一 GPU-only 基准，但本地 EXL2 资产 README 标注 Bits 8.0，因此不纳入严格同构 4bit 主结论。",
            "当前结果反映部署栈整体表现，而不是同构量化条件下的纯引擎上限。",
        ],
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print_info(f"CSV 汇总已导出：{csv_path}")
    print_info(f"Markdown 报告已导出：{markdown_path}")
    print_info(f"JSON 摘要已导出：{summary_path}")
    print_info(f"图表已导出到目录：{args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
