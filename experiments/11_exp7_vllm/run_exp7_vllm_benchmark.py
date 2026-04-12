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
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_TEST_FILE = REPO_ROOT / "data_prepare" / "splits" / "test.json"
DEFAULT_DATASET_FILE = REPO_ROOT / "data_prepare" / "genesis_franka_toolcall_alpaca.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import configure_report_matplotlib, pick_plot_text
from src.finetune_core.metrics import GPUMonitor
from src.utils.run_meta import record_run_meta
from src.utils.vllm_compat import get_vllm_environment_compat_error


BATCH_SIZE = 1
BENCHMARK_NUM_SAMPLES = 200
ACCURACY_NUM_SAMPLES = 200
MAX_NEW_TOKENS = 128
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.9
POST_RUN_SLEEP_SECONDS = 15
VRAM_POLL_INTERVAL_SEC = 0.2


CASE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Transformers_16bit",
        "backend": "transformers",
        "runtime_quantization": None,
        "vllm_dtype": None,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "report_quantization": "16bit",
        "artifact_key": "merged_fp16",
        "stack_label": "Transformers + FP16",
        "format_label": "HF Safetensors",
        "quant_note": "Transformers 直接以 float16 加载 merged 模型。",
    },
    {
        "name": "vLLM_AWQ",
        "backend": "vllm",
        "runtime_quantization": "awq",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.80,
        "report_quantization": "awq",
        "artifact_key": "merged_awq",
        "stack_label": "vLLM + AWQ",
        "format_label": "AWQ",
        "quant_note": "使用离线 AWQ 量化模型目录，由 vLLM 直接加载，并显式使用 float16 以满足 AWQ dtype 要求。",
    },
    {
        "name": "vLLM_GGUF",
        "backend": "vllm",
        "runtime_quantization": None,
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.80,
        "report_quantization": "gguf",
        "artifact_key": "merged_gguf",
        "stack_label": "vLLM + GGUF",
        "format_label": "GGUF",
        "quant_note": "使用 GGUF 文件格式模型，由 vLLM 通过 load_format=gguf 加载，并降低显存利用率阈值以适配 8GB 级显卡。",
    },
]


MODEL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "merged_fp16": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
    "merged_awq": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-awq",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-awq",
        "hf_repo_id": None,
        "allow_patterns": None,
    },
    "merged_gguf": {
        "model_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged-q4_k_m.f16.gguf",
        "tokenizer_path": REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
        "hf_repo_id": None,
        "allow_patterns": ["*.gguf"],
        "download_hint": "当前仓库未配置 GGUF 自动下载源，请先本地准备 GGUF 模型文件。",
    },
}


BACKEND_DEPENDENCIES: dict[str, list[dict[str, str]]] = {
    "transformers": [
        {"import_name": "transformers", "pip_name": "transformers"},
        {"import_name": "torch", "pip_name": "torch"},
    ],
    "vllm": [
        {"import_name": "vllm", "pip_name": "vllm"},
        {"import_name": "torch", "pip_name": "torch"},
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验11 Exp7：Transformers 16bit / vLLM AWQ / vLLM GGUF 速度与精度对比。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--gpu-id", type=int, default=None, help="nvidia-smi 监控的物理 GPU 编号，默认自动推断。")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--benchmark-prompts-file", type=Path, default=DEFAULT_BENCHMARK_PROMPTS)
    parser.add_argument("--benchmark-num-samples", type=int, default=BENCHMARK_NUM_SAMPLES)
    parser.add_argument("--accuracy-num-samples", type=int, default=ACCURACY_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--accuracy-seed", type=int, default=42)
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


def infer_gpu_id(explicit_gpu_id: int | None) -> int:
    if explicit_gpu_id is not None:
        return int(explicit_gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


def clone_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return dict(artifact)


def build_case_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for cfg in CASE_CONFIGS:
        case = dict(cfg)
        case["artifact"] = clone_artifact(MODEL_ARTIFACTS[str(cfg["artifact_key"])])
        cases.append(case)
    return cases


def ensure_backend_dependencies(backend: str, *, auto_install: bool) -> tuple[bool, str | None]:
    deps = list(BACKEND_DEPENDENCIES.get(backend, []))
    for dep in deps:
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

    repo_id = artifact.get("hf_repo_id")
    if not auto_download:
        hint = artifact.get("download_hint")
        if hint:
            return False, f"模型缺失：{model_path}。{hint}"
        return False, f"模型缺失：{model_path}。可使用 --auto-download-missing-models 自动下载。"
    if not repo_id:
        return False, f"模型缺失：{model_path}。当前未配置可自动下载的 Hugging Face Repo ID。"

    try:
        target_dir = model_path if model_path.suffix == "" else model_path.parent
        print_info(f"模型不存在，开始从 Hugging Face 下载：{repo_id} -> {target_dir}")
        download_model_from_hf(
            repo_id=str(repo_id),
            target_dir=target_dir,
            hf_token=hf_token,
            allow_patterns=artifact.get("allow_patterns"),
        )
    except Exception as exc:
        return False, f"下载模型失败：{repo_id} -> {model_path}，错误：{exc}"

    if not model_path.exists():
        if model_path.suffix.lower() == ".gguf" and model_path.parent.exists():
            matches = list(model_path.parent.rglob("*.gguf"))
            if matches:
                artifact["model_path"] = matches[0]
                return True, None
        return False, f"模型下载完成后仍未找到目标路径：{model_path}"
    return True, None


def prepare_case(case_cfg: dict[str, Any], *, args: argparse.Namespace) -> tuple[bool, str]:
    if str(case_cfg["backend"]) == "vllm":
        compat_error = get_vllm_environment_compat_error()
        if compat_error:
            return False, compat_error
    ok, reason = ensure_backend_dependencies(str(case_cfg["backend"]), auto_install=args.auto_install_deps)
    if not ok:
        return False, reason or "依赖检查失败。"
    ok, reason = ensure_model_artifact(
        case_cfg["artifact"],
        auto_download=args.auto_download_missing_models,
        hf_token=args.hf_token,
    )
    if not ok:
        return False, reason or "模型检查失败。"
    return True, ""


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


def run_command_with_process_vram(
    command: list[str],
    *,
    log_path: Path,
    gpu_id: int,
    poll_interval_sec: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    env = build_runtime_env(gpu_id=gpu_id)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    monitor = GPUMonitor(gpu_indices=[gpu_id], interval_sec=max(0.05, float(poll_interval_sec)), target_pid=proc.pid)
    monitor.start()
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate()
    finally:
        monitor.stop()

    vram_summary = monitor.summary()
    ensure_dir(log_path.parent)
    log_payload = {
        "target_pid": proc.pid,
        "vram_summary": vram_summary,
    }
    log_path.write_text(
        f"$ {' '.join(command)}\n\n[vram]\n{json.dumps(log_payload, ensure_ascii=False, indent=2)}\n\n[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command, output=stdout, stderr=stderr)
    return (
        subprocess.CompletedProcess(command, proc.returncode, stdout=stdout, stderr=stderr),
        vram_summary,
    )


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


def extract_process_peak_vram_mb(vram_summary: dict[str, Any] | None) -> float:
    if not isinstance(vram_summary, dict):
        return 0.0
    value = vram_summary.get("peak_vram_mb", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def inject_accuracy_process_vram_metrics(accuracy_report: dict[str, Any], vram_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(accuracy_report, dict):
        accuracy_report = {}
    peak_vram_mb = extract_process_peak_vram_mb(vram_summary)
    avg_vram_mb = 0.0
    if isinstance(vram_summary, dict):
        try:
            avg_vram_mb = float(vram_summary.get("avg_vram_mb", 0.0) or 0.0)
        except (TypeError, ValueError):
            avg_vram_mb = 0.0
        accuracy_report["process_vram_summary"] = vram_summary
    accuracy_report["avg_peak_vram_mb"] = avg_vram_mb
    accuracy_report["max_peak_vram_mb"] = peak_vram_mb
    samples = accuracy_report.get("samples")
    if isinstance(samples, list):
        for item in samples:
            if isinstance(item, dict):
                item["peak_vram_mb"] = peak_vram_mb
    return accuracy_report


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


def map_quantization_for_cli(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return str(value)


def resolve_case_gpu_memory_utilization(case_cfg: dict[str, Any], args: argparse.Namespace) -> float:
    value = case_cfg.get("gpu_memory_utilization", args.gpu_memory_utilization)
    return float(value)


def resolve_case_vllm_dtype(case_cfg: dict[str, Any]) -> str | None:
    value = case_cfg.get("vllm_dtype")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_benchmark_command(
    case_cfg: dict[str, Any],
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
        str(case_cfg["backend"]),
        "--model-path",
        str(Path(artifact["model_path"]).resolve()),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.benchmark_num_samples),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(resolve_case_gpu_memory_utilization(case_cfg, args)),
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

    quant = map_quantization_for_cli(case_cfg.get("runtime_quantization"))
    if quant is not None:
        command.extend(["--quantization", quant])
    vllm_dtype = resolve_case_vllm_dtype(case_cfg)
    if str(case_cfg["backend"]) == "vllm" and vllm_dtype is not None:
        command.extend(["--vllm-dtype", vllm_dtype])
    return command


def build_accuracy_override_payload(
    case_cfg: dict[str, Any],
    artifact: dict[str, Any],
    *,
    args: argparse.Namespace,
    report_file: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "test": {
            "accuracy_eval": {
                "mode": "local",
                "test_file": str(Path(args.test_file).resolve()),
                "dataset_file": str(Path(args.dataset_file).resolve()),
                "report_file": str(report_file.resolve()),
                "num_samples": int(args.accuracy_num_samples),
                "seed": int(args.accuracy_seed),
                "model_path": str(Path(artifact["model_path"]).resolve()),
                "tokenizer_path": str(Path(artifact["tokenizer_path"]).resolve()) if artifact.get("tokenizer_path") else None,
                "backend": str(case_cfg["backend"]),
                "quantization": case_cfg.get("runtime_quantization"),
                "max_new_tokens": int(args.max_new_tokens),
                "max_model_len": int(args.max_model_len),
                "gpu_memory_utilization": resolve_case_gpu_memory_utilization(case_cfg, args),
                "vllm_dtype": resolve_case_vllm_dtype(case_cfg),
                "trust_remote_code": True,
                "temperature": 0.0,
            }
        }
    }
    return payload


def write_accuracy_override(
    case_cfg: dict[str, Any],
    artifact: dict[str, Any],
    *,
    args: argparse.Namespace,
    report_file: Path,
    override_path: Path,
) -> Path:
    ensure_dir(override_path.parent)
    payload = build_accuracy_override_payload(case_cfg, artifact, args=args, report_file=report_file)
    override_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return override_path


def build_accuracy_command(*, args: argparse.Namespace, override_path: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "cli.py"),
        "eval",
        "accuracy",
        "--base-config",
        str(Path(args.base_config).resolve()),
        "--config",
        str(override_path.resolve()),
    ]


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


def peak_vram_for_plot(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def summarize_case_result(
    case_cfg: dict[str, Any],
    benchmark_report: dict[str, Any],
    accuracy_report: dict[str, Any],
    *,
    benchmark_peak_vram_mb: float | None,
    benchmark_status: str,
    accuracy_status: str,
    benchmark_path: Path,
    accuracy_path: Path,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    benchmark_ok = benchmark_status == "success"
    accuracy_ok = accuracy_status == "success"
    overall_status = "success" if benchmark_ok and accuracy_ok else f"benchmark={benchmark_status};accuracy={accuracy_status}"
    return {
        "Name": case_cfg["name"],
        "Stack Label": case_cfg["stack_label"],
        "Backend": case_cfg["backend"],
        "Quantization": case_cfg["report_quantization"],
        "Runtime Quantization": str(case_cfg.get("runtime_quantization") or ""),
        "Model Format": case_cfg["format_label"],
        "Quantization Note": case_cfg["quant_note"],
        "Benchmark Num Samples": int(benchmark_report.get("num_samples", 0) or 0) if benchmark_ok else 0,
        "Benchmark Avg Latency (s)": _get_float(benchmark_report, "avg_latency") if benchmark_ok else math.nan,
        "Benchmark P50 Latency (s)": _get_float(benchmark_report, "p50_latency") if benchmark_ok else math.nan,
        "Benchmark P95 Latency (s)": _get_float(benchmark_report, "p95_latency") if benchmark_ok else math.nan,
        "Benchmark Sample Throughput (samples/s)": _get_float(benchmark_report, "sample_throughput_sps") if benchmark_ok else math.nan,
        "Benchmark Token Throughput (tokens/s)": _get_float(benchmark_report, "token_throughput_tps") if benchmark_ok else math.nan,
        "Benchmark Peak VRAM (MB)": round(float(benchmark_peak_vram_mb), 2) if isinstance(benchmark_peak_vram_mb, (int, float)) and benchmark_peak_vram_mb > 0 else math.nan,
        "Benchmark Avg Process RSS (MB)": _get_float(benchmark_report, "avg_process_rss_mb") if benchmark_ok else math.nan,
        "Accuracy Num Samples": int(accuracy_report.get("num_samples_evaluated", 0) or 0) if accuracy_ok else 0,
        "Parse OK Rate": _get_float(accuracy_report, "parse_ok_rate") if accuracy_ok else math.nan,
        "Exact Match Rate": _get_float(accuracy_report, "exact_match_rate") if accuracy_ok else math.nan,
        "Action Match Rate": _get_float(accuracy_report, "action_match_rate") if accuracy_ok else math.nan,
        "Accuracy Avg Latency (s)": _get_float(accuracy_report, "avg_latency_sec") if accuracy_ok else math.nan,
        "Accuracy Avg Throughput (tokens/s)": _get_float(accuracy_report, "avg_throughput_tps") if accuracy_ok else math.nan,
        "Accuracy Avg Peak VRAM (MB)": _get_float(accuracy_report, "avg_peak_vram_mb") if accuracy_ok else math.nan,
        "Accuracy Max Peak VRAM (MB)": _get_float(accuracy_report, "max_peak_vram_mb") if accuracy_ok else math.nan,
        "Benchmark Status": benchmark_status,
        "Accuracy Status": accuracy_status,
        "Overall Status": overall_status,
        "Benchmark Report": str(benchmark_path.resolve()),
        "Accuracy Report": str(accuracy_path.resolve()),
        "Model Path": str(Path(artifact["model_path"]).resolve()),
        "Tokenizer Path": str(Path(artifact["tokenizer_path"]).resolve()) if artifact.get("tokenizer_path") else "",
    }


def run_single_case(
    case_cfg: dict[str, Any],
    *,
    args: argparse.Namespace,
    results_dir: Path,
    gpu_id: int,
) -> dict[str, Any]:
    artifact = case_cfg["artifact"]
    name = str(case_cfg["name"])
    benchmark_json = results_dir / f"{name}_benchmark.json"
    accuracy_json = results_dir / f"{name}_accuracy.json"
    accuracy_override = TEMP_DIR / f"{name}_accuracy_override.yaml"
    benchmark_log = LOGS_DIR / f"{name}_benchmark.log"
    accuracy_log = LOGS_DIR / f"{name}_accuracy.log"

    benchmark_status = "pending"
    accuracy_status = "pending"
    benchmark_report: dict[str, Any] = {}
    accuracy_report: dict[str, Any] = {}
    benchmark_peak_vram_mb = 0.0
    benchmark_vram_summary: dict[str, Any] = {}
    accuracy_vram_summary: dict[str, Any] = {}

    benchmark_command = build_benchmark_command(case_cfg, artifact, args=args, output_json=benchmark_json)

    try:
        print_info(f"[{name}] 开始执行 benchmark")
        _, benchmark_vram_summary = run_command_with_process_vram(
            benchmark_command,
            log_path=benchmark_log,
            gpu_id=gpu_id,
            poll_interval_sec=args.vram_poll_interval,
        )
        benchmark_status = "success"
        benchmark_report = load_json_if_exists(benchmark_json)
    except subprocess.CalledProcessError as exc:
        persist_failure_log(benchmark_command, exc, log_path=benchmark_log)
        benchmark_status = classify_failure(exc)
        print_error(f"[{name}] benchmark 失败，已记录日志：{benchmark_log}")
    finally:
        benchmark_peak_vram_mb = extract_process_peak_vram_mb(benchmark_vram_summary)
        cleanup_runtime_state()
        print_info(f"[{name}] benchmark 后冷却 {args.sleep_seconds} 秒")
        time.sleep(args.sleep_seconds)

    override_path = write_accuracy_override(
        case_cfg,
        artifact,
        args=args,
        report_file=accuracy_json,
        override_path=accuracy_override,
    )
    accuracy_command = build_accuracy_command(args=args, override_path=override_path)

    try:
        print_info(f"[{name}] 开始执行 accuracy")
        _, accuracy_vram_summary = run_command_with_process_vram(
            accuracy_command,
            log_path=accuracy_log,
            gpu_id=gpu_id,
            poll_interval_sec=args.vram_poll_interval,
        )
        accuracy_status = "success"
        accuracy_report = load_json_if_exists(accuracy_json)
        accuracy_report = inject_accuracy_process_vram_metrics(accuracy_report, accuracy_vram_summary)
        accuracy_json.write_text(json.dumps(accuracy_report, ensure_ascii=False, indent=2), encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        persist_failure_log(accuracy_command, exc, log_path=accuracy_log)
        accuracy_status = classify_failure(exc)
        print_error(f"[{name}] accuracy 失败，已记录日志：{accuracy_log}")
    finally:
        cleanup_runtime_state()
        print_info(f"[{name}] accuracy 后冷却 {args.sleep_seconds} 秒")
        time.sleep(args.sleep_seconds)

    return summarize_case_result(
        case_cfg,
        benchmark_report,
        accuracy_report,
        benchmark_peak_vram_mb=benchmark_peak_vram_mb,
        benchmark_status=benchmark_status,
        accuracy_status=accuracy_status,
        benchmark_path=benchmark_json,
        accuracy_path=accuracy_json,
        artifact=artifact,
    )


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

    labels = df["Name"].tolist()
    x = list(range(len(df)))

    fig1, ax1 = plt.subplots(figsize=(16, 8))
    lat_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Avg Latency (s)"].tolist()]
    ax1.bar(labels, lat_values, color="#4c78a8")
    ax1.set_title(pick_plot_text("图1：三方案速度对比", "Figure 1: Benchmark Latency Comparison"))
    ax1.set_xlabel(pick_plot_text("方案", "Case"))
    ax1.set_ylabel("Seconds")
    ax1.tick_params(axis="x", rotation=15)
    fig1.tight_layout()
    fig1.savefig(output_dir / "exp7_vllm_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 8))
    token_tps = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Token Throughput (tokens/s)"].tolist()]
    ax2.bar(labels, token_tps, color="#54a24b")
    ax2.set_title(pick_plot_text("图2：三方案吞吐对比", "Figure 2: Benchmark Throughput Comparison"))
    ax2.set_xlabel(pick_plot_text("方案", "Case"))
    ax2.set_ylabel("Tokens / Second")
    ax2.tick_params(axis="x", rotation=15)
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp7_vllm_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(16, 8))
    width = 0.28
    exact_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Exact Match Rate"].tolist()]
    action_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Action Match Rate"].tolist()]
    parse_values = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Parse OK Rate"].tolist()]
    ax3.bar([item - width for item in x], parse_values, width=width, color="#72b7b2", label="Parse OK")
    ax3.bar(x, exact_values, width=width, color="#f58518", label="Exact Match")
    ax3.bar([item + width for item in x], action_values, width=width, color="#e45756", label="Action Match")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title(pick_plot_text("图3：三方案精度对比", "Figure 3: Accuracy Comparison"))
    ax3.set_xlabel(pick_plot_text("方案", "Case"))
    ax3.set_ylabel("Rate")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp7_vllm_accuracy_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(16, 8))
    benchmark_vram = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Benchmark Peak VRAM (MB)"].tolist()]
    accuracy_vram = [float(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0.0 for v in df["Accuracy Max Peak VRAM (MB)"].tolist()]
    ax4.bar([item - 0.18 for item in x], benchmark_vram, width=0.36, color="#b279a2", label="Benchmark Peak VRAM")
    ax4.bar([item + 0.18 for item in x], accuracy_vram, width=0.36, color="#bab0ab", label="Accuracy Max VRAM")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15)
    ax4.set_title(pick_plot_text("图4：显存占用对比", "Figure 4: VRAM Comparison"))
    ax4.set_xlabel(pick_plot_text("方案", "Case"))
    ax4.set_ylabel("MB")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(output_dir / "exp7_vllm_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)


def write_markdown_report(df: pd.DataFrame, *, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    lines: list[str] = [
        "# Exp7 推理部署对比实验报告",
        "",
        "## 实验目标",
        "",
        "- 将原来的单一 `vLLM` 显存预算实验改为三方案统一对比。",
        "- 当前对比矩阵为 `Transformers 16bit`、`vLLM AWQ`、`vLLM GGUF`。",
        "- 每个方案都同时执行 benchmark 与 accuracy，分别观察速度、吞吐、解析率、精确匹配率和动作匹配率。",
        "",
        "## 公平性口径",
        "",
        "- 三组方案统一使用同一份 `merged` 系列模型资产，只改变加载后端或模型格式。",
        "- benchmark 统一使用相同 prompts、batch size、num samples、max_new_tokens、max_model_len。",
        "- accuracy 统一使用相同测试集、相同随机种子、相同 system prompt 与生成参数。",
        "- `Transformers 16bit` 代表未量化基线；`vLLM AWQ` 与 `vLLM GGUF` 代表两种不同的部署格式。",
        "",
        "## 速度结果",
        "",
        "| 方案 | Backend | 载入格式 | Avg Latency (s) | P50 (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {backend} | {fmt} | {avg} | {p50} | {p95} | {sps} | {tps} | {vram} | {status} |".format(
                name=row["Name"],
                backend=row["Backend"],
                fmt=row["Model Format"],
                avg=_fmt_metric(row["Benchmark Avg Latency (s)"]),
                p50=_fmt_metric(row["Benchmark P50 Latency (s)"]),
                p95=_fmt_metric(row["Benchmark P95 Latency (s)"]),
                sps=_fmt_metric(row["Benchmark Sample Throughput (samples/s)"]),
                tps=_fmt_metric(row["Benchmark Token Throughput (tokens/s)"]),
                vram=_fmt_metric(row["Benchmark Peak VRAM (MB)"], digits=2),
                status=row["Benchmark Status"],
            )
        )

    lines.extend(
        [
            "",
            "## 精度结果",
            "",
            "| 方案 | Parse OK | Exact Match | Action Match | Accuracy Avg Latency (s) | Accuracy Tokens/s | Accuracy Max VRAM (MB) | 状态 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {parse} | {exact} | {action} | {lat} | {tps} | {vram} | {status} |".format(
                name=row["Name"],
                parse=_fmt_metric(row["Parse OK Rate"]),
                exact=_fmt_metric(row["Exact Match Rate"]),
                action=_fmt_metric(row["Action Match Rate"]),
                lat=_fmt_metric(row["Accuracy Avg Latency (s)"]),
                tps=_fmt_metric(row["Accuracy Avg Throughput (tokens/s)"]),
                vram=_fmt_metric(row["Accuracy Max Peak VRAM (MB)"], digits=2),
                status=row["Accuracy Status"],
            )
        )

    success_df = df[df["Overall Status"] == "success"].copy()
    lines.extend(["", "## 结果分析", ""])
    if success_df.empty:
        lines.append("- 当前没有方案同时完成 benchmark 与 accuracy，请优先检查对应日志。")
    else:
        fastest_row = success_df.sort_values("Benchmark Avg Latency (s)", ascending=True).iloc[0]
        best_exact_row = success_df.sort_values("Exact Match Rate", ascending=False).iloc[0]
        best_action_row = success_df.sort_values("Action Match Rate", ascending=False).iloc[0]
        lines.append(
            f"- 速度最优方案为 `{fastest_row['Name']}`，其 benchmark 平均延迟为 `{float(fastest_row['Benchmark Avg Latency (s)']):.4f}s`。"
        )
        lines.append(
            f"- `Exact Match` 最高的方案为 `{best_exact_row['Name']}`，精确匹配率为 `{float(best_exact_row['Exact Match Rate']):.4f}`。"
        )
        lines.append(
            f"- `Action Match` 最高的方案为 `{best_action_row['Name']}`，动作匹配率为 `{float(best_action_row['Action Match Rate']):.4f}`。"
        )
        lines.append("- 如果速度与精度最优方案不是同一个，就说明当前实验存在明显的部署权衡。")

    lines.extend(
        [
            "",
            "## 解读建议",
            "",
            "- 如果你要写“部署效率”，优先引用 benchmark 的 `Avg Latency`、`Tokens/s` 和 `Peak VRAM`。",
            "- 如果你要写“任务可用性”，优先引用 accuracy 的 `Parse OK Rate`、`Exact Match Rate` 和 `Action Match Rate`。",
            "- `vLLM GGUF` 这里表示以 GGUF 文件格式加载；它与 `vLLM AWQ` 不是同一种量化机制，因此结论应写成“当前仓库内三种加载方案的端到端表现对比”，不要写成“同构量化下的纯引擎结论”。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.results_dir)
    ensure_dir(LOGS_DIR)
    ensure_dir(TEMP_DIR)

    gpu_id = infer_gpu_id(args.gpu_id)
    print_info(
        "Exp7 对比实验开始："
        f" GPU={gpu_id}, benchmark_samples={args.benchmark_num_samples}, accuracy_samples={args.accuracy_num_samples}"
    )

    rows: list[dict[str, Any]] = []
    cases = build_case_matrix()
    for case_cfg in cases:
        ok, reason = prepare_case(case_cfg, args=args)
        if not ok:
            print_error(f"[{case_cfg['name']}] 预检查失败：{reason}")
            rows.append(
                {
                    "Name": case_cfg["name"],
                    "Stack Label": case_cfg["stack_label"],
                    "Backend": case_cfg["backend"],
                    "Quantization": case_cfg["report_quantization"],
                    "Runtime Quantization": str(case_cfg.get("runtime_quantization") or ""),
                    "Model Format": case_cfg["format_label"],
                    "Quantization Note": case_cfg["quant_note"],
                    "Benchmark Num Samples": 0,
                    "Benchmark Avg Latency (s)": math.nan,
                    "Benchmark P50 Latency (s)": math.nan,
                    "Benchmark P95 Latency (s)": math.nan,
                    "Benchmark Sample Throughput (samples/s)": math.nan,
                    "Benchmark Token Throughput (tokens/s)": math.nan,
                    "Benchmark Peak VRAM (MB)": math.nan,
                    "Benchmark Avg Process RSS (MB)": math.nan,
                    "Accuracy Num Samples": 0,
                    "Parse OK Rate": math.nan,
                    "Exact Match Rate": math.nan,
                    "Action Match Rate": math.nan,
                    "Accuracy Avg Latency (s)": math.nan,
                    "Accuracy Avg Throughput (tokens/s)": math.nan,
                    "Accuracy Avg Peak VRAM (MB)": math.nan,
                    "Accuracy Max Peak VRAM (MB)": math.nan,
                    "Benchmark Status": "precheck_failed",
                    "Accuracy Status": "precheck_failed",
                    "Overall Status": reason,
                    "Benchmark Report": "",
                    "Accuracy Report": "",
                    "Model Path": str(Path(case_cfg["artifact"]["model_path"]).resolve()),
                    "Tokenizer Path": str(Path(case_cfg["artifact"]["tokenizer_path"]).resolve()) if case_cfg["artifact"].get("tokenizer_path") else "",
                }
            )
            continue

        rows.append(run_single_case(case_cfg, args=args, results_dir=args.results_dir, gpu_id=gpu_id))

    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "exp7_vllm_engine_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    draw_figures(df, output_dir=args.results_dir)

    markdown_path = args.results_dir / "exp7_vllm_report.md"
    write_markdown_report(df, output_path=markdown_path)

    summary_path = args.results_dir / "exp7_vllm_summary.json"
    success_df = df[df["Overall Status"] == "success"].copy()
    summary_payload = {
        "experiment": "exp7_vllm_engine_compare",
        "comparison_scope": "Transformers 16bit vs vLLM AWQ vs vLLM GGUF",
        "benchmark_num_samples": int(args.benchmark_num_samples),
        "accuracy_num_samples": int(args.accuracy_num_samples),
        "gpu_id": gpu_id,
        "fairness_notes": [
            "三组方案统一使用同一套 merged 模型系列资产与相同的评测参数。",
            "benchmark 与 accuracy 分开执行，避免前一个引擎实例残留影响后一个 case。",
            "GGUF 与 AWQ 不是同构量化格式，结论应理解为部署方案对比，而不是纯量化算法优劣。",
        ],
        "best_benchmark_latency_case": (
            success_df.sort_values("Benchmark Avg Latency (s)", ascending=True).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "best_exact_match_case": (
            success_df.sort_values("Exact Match Rate", ascending=False).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "best_action_match_case": (
            success_df.sort_values("Action Match Rate", ascending=False).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = record_run_meta(
        args.results_dir,
        cli_args=vars(args),
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        data_paths=[args.benchmark_prompts_file, args.test_file, args.dataset_file],
        extra_meta={
            "entry": "experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py",
            "stage": "exp7_engine_compare",
            "gpu_id": gpu_id,
            "summary_path": str(summary_path.resolve()),
            "cases": [case["name"] for case in CASE_CONFIGS],
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
