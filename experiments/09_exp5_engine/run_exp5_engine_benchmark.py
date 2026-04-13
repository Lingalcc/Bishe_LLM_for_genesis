#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TEMP_DIR = EXPERIMENT_DIR / ".cache"
PREQUANTIZED_MODEL_DIR = TEMP_DIR / "prequantized_models"
PREQUANTIZED_EXPORT_SCRIPT = EXPERIMENT_DIR / "export_prequantized_bnb_model.py"
DEFAULT_BENCHMARK_PROMPTS = EXPERIMENT_DIR / "prompts" / "default_prompts.json"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_TEST_FILE = REPO_ROOT / "data_prepare" / "splits" / "test.json"
DEFAULT_DATASET_FILE = REPO_ROOT / "data_prepare" / "genesis_franka_toolcall_alpaca.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting import configure_report_matplotlib, pick_plot_text
from src.utils.run_meta import record_run_meta
from src.utils.vllm_compat import get_vllm_environment_compat_error


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
}


CASE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Transformers_16bit",
        "family": "Transformers",
        "backend": "transformers",
        "runtime_quantization": None,
        "report_quantization": "16bit",
        "artifact_key": "merged_fp16",
        "stack_label": "Transformers + FP16",
        "format_label": "HF Safetensors",
        "quant_note": "直接加载 merged safetensors 模型，作为未量化基线。",
    },
    {
        "name": "Transformers_8bit",
        "family": "Transformers",
        "backend": "transformers",
        "runtime_quantization": "8bit",
        "report_quantization": "8bit",
        "artifact_key": "merged_fp16",
        "stack_label": "Transformers + bitsandbytes 8bit",
        "format_label": "HF Safetensors",
        "quant_note": "运行时使用 bitsandbytes 8bit 量化加载 merged 模型。",
    },
    {
        "name": "Transformers_4bit",
        "family": "Transformers",
        "backend": "transformers",
        "runtime_quantization": "4bit",
        "report_quantization": "4bit",
        "artifact_key": "merged_fp16",
        "stack_label": "Transformers + bitsandbytes 4bit",
        "format_label": "HF Safetensors",
        "quant_note": "运行时使用 bitsandbytes NF4 4bit 量化加载 merged 模型。",
    },
    {
        "name": "vLLM_16bit",
        "family": "vLLM",
        "backend": "vllm",
        "runtime_quantization": None,
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.90,
        "report_quantization": "16bit",
        "artifact_key": "merged_fp16",
        "stack_label": "vLLM + FP16",
        "format_label": "HF Safetensors",
        "quant_note": "vLLM 直接加载 merged safetensors 模型，作为 vLLM 未量化基线。",
    },
    {
        "name": "vLLM_8bit",
        "family": "vLLM",
        "backend": "vllm",
        "runtime_quantization": "bitsandbytes",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.85,
        "bitsandbytes_mode": "8bit",
        "report_quantization": "8bit",
        "artifact_key": "merged_fp16",
        "stack_label": "vLLM + bitsandbytes 8bit",
        "format_label": "HF Pre-Quantized BNB",
        "quant_note": "先将 merged 模型导出为 bitsandbytes 8bit 预量化目录，再由 vLLM 直接读取该目录。",
    },
    {
        "name": "vLLM_4bit",
        "family": "vLLM",
        "backend": "vllm",
        "runtime_quantization": "bitsandbytes",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.85,
        "bitsandbytes_mode": "4bit",
        "report_quantization": "4bit",
        "artifact_key": "merged_fp16",
        "stack_label": "vLLM + bitsandbytes 4bit",
        "format_label": "HF Pre-Quantized BNB",
        "quant_note": "先将 merged 模型导出为 bitsandbytes 4bit 预量化目录，再由 vLLM 直接读取该目录。",
    },
    {
        "name": "vLLM_AWQ",
        "family": "vLLM",
        "backend": "vllm",
        "runtime_quantization": "compressed-tensors",
        "vllm_dtype": "float16",
        "gpu_memory_utilization": 0.80,
        "report_quantization": "awq",
        "artifact_key": "merged_awq",
        "stack_label": "vLLM + AWQ",
        "format_label": "Compressed Tensors (AWQ)",
        "quant_note": "加载 llmcompressor 导出的 AWQ 压缩目录，走 vLLM compressed-tensors 兼容路径。",
    },
]


def _load_exp7_module() -> Any:
    module_path = REPO_ROOT / "experiments" / "11_exp7_vllm" / "run_exp7_vllm_benchmark.py"
    spec = importlib.util.spec_from_file_location("exp7_vllm_benchmark_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 exp7 模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP7 = _load_exp7_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实验09 Exp5：Transformers / vLLM 的 7 组量化部署性能与精度对比。"
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--gpu-id", type=int, default=None, help="nvidia-smi 监控的物理 GPU 编号，默认自动推断。")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--benchmark-prompts-file", type=Path, default=DEFAULT_BENCHMARK_PROMPTS)
    parser.add_argument("--benchmark-num-samples", type=int, default=200)
    parser.add_argument("--accuracy-num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--accuracy-seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=int, default=15)
    parser.add_argument("--vram-poll-interval", type=float, default=0.2)
    parser.add_argument("--auto-install-deps", action="store_true", help="缺少依赖时自动执行 pip install。")
    parser.add_argument("--auto-download-missing-models", action="store_true", help="缺少模型资产时尝试自动下载。")
    parser.add_argument("--skip-vllm-compat-check", action="store_true", help="显式跳过当前 vLLM 环境兼容性保守检查。")
    parser.add_argument(
        "--auto-export-bnb-models",
        action="store_true",
        help="当 vLLM 8bit/4bit 的预量化目录缺失时，自动调用导出脚本生成。",
    )
    parser.add_argument(
        "--force-reexport-bnb-models",
        action="store_true",
        help="强制重新导出 vLLM 8bit/4bit 的预量化目录，会覆盖旧导出结果。",
    )
    parser.add_argument(
        "--case-names",
        type=str,
        default="",
        help="只运行指定 case，使用逗号分隔，例如 vLLM_16bit,vLLM_8bit。",
    )
    parser.add_argument(
        "--reuse-existing-summary",
        type=Path,
        default=None,
        help="先读取已有 summary.json 中的 rows，再把本轮新结果按 Name 合并进去。",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "",
    )
    return parser.parse_args(argv)


def _clone_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return dict(artifact)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prequantized_dir_name(source_model_dir: Path, mode: str) -> str:
    normalized = str(mode).strip().lower()
    return f"{source_model_dir.name}-bnb-{normalized}"


def _ensure_python_dependencies(
    dependencies: list[dict[str, str]],
    *,
    auto_install: bool,
) -> tuple[bool, str | None]:
    for dep in dependencies:
        if EXP7.maybe_import(dep["import_name"]):
            continue
        if not auto_install:
            return False, f"缺少依赖 {dep['import_name']}，可使用 --auto-install-deps 自动安装。"
        try:
            EXP7.install_python_package(dep["pip_name"])
        except Exception as exc:
            return False, f"安装依赖 {dep['pip_name']} 失败：{exc}"
        if not EXP7.maybe_import(dep["import_name"]):
            return False, f"依赖 {dep['import_name']} 安装后仍不可用。"
    return True, None


def _validate_prequantized_bnb_dir(model_dir: Path, *, mode: str) -> tuple[bool, str]:
    if not model_dir.exists():
        return False, f"目录不存在：{model_dir}"
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False, f"缺少配置文件：{config_path}"
    try:
        payload = _read_json(config_path)
    except Exception as exc:
        return False, f"读取配置失败：{config_path}，错误：{exc}"

    quant_cfg = payload.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        return False, "config.json 中缺少 quantization_config。"
    if str(quant_cfg.get("quant_method") or "").strip().lower() != "bitsandbytes":
        return False, "config.json 的 quantization_config.quant_method 不是 bitsandbytes。"

    normalized = str(mode).strip().lower()
    if normalized == "8bit" and not bool(quant_cfg.get("load_in_8bit")):
        return False, "config.json 未声明 load_in_8bit=true。"
    if normalized == "4bit" and not bool(quant_cfg.get("load_in_4bit")):
        return False, "config.json 未声明 load_in_4bit=true。"

    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin")) or any(model_dir.glob("*.pt"))
    if not has_weights:
        return False, "目录中未找到模型权重文件。"
    return True, ""


def _build_prequantized_export_command(
    *,
    source_model_dir: Path,
    output_dir: Path,
    mode: str,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PREQUANTIZED_EXPORT_SCRIPT.resolve()),
        "--source-model-dir",
        str(source_model_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--mode",
        str(mode),
    ]
    if force:
        command.append("--force")
    return command


def _prepare_prequantized_bnb_artifact(
    case_cfg: dict[str, Any],
    *,
    args: argparse.Namespace,
    gpu_id: int,
) -> tuple[bool, str]:
    mode = str(case_cfg.get("bitsandbytes_mode") or "").strip().lower()
    if not mode:
        return True, ""

    source_model_dir = Path(MODEL_ARTIFACTS[str(case_cfg["artifact_key"])]["model_path"])
    target_dir = PREQUANTIZED_MODEL_DIR / _prequantized_dir_name(source_model_dir, mode)
    target_path_text = str(target_dir.resolve()) if target_dir.exists() else str(target_dir)
    case_cfg["prepared_model_path"] = target_path_text
    case_cfg["model_override_path"] = target_path_text

    ok, reason = _validate_prequantized_bnb_dir(target_dir, mode=mode)
    if ok and not args.force_reexport_bnb_models:
        case_cfg["artifact"]["model_path"] = target_dir
        case_cfg["artifact"]["tokenizer_path"] = target_dir
        return True, ""

    if not args.auto_export_bnb_models:
        command_preview = " ".join(
            [
                "python",
                str(PREQUANTIZED_EXPORT_SCRIPT.relative_to(REPO_ROOT)),
                "--source-model-dir",
                str(source_model_dir),
                "--output-dir",
                str(target_dir),
                "--mode",
                mode,
            ]
        )
        if args.force_reexport_bnb_models and target_dir.exists():
            command_preview += " --force"
        if reason:
            return False, f"缺少可用的预量化目录：{reason}。请先执行 `{command_preview}`，或为主实验脚本增加 `--auto-export-bnb-models`。"
        return False, f"缺少可用的预量化目录：{target_dir}。请先执行 `{command_preview}`，或为主实验脚本增加 `--auto-export-bnb-models`。"

    export_log_path = LOGS_DIR / f"export_{target_dir.name}.log"
    command = _build_prequantized_export_command(
        source_model_dir=source_model_dir,
        output_dir=target_dir,
        mode=mode,
        force=bool(args.force_reexport_bnb_models),
    )
    EXP7.print_info(f"[{case_cfg['name']}] 开始导出预量化模型目录：{target_dir}")
    try:
        EXP7.run_command(command, log_path=export_log_path, gpu_id=gpu_id)
    except Exception as exc:
        if hasattr(EXP7, "persist_failure_log") and hasattr(exc, "returncode"):
            try:
                EXP7.persist_failure_log(command, exc, log_path=export_log_path)
            except Exception:
                pass
        return False, f"导出预量化目录失败，详见日志：{export_log_path}。错误：{exc}"

    ok, reason = _validate_prequantized_bnb_dir(target_dir, mode=mode)
    if not ok:
        return False, f"导出完成后校验失败：{reason}"

    case_cfg["artifact"]["model_path"] = target_dir
    case_cfg["artifact"]["tokenizer_path"] = target_dir
    case_cfg["prepared_model_path"] = str(target_dir.resolve())
    case_cfg["model_override_path"] = str(target_dir.resolve())
    return True, ""


def prepare_case(case_cfg: dict[str, Any], *, args: argparse.Namespace, gpu_id: int) -> tuple[bool, str]:
    if str(case_cfg["backend"]) == "vllm":
        compat_error = get_vllm_environment_compat_error()
        if compat_error:
            return False, compat_error

    ok, reason = EXP7.ensure_backend_dependencies(str(case_cfg["backend"]), auto_install=args.auto_install_deps)
    if not ok:
        return False, reason or "依赖检查失败。"

    if case_cfg.get("bitsandbytes_mode"):
        ok, reason = _ensure_python_dependencies(
            [
                {"import_name": "transformers", "pip_name": "transformers"},
                {"import_name": "bitsandbytes", "pip_name": "bitsandbytes"},
                {"import_name": "torch", "pip_name": "torch"},
            ],
            auto_install=args.auto_install_deps,
        )
        if not ok:
            return False, reason or "BNB 预量化导出依赖检查失败。"
        ok, reason = _prepare_prequantized_bnb_artifact(case_cfg, args=args, gpu_id=gpu_id)
        if not ok:
            return False, reason or "BNB 预量化目录准备失败。"

    ok, reason = EXP7.ensure_model_artifact(
        case_cfg["artifact"],
        auto_download=args.auto_download_missing_models,
        hf_token=args.hf_token,
    )
    if not ok:
        return False, reason or "模型检查失败。"
    return True, ""


def build_case_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for cfg in CASE_CONFIGS:
        case = dict(cfg)
        artifact = _clone_artifact(MODEL_ARTIFACTS[str(cfg["artifact_key"])])
        case["artifact"] = artifact
        case["prepared_model_path"] = ""
        case["model_override_path"] = ""
        cases.append(case)
    return cases


def parse_case_names(raw: str) -> list[str]:
    if not raw.strip():
        return []
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {case["name"] for case in CASE_CONFIGS}
    invalid = [name for name in requested if name not in allowed]
    if invalid:
        raise ValueError(f"不支持的 case 名称: {invalid}；可选值: {sorted(allowed)}")
    return requested


def filter_cases(cases: list[dict[str, Any]], selected_names: list[str]) -> list[dict[str, Any]]:
    if not selected_names:
        return cases
    selected = set(selected_names)
    return [case for case in cases if case["name"] in selected]


def _case_palette(df: pd.DataFrame) -> list[str]:
    palette: list[str] = []
    for _, row in df.iterrows():
        name = str(row.get("Name", ""))
        status = str(row.get("Overall Status", "")).lower()
        if "oom" in status:
            palette.append("#e45756")
        elif "failed" in status or "precheck" in status or "missing" in status:
            palette.append("#9d9d9d")
        elif name.startswith("Transformers_"):
            palette.append("#4c78a8")
        elif name == "vLLM_AWQ":
            palette.append("#f58518")
        else:
            palette.append("#54a24b")
    return palette


def _bar_values(series: pd.Series) -> list[float]:
    values: list[float] = []
    for value in series.tolist():
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
        else:
            values.append(0.0)
    return values


def _status_labels(df: pd.DataFrame, benchmark: bool = True) -> list[str]:
    labels: list[str] = []
    column = "Benchmark Status" if benchmark else "Accuracy Status"
    for _, row in df.iterrows():
        status = str(row.get(column, "")).strip()
        labels.append("" if status == "success" else status.upper())
    return labels


def _annotate_non_success(ax: Any, xs: list[float], values: list[float], labels: list[str]) -> None:
    ymax = max(values) if values else 0.0
    baseline = ymax * 0.03 if ymax > 0 else 0.03
    for x, value, label in zip(xs, values, labels):
        if not label:
            continue
        y = value + baseline if value > 0 else baseline
        ax.text(x, y, label, rotation=90, ha="center", va="bottom", fontsize=9, color="#b22222")


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

    EXP7.ensure_dir(output_dir)
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    labels = df["Name"].tolist()
    x = list(range(len(df)))
    colors = _case_palette(df)

    fig1, ax1 = plt.subplots(figsize=(18, 8))
    latency_values = _bar_values(df["Benchmark Avg Latency (s)"])
    ax1.bar(x, latency_values, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18)
    ax1.set_title(pick_plot_text("图1：7 组部署方案平均延迟对比", "Figure 1: Avg Benchmark Latency Across 7 Cases"))
    ax1.set_xlabel(pick_plot_text("方案", "Case"))
    ax1.set_ylabel("Seconds")
    _annotate_non_success(ax1, x, latency_values, _status_labels(df, benchmark=True))
    fig1.tight_layout()
    fig1.savefig(output_dir / "exp5_engine_latency_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(18, 8))
    throughput_values = _bar_values(df["Benchmark Token Throughput (tokens/s)"])
    ax2.bar(x, throughput_values, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18)
    ax2.set_title(pick_plot_text("图2：7 组部署方案 Token 吞吐对比", "Figure 2: Benchmark Token Throughput Across 7 Cases"))
    ax2.set_xlabel(pick_plot_text("方案", "Case"))
    ax2.set_ylabel("Tokens / Second")
    _annotate_non_success(ax2, x, throughput_values, _status_labels(df, benchmark=True))
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp5_engine_throughput_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(18, 8))
    width = 0.24
    parse_values = _bar_values(df["Parse OK Rate"])
    exact_values = _bar_values(df["Exact Match Rate"])
    action_values = _bar_values(df["Action Match Rate"])
    ax3.bar([item - width for item in x], parse_values, width=width, color="#72b7b2", label="Parse OK")
    ax3.bar(x, exact_values, width=width, color="#f58518", label="Exact Match")
    ax3.bar([item + width for item in x], action_values, width=width, color="#e45756", label="Action Match")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=18)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title(pick_plot_text("图3：7 组部署方案精度对比", "Figure 3: Accuracy Across 7 Cases"))
    ax3.set_xlabel(pick_plot_text("方案", "Case"))
    ax3.set_ylabel("Rate")
    ax3.legend()
    _annotate_non_success(ax3, x, action_values, _status_labels(df, benchmark=False))
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp5_engine_accuracy_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(18, 8))
    benchmark_vram = _bar_values(df["Benchmark Peak VRAM (MB)"])
    accuracy_vram = _bar_values(df["Accuracy Max Peak VRAM (MB)"])
    ax4.bar([item - 0.18 for item in x], benchmark_vram, width=0.36, color="#b279a2", label="Benchmark Peak VRAM")
    ax4.bar([item + 0.18 for item in x], accuracy_vram, width=0.36, color="#bab0ab", label="Accuracy Max VRAM")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=18)
    ax4.set_title(pick_plot_text("图4：7 组部署方案显存对比", "Figure 4: VRAM Across 7 Cases"))
    ax4.set_xlabel(pick_plot_text("方案", "Case"))
    ax4.set_ylabel("MB")
    ax4.legend()
    _annotate_non_success(ax4, x, benchmark_vram, _status_labels(df, benchmark=True))
    fig4.tight_layout()
    fig4.savefig(output_dir / "exp5_engine_memory_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)


def _fmt_metric(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    text = str(value).strip()
    return text or "-"


def write_markdown_report(df: pd.DataFrame, *, output_path: Path) -> None:
    EXP7.ensure_dir(output_path.parent)
    lines: list[str] = [
        "# Exp5 七组推理部署对比实验报告",
        "",
        "## 实验目标",
        "",
        "- 在同一份 `qwen2.5-3b-genesis-merged` 任务模型上，对比 `Transformers` 与 `vLLM` 的 16bit / 8bit / 4bit 部署表现。",
        "- 额外纳入 `vLLM + AWQ` 作为第 7 组，观察预量化压缩目录在速度、显存与精度上的落点。",
        "- 如果某组在 benchmark 或 accuracy 阶段出现显存不足，实验会记录其状态并继续执行后续组别。",
        "",
        "## 对比矩阵",
        "",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"- `{row['Name']}`：{row['Stack Label']}；量化标签 `{row['Quantization']}`；格式 `{row['Model Format']}`。"
        )

    lines.extend(
        [
            "",
            "## 速度结果",
            "",
            "| 方案 | Family | Backend | 量化 | Avg Latency (s) | P95 (s) | Samples/s | Tokens/s | Peak VRAM (MB) | 状态 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            "| {name} | {family} | {backend} | {quant} | {avg} | {p95} | {sps} | {tps} | {vram} | {status} |".format(
                name=row["Name"],
                family=row["Family"],
                backend=row["Backend"],
                quant=row["Quantization"],
                avg=_fmt_metric(row["Benchmark Avg Latency (s)"]),
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

    failed_df = df[df["Overall Status"] != "success"].copy()
    lines.extend(["", "## 异常记录", ""])
    if failed_df.empty:
        lines.append("- 本轮 7 组方案都完成了 benchmark 与 accuracy。")
    else:
        for _, row in failed_df.iterrows():
            lines.append(
                f"- `{row['Name']}`：benchmark=`{row['Benchmark Status']}`，accuracy=`{row['Accuracy Status']}`，overall=`{row['Overall Status']}`。"
            )

    success_df = df[df["Overall Status"] == "success"].copy()
    lines.extend(["", "## 结果分析", ""])
    if success_df.empty:
        lines.append("- 当前没有方案同时完成 benchmark 与 accuracy，请优先检查 `logs/` 下对应日志。")
    else:
        fastest_row = success_df.sort_values("Benchmark Avg Latency (s)", ascending=True).iloc[0]
        best_exact_row = success_df.sort_values("Exact Match Rate", ascending=False).iloc[0]
        best_action_row = success_df.sort_values("Action Match Rate", ascending=False).iloc[0]
        lowest_vram_row = success_df.sort_values("Benchmark Peak VRAM (MB)", ascending=True).iloc[0]
        lines.append(
            f"- 速度最优方案为 `{fastest_row['Name']}`，benchmark 平均延迟为 `{float(fastest_row['Benchmark Avg Latency (s)']):.4f}s`。"
        )
        lines.append(
            f"- `Exact Match` 最高方案为 `{best_exact_row['Name']}`，精确匹配率为 `{float(best_exact_row['Exact Match Rate']):.4f}`。"
        )
        lines.append(
            f"- `Action Match` 最高方案为 `{best_action_row['Name']}`，动作匹配率为 `{float(best_action_row['Action Match Rate']):.4f}`。"
        )
        lines.append(
            f"- 显存占用最低的成功方案为 `{lowest_vram_row['Name']}`，benchmark 峰值显存为 `{float(lowest_vram_row['Benchmark Peak VRAM (MB)']):.2f} MB`。"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `Transformers 8bit/4bit` 与 `vLLM 8bit/4bit` 都使用同一份 merged 模型，只是运行时后端和量化路径不同。",
            "- `vLLM 8bit/4bit` 使用的是预先导出的 bitsandbytes 量化目录，不再走运行时从 FP16 safetensors 现量化的路径。",
            "- `vLLM_AWQ` 使用的是预先压缩好的 AWQ 目录，因此它与运行时 bitsandbytes 量化并不是完全同构的量化路径。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _blank_row(case_cfg: dict[str, Any], reason: str) -> dict[str, Any]:
    artifact = case_cfg["artifact"]
    return {
        "Name": case_cfg["name"],
        "Family": case_cfg["family"],
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
        "Model Path": str(Path(artifact["model_path"]).resolve()) if Path(artifact["model_path"]).exists() else str(artifact["model_path"]),
        "Tokenizer Path": str(Path(artifact["tokenizer_path"]).resolve()) if artifact.get("tokenizer_path") and Path(artifact["tokenizer_path"]).exists() else str(artifact.get("tokenizer_path") or ""),
        "Prepared Model Path": str(case_cfg.get("prepared_model_path", "")),
        "Model Override Path": str(case_cfg.get("model_override_path", "")),
    }


def _run_case_rows(cases: list[dict[str, Any]], *, args: argparse.Namespace, results_dir: Path, gpu_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_cfg in cases:
        ok, reason = prepare_case(case_cfg, args=args, gpu_id=gpu_id)
        if not ok:
            EXP7.print_error(f"[{case_cfg['name']}] 预检查失败：{reason}")
            rows.append(_blank_row(case_cfg, reason))
            continue

        row = EXP7.run_single_case(case_cfg, args=args, results_dir=results_dir, gpu_id=gpu_id)
        row["Family"] = case_cfg["family"]
        row["Prepared Model Path"] = str(case_cfg.get("prepared_model_path", ""))
        row["Model Override Path"] = str(case_cfg.get("model_override_path", ""))
        rows.append(row)
    return rows


def _finalize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    preferred = ["Name", "Family", "Stack Label"]
    columns = [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in set(preferred)]
    return df[columns]


def load_existing_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(summary_path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"已有 summary 的 rows 字段格式错误: {summary_path}")
    cleaned: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict) and item.get("Name"):
            cleaned.append(dict(item))
    return cleaned


def merge_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        name = str(row.get("Name", "")).strip()
        if name:
            merged[name] = dict(row)
    for row in new_rows:
        name = str(row.get("Name", "")).strip()
        if name:
            merged[name] = dict(row)

    ordered: list[dict[str, Any]] = []
    for case in CASE_CONFIGS:
        name = str(case["name"])
        if name in merged:
            ordered.append(merged.pop(name))
    for _, row in sorted(merged.items(), key=lambda item: item[0]):
        ordered.append(row)
    return ordered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_names = parse_case_names(args.case_names)

    if args.skip_vllm_compat_check:
        os.environ["LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK"] = "1"

    EXP7.RESULTS_DIR = args.results_dir
    EXP7.LOGS_DIR = LOGS_DIR
    EXP7.TEMP_DIR = TEMP_DIR

    EXP7.ensure_dir(args.results_dir)
    EXP7.ensure_dir(LOGS_DIR)
    EXP7.ensure_dir(TEMP_DIR)
    EXP7.ensure_dir(PREQUANTIZED_MODEL_DIR)

    gpu_id = EXP7.infer_gpu_id(args.gpu_id)
    EXP7.print_info(
        "Exp5 七组对比实验开始："
        f" GPU={gpu_id}, benchmark_samples={args.benchmark_num_samples}, accuracy_samples={args.accuracy_num_samples}"
    )

    case_matrix = filter_cases(build_case_matrix(), selected_names)
    if not case_matrix:
        raise ValueError("本轮没有可执行的 case。")

    rows = _run_case_rows(case_matrix, args=args, results_dir=args.results_dir, gpu_id=gpu_id)
    if args.reuse_existing_summary is not None:
        existing_rows = load_existing_rows(args.reuse_existing_summary)
        rows = merge_rows(existing_rows, rows)

    df = _finalize_rows(rows)

    csv_path = args.results_dir / "exp5_engine_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    draw_figures(df, output_dir=args.results_dir)

    markdown_path = args.results_dir / "exp5_engine_report.md"
    write_markdown_report(df, output_path=markdown_path)

    success_df = df[df["Overall Status"] == "success"].copy()
    summary_payload = {
        "experiment": "exp5_engine_quantized_compare",
        "comparison_scope": "Transformers 16/8/4bit vs vLLM 16/8/4bit vs vLLM AWQ",
        "benchmark_num_samples": int(args.benchmark_num_samples),
        "accuracy_num_samples": int(args.accuracy_num_samples),
        "gpu_id": gpu_id,
        "skip_vllm_compat_check": bool(args.skip_vllm_compat_check),
        "selected_case_names": selected_names,
        "reuse_existing_summary": (
            str(args.reuse_existing_summary.resolve()) if args.reuse_existing_summary is not None else ""
        ),
        "fairness_notes": [
            "前六组都基于同一份 qwen2.5-3b-genesis-merged 模型资产，只改变后端和运行时量化方式。",
            "vLLM 8bit/4bit 改为读取真正的 bitsandbytes 预量化目录，而不是从 FP16 safetensors 运行时现量化。",
            "vLLM_AWQ 使用的是预压缩目录，因此它代表的是另一条部署路径，不应与运行时 BNB 量化简单等同。",
            "benchmark 与 accuracy 分开执行；任一 case 出现 OOM 或初始化失败时，会记录状态并继续下一个 case。",
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
        "lowest_benchmark_vram_case": (
            success_df.sort_values("Benchmark Peak VRAM (MB)", ascending=True).iloc[0]["Name"]
            if not success_df.empty
            else None
        ),
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = args.results_dir / "exp5_engine_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    record_run_meta(
        args.results_dir,
        cli_args=vars(args),
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        data_paths=[args.benchmark_prompts_file, args.test_file, args.dataset_file],
        extra_meta={
            "entry": "experiments/09_exp5_engine/run_exp5_engine_benchmark.py",
            "stage": "exp5_engine_quantized_compare",
            "comparison_scope": summary_payload["comparison_scope"],
            "result_csv": str(csv_path.resolve()),
            "result_markdown": str(markdown_path.resolve()),
            "result_summary": str(summary_path.resolve()),
            "cases": [case["name"] for case in CASE_CONFIGS],
        },
    )

    EXP7.print_info(f"Exp5 对比实验完成，CSV 输出：{csv_path}")
    EXP7.print_info(f"Markdown 报告：{markdown_path}")
    EXP7.print_info(f"Summary JSON：{summary_path}")
    EXP7.print_info(f"图表目录：{args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
