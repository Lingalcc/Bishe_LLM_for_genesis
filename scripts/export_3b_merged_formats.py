#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.prompting import DEFAULT_EVAL_SYSTEM_PROMPT, build_eval_messages

DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged"
DEFAULT_CALIBRATION_DATASET_PATH = REPO_ROOT / "data_prepare" / "splits" / "train.json"
DEFAULT_LLAMA_CPP_DIR = REPO_ROOT / "third_party" / "PowerInfer" / "smallthinker"


def derive_default_gguf_output_path(model_path: Path) -> Path:
    return model_path.parent / f"{model_path.name}-q4_k_m.gguf"


def derive_default_awq_output_path(model_path: Path) -> Path:
    return model_path.parent / f"{model_path.name}-awq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 3B merged 模型导出为 GGUF(F16) 和 AWQ 量化格式。"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="待导出的 merged Hugging Face 模型目录。",
    )
    parser.add_argument(
        "--export",
        choices=("all", "gguf", "awq"),
        default="all",
        help="选择导出目标，默认同时导出 GGUF(F16) 与 AWQ。",
    )
    parser.add_argument(
        "--gguf-output-path",
        type=Path,
        default=None,
        help="GGUF 输出文件路径，默认放在 model 目录下并追加 -f16.gguf。",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=DEFAULT_LLAMA_CPP_DIR,
        help="包含 convert_hf_to_gguf.py 的 llama.cpp/PowerInfer 目录。",
    )
    parser.add_argument(
        "--gguf-outtype",
        choices=("f16", "q4_k_m"),
        default="q4_k_m",
        help="GGUF 输出精度，可选 f16 或 q4_k_m，默认 q4_k_m。",
    )
    parser.add_argument(
        "--keep-gguf-f16",
        action="store_true",
        help="当 GGUF 目标为量化格式时，是否保留中间产物 F16 GGUF。",
    )
    parser.add_argument(
        "--awq-output-path",
        type=Path,
        default=None,
        help="AWQ 量化模型输出目录，默认放在 model 目录下并追加 -awq。",
    )
    parser.add_argument(
        "--awq-calibration-dataset",
        type=Path,
        default=DEFAULT_CALIBRATION_DATASET_PATH,
        help="AWQ 校准数据集 JSON 路径，默认使用训练集。",
    )
    parser.add_argument(
        "--awq-calibration-samples",
        type=int,
        default=128,
        help="AWQ 校准样本数，默认 128。",
    )
    parser.add_argument(
        "--awq-max-calib-seq-len",
        type=int,
        default=512,
        help="AWQ 校准时的最大序列长度。",
    )
    parser.add_argument(
        "--awq-w-bit",
        type=int,
        default=4,
        help="AWQ 权重量化位宽，默认 4。",
    )
    parser.add_argument(
        "--awq-group-size",
        type=int,
        default=128,
        help="AWQ 分组大小，默认 128。",
    )
    parser.add_argument(
        "--awq-version",
        type=str,
        default="GEMM",
        help="AWQ 后端版本，常用为 GEMM 或 GEMV，默认 GEMM。",
    )
    parser.add_argument(
        "--awq-no-zero-point",
        action="store_true",
        help="关闭 AWQ zero_point，默认开启。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载模型与 tokenizer 时启用 trust_remote_code。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出已存在则覆盖。",
    )
    return parser.parse_args()


def validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"模型路径不是目录：{model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"模型目录缺少 config.json：{model_path}")


def prepare_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"输出路径已存在，请先删除或加 --overwrite：{path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def build_gguf_command(
    *,
    model_path: Path,
    gguf_output_path: Path,
    llama_cpp_dir: Path,
    gguf_outtype: str,
) -> list[str]:
    converter_path = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter_path.exists():
        raise FileNotFoundError(f"未找到 GGUF 转换脚本：{converter_path}")
    return [
        sys.executable,
        str(converter_path),
        str(model_path),
        "--outfile",
        str(gguf_output_path),
        "--outtype",
        gguf_outtype,
    ]


def build_gguf_quantize_command(
    *,
    llama_cpp_dir: Path,
    input_f16_path: Path,
    output_quantized_path: Path,
    gguf_outtype: str,
) -> list[str]:
    quantize_path = llama_cpp_dir / "tools" / "quantize"
    if not quantize_path.exists():
        raise FileNotFoundError(f"未找到 GGUF 量化工具：{quantize_path}")
    return [
        str(quantize_path),
        str(input_f16_path),
        str(output_quantized_path),
        gguf_outtype.upper(),
    ]


def run_gguf_export(
    *,
    model_path: Path,
    gguf_output_path: Path,
    llama_cpp_dir: Path,
    gguf_outtype: str,
    overwrite: bool,
    keep_gguf_f16: bool,
) -> None:
    prepare_output_path(gguf_output_path, overwrite=overwrite)
    if gguf_outtype == "f16":
        command = build_gguf_command(
            model_path=model_path,
            gguf_output_path=gguf_output_path,
            llama_cpp_dir=llama_cpp_dir,
            gguf_outtype=gguf_outtype,
        )
        print(f"[INFO] 开始导出 GGUF(F16)：{gguf_output_path}", flush=True)
        print(f"[INFO] 执行命令：{' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)
        print(f"[OK] GGUF 导出完成：{gguf_output_path}", flush=True)
        return

    intermediate_f16_path = gguf_output_path.with_name(f"{gguf_output_path.stem}.f16.gguf")
    prepare_output_path(intermediate_f16_path, overwrite=True)
    export_command = build_gguf_command(
        model_path=model_path,
        gguf_output_path=intermediate_f16_path,
        llama_cpp_dir=llama_cpp_dir,
        gguf_outtype="f16",
    )
    quantize_command = build_gguf_quantize_command(
        llama_cpp_dir=llama_cpp_dir,
        input_f16_path=intermediate_f16_path,
        output_quantized_path=gguf_output_path,
        gguf_outtype=gguf_outtype,
    )
    print(f"[INFO] 开始导出 GGUF(F16) 中间文件：{intermediate_f16_path}", flush=True)
    print(f"[INFO] 执行命令：{' '.join(export_command)}", flush=True)
    subprocess.run(export_command, check=True, cwd=REPO_ROOT)
    print(f"[INFO] 开始量化 GGUF({gguf_outtype.upper()})：{gguf_output_path}", flush=True)
    print(f"[INFO] 执行命令：{' '.join(quantize_command)}", flush=True)
    subprocess.run(quantize_command, check=True, cwd=REPO_ROOT)
    if keep_gguf_f16:
        print(f"[OK] GGUF 中间文件已保留：{intermediate_f16_path}", flush=True)
    else:
        intermediate_f16_path.unlink(missing_ok=True)
    print(f"[OK] GGUF 导出完成：{gguf_output_path}", flush=True)


def _record_to_calibration_text(record: Any) -> str:
    if isinstance(record, str):
        text = record.strip()
        return text
    if not isinstance(record, dict):
        return ""

    instruction = str(record.get("instruction", "") or "").strip()
    input_text = str(record.get("input", "") or "").strip()
    prompt_parts: list[str] = []
    if instruction:
        prompt_parts.append(f"用户指令：{instruction}")
    if input_text:
        prompt_parts.append(f"补充输入：{input_text}")
    return "\n".join(prompt_parts).strip()


def _render_calibration_chat_text(record: dict[str, Any], tokenizer: Any) -> str:
    instruction = str(record.get("instruction", "") or "").strip()
    input_text = str(record.get("input", "") or "").strip()
    if not instruction:
        return ""

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n[input]\n{input_text}"

    messages = build_eval_messages(
        instruction=user_content,
        cfg_system_prompt=DEFAULT_EVAL_SYSTEM_PROMPT,
        sample_system_prompt=str(record.get("system", "") or "").strip(),
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)).strip()
    return _record_to_calibration_text(record)


def load_calibration_texts(dataset_path: Path, sample_limit: int, *, tokenizer: Any | None = None) -> list[str]:
    if sample_limit <= 0:
        raise ValueError("awq_calibration_samples 必须大于 0。")
    if not dataset_path.exists():
        raise FileNotFoundError(f"校准数据集不存在：{dataset_path}")

    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"校准数据集必须是 JSON 数组：{dataset_path}")

    texts: list[str] = []
    for record in records:
        if isinstance(record, dict):
            text = _render_calibration_chat_text(record, tokenizer)
        else:
            text = _record_to_calibration_text(record)
        if text:
            texts.append(text)
        if len(texts) >= sample_limit:
            break

    if not texts:
        raise ValueError(f"未能从校准数据集中提取有效文本：{dataset_path}")
    return texts


def _import_awq_components() -> tuple[Any, Any]:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "未检测到 AWQ 依赖。请先安装 `autoawq` 或兼容的 `awq` 包后再执行 AWQ 导出。"
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("缺少 transformers 依赖，无法执行 AWQ 导出。") from exc

    return AutoAWQForCausalLM, AutoTokenizer


def run_awq_export(
    *,
    model_path: Path,
    awq_output_path: Path,
    calibration_dataset_path: Path,
    calibration_samples: int,
    max_calib_seq_len: int,
    w_bit: int,
    group_size: int,
    version: str,
    zero_point: bool,
    trust_remote_code: bool,
    overwrite: bool,
) -> None:
    prepare_output_path(awq_output_path, overwrite=overwrite)
    AutoAWQForCausalLM, AutoTokenizer = _import_awq_components()

    print(f"[INFO] 开始导出 AWQ：{awq_output_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    calibration_texts = load_calibration_texts(
        calibration_dataset_path,
        calibration_samples,
        tokenizer=tokenizer,
    )
    print(
        f"[INFO] 已加载 {len(calibration_texts)} 条校准文本，数据集：{calibration_dataset_path}",
        flush=True,
    )
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        safetensors=True,
        device_map="auto",
    )

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": version,
    }
    print(f"[INFO] AWQ 量化配置：{quant_config}", flush=True)

    quantize_kwargs = {
        "quant_config": quant_config,
        "calib_data": calibration_texts,
        "max_calib_seq_len": max_calib_seq_len,
    }
    try:
        model.quantize(tokenizer, **quantize_kwargs)
    except TypeError:
        print("[WARN] 当前 AWQ 版本不支持 max_calib_seq_len，回退到基础 quantize 调用。", flush=True)
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_texts,
        )

    model.save_quantized(str(awq_output_path))
    tokenizer.save_pretrained(str(awq_output_path))
    print(f"[OK] AWQ 导出完成：{awq_output_path}", flush=True)


def main() -> None:
    args = parse_args()
    validate_model_path(args.model_path)

    gguf_output_path = args.gguf_output_path or derive_default_gguf_output_path(args.model_path)
    awq_output_path = args.awq_output_path or derive_default_awq_output_path(args.model_path)

    if args.export in {"all", "gguf"}:
        run_gguf_export(
            model_path=args.model_path,
            gguf_output_path=gguf_output_path,
            llama_cpp_dir=args.llama_cpp_dir,
            gguf_outtype=args.gguf_outtype,
            overwrite=args.overwrite,
            keep_gguf_f16=args.keep_gguf_f16,
        )

    if args.export in {"all", "awq"}:
        run_awq_export(
            model_path=args.model_path,
            awq_output_path=awq_output_path,
            calibration_dataset_path=args.awq_calibration_dataset,
            calibration_samples=args.awq_calibration_samples,
            max_calib_seq_len=args.awq_max_calib_seq_len,
            w_bit=args.awq_w_bit,
            group_size=args.awq_group_size,
            version=args.awq_version,
            zero_point=not args.awq_no_zero_point,
            trust_remote_code=args.trust_remote_code,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
