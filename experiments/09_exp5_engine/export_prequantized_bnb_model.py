#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 HF safetensors 模型导出为 bitsandbytes 预量化模型目录。")
    parser.add_argument("--source-model-dir", type=Path, required=True, help="原始 merged 模型目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="导出的预量化模型目录。")
    parser.add_argument("--mode", choices=["8bit", "4bit"], required=True, help="导出模式。")
    parser.add_argument("--max-shard-size", type=str, default="5GB", help="save_pretrained 的分片大小。")
    parser.add_argument("--force", action="store_true", help="如果输出目录已存在，则先删除后重导。")
    return parser.parse_args(argv)


def print_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def print_error(message: str) -> None:
    print(f"[ERROR] {message}", flush=True)


def ensure_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在：{path}；如需重导请增加 --force")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_quantization_config(mode: str):
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            llm_int8_threshold=6.0,
        )
    return BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )


def validate_export(output_dir: Path, *, mode: str) -> None:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"导出后缺少配置文件：{config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    quant_cfg = payload.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        raise RuntimeError("导出后的 config.json 缺少 quantization_config。")
    if str(quant_cfg.get("quant_method") or "").strip().lower() != "bitsandbytes":
        raise RuntimeError("导出后的 quantization_config.quant_method 不是 bitsandbytes。")
    if mode == "8bit" and not bool(quant_cfg.get("load_in_8bit")):
        raise RuntimeError("导出后的 config.json 未包含 load_in_8bit=true。")
    if mode == "4bit" and not bool(quant_cfg.get("load_in_4bit")):
        raise RuntimeError("导出后的 config.json 未包含 load_in_4bit=true。")

    has_weights = any(output_dir.glob("*.safetensors")) or any(output_dir.glob("*.bin")) or any(output_dir.glob("*.pt"))
    if not has_weights:
        raise RuntimeError(f"导出目录中未找到模型权重文件：{output_dir}")


def write_manifest(output_dir: Path, *, source_model_dir: Path, mode: str) -> None:
    payload = {
        "source_model_dir": str(source_model_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    }
    (output_dir / "llm_genesis_bnb_export.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.source_model_dir.exists():
        print_error(f"原始模型目录不存在：{args.source_model_dir}")
        return 1

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        print_error(f"导入依赖失败：{exc}")
        return 1

    if not torch.cuda.is_available():
        print_error("当前环境没有可用 GPU，bitsandbytes 预量化导出需要 CUDA GPU。")
        return 1

    quantization_config = build_quantization_config(args.mode)
    temp_output_dir = args.output_dir.parent / f".{args.output_dir.name}.tmp"

    try:
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
        ensure_empty_dir(temp_output_dir, force=True)

        print_info(f"开始加载原始模型：{args.source_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.source_model_dir.resolve()),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.source_model_dir.resolve()),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        print_info(f"开始保存预量化模型：{temp_output_dir}")
        model.save_pretrained(
            str(temp_output_dir.resolve()),
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(str(temp_output_dir.resolve()))
        validate_export(temp_output_dir, mode=args.mode)
        write_manifest(temp_output_dir, source_model_dir=args.source_model_dir, mode=args.mode)

        if args.output_dir.exists():
            if not args.force:
                raise FileExistsError(f"输出目录已存在：{args.output_dir}；如需覆盖请增加 --force")
            shutil.rmtree(args.output_dir)
        temp_output_dir.replace(args.output_dir)
        print_info(f"导出完成：{args.output_dir}")
        return 0
    except Exception as exc:
        print_error(f"导出失败：{exc}")
        return 1
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
