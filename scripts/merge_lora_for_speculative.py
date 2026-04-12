#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import activations as transformer_activations


REPO_ROOT = Path(__file__).resolve().parents[1]


def patch_awq_transformers_compat() -> None:
    # 某些环境中的 autoawq 仍依赖已从新版 transformers 中移除的旧名称。
    if not hasattr(transformer_activations, "PytorchGELUTanh") and hasattr(transformer_activations, "GELUTanh"):
        transformer_activations.PytorchGELUTanh = transformer_activations.GELUTanh


@dataclass(frozen=True)
class MergeTask:
    name: str
    base_model_path: Path
    adapter_path: Path
    output_path: Path


MERGE_TASKS: tuple[MergeTask, ...] = (
    MergeTask(
        name="主模型 3B",
        base_model_path=REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct",
        adapter_path=REPO_ROOT / "output" / "qwen2.5-3b-genesis-lora-rank-4",
        output_path=REPO_ROOT / "model" / "qwen2.5-3b-genesis-merged",
    ),
    MergeTask(
        name="草稿模型 0.5B",
        base_model_path=REPO_ROOT / "model" / "Qwen_Qwen2.5-0.5B-Instruct",
        adapter_path=REPO_ROOT / "output" / "qwen2.5-0.5b-genesis-lora-rank-4",
        output_path=REPO_ROOT / "model" / "qwen2.5-0.5b-genesis-merged",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 speculative decoding 合并并导出 LoRA 模型。")
    parser.add_argument(
        "--task",
        choices=("all", "target", "assistant"),
        default="all",
        help="选择要执行的合并任务，默认同时执行两组。",
    )
    parser.add_argument("--base-model-path", type=Path, default=None, help="自定义基础模型目录。")
    parser.add_argument("--adapter-path", type=Path, default=None, help="自定义 LoRA 适配器目录。")
    parser.add_argument("--output-path", type=Path, default=None, help="自定义 merged 模型输出目录。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出目录已存在则先删除再重建。",
    )
    return parser.parse_args()


def select_tasks(task_name: str) -> tuple[MergeTask, ...]:
    if task_name == "target":
        return (MERGE_TASKS[0],)
    if task_name == "assistant":
        return (MERGE_TASKS[1],)
    return MERGE_TASKS


def validate_task(task: MergeTask) -> None:
    if not task.base_model_path.exists():
        raise FileNotFoundError(f"基础模型目录不存在：{task.base_model_path}")
    if not task.adapter_path.exists():
        raise FileNotFoundError(f"LoRA 适配器目录不存在：{task.adapter_path}")


def build_custom_task(args: argparse.Namespace) -> MergeTask | None:
    custom_values = [args.base_model_path, args.adapter_path, args.output_path]
    if all(value is None for value in custom_values):
        return None
    if any(value is None for value in custom_values):
        raise ValueError("使用自定义合并模式时，必须同时提供 --base-model-path、--adapter-path、--output-path。")
    return MergeTask(
        name="自定义合并任务",
        base_model_path=args.base_model_path.resolve(),
        adapter_path=args.adapter_path.resolve(),
        output_path=args.output_path.resolve(),
    )


def prepare_output_path(output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请先删除或加 --overwrite：{output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def merge_lora(task: MergeTask, *, overwrite: bool) -> None:
    validate_task(task)
    prepare_output_path(task.output_path, overwrite=overwrite)
    patch_awq_transformers_compat()

    print(f"[INFO] 开始合并 {task.name}", flush=True)
    print(f"[INFO] Base Model    : {task.base_model_path}", flush=True)
    print(f"[INFO] LoRA Adapter  : {task.adapter_path}", flush=True)
    print(f"[INFO] Output Path   : {task.output_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(task.base_model_path, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        task.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, task.adapter_path, is_trainable=False)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(task.output_path)
    tokenizer.save_pretrained(task.output_path)

    print(f"[OK] 合并完成并已保存：{task.output_path}", flush=True)


def main() -> None:
    args = parse_args()
    custom_task = build_custom_task(args)
    tasks = (custom_task,) if custom_task is not None else select_tasks(args.task)
    for task in tasks:
        merge_lora(task, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
