#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPO_ROOT / "data_prepare" / "splits" / "train.json"
DEFAULT_OUTPUT_PATH = EXPERIMENT_DIR / "reports" / "layer_scores.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "Qwen_Qwen2.5-3B-Instruct"

PROJECTION_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
LAYER_PATTERN = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于一阶泰勒展开计算 Transformer 各层敏感度分数。"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="基础模型路径或 HuggingFace 模型名。",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="用于抽样打分的训练集 JSON 文件。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="层打分结果输出路径。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="参与敏感度估计的样本数。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="tokenizer 截断长度。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"数据集根节点必须是列表: {path}")
    return [row for row in payload if isinstance(row, dict)]


def sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0。")
    if len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(row.get("instruction", "")).strip()
    user_input = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()

    user_content = instruction if not user_input else f"{instruction}\n{user_input}"
    return [
        {
            "role": "system",
            "content": (
                "你是 Franka 机械臂控制指令生成器。"
                "请把用户自然语言转换为可执行的 JSON action。"
                "只输出 JSON，不要输出解释。"
            ),
        },
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def encode_supervised_example(
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    max_length: int,
) -> dict[str, torch.Tensor]:
    messages = build_messages(row)

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    else:
        merged_text = (
            f"<|system|>\n{messages[0]['content']}\n"
            f"<|user|>\n{messages[1]['content']}\n"
            f"<|assistant|>\n{messages[2]['content']}"
        )
        encoded = tokenizer(
            merged_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"]

    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_scored_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        match = LAYER_PATTERN.search(name)
        if match is None:
            continue
        projection_name = name.rsplit(".", 1)[-1]
        if projection_name not in PROJECTION_NAMES:
            continue
        yield int(match.group(1)), projection_name, name, module


def ensure_grad_enabled(model: torch.nn.Module) -> None:
    for _, _, _, module in iter_scored_modules(model):
        module.weight.requires_grad_(True)


def main() -> None:
    args = parse_args()
    device = resolve_device()
    dtype = choose_dtype()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[layer_scoring] model   : {args.model_path}", flush=True)
    print(f"[layer_scoring] data    : {args.data_path}", flush=True)
    print(f"[layer_scoring] output  : {args.output_path}", flush=True)
    print(f"[layer_scoring] device  : {device}", flush=True)
    print(f"[layer_scoring] dtype   : {dtype}", flush=True)

    rows = load_json_list(args.data_path)
    sampled_rows = sample_rows(rows, args.sample_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.train()
    ensure_grad_enabled(model)

    layer_score_sums: dict[int, float] = defaultdict(float)
    layer_projection_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    layer_projection_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    processed_samples = 0

    for sample_idx, row in enumerate(sampled_rows, start=1):
        batch = encode_supervised_example(
            tokenizer=tokenizer,
            row=row,
            max_length=args.max_length,
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            print(f"[layer_scoring] 跳过异常样本 sample={sample_idx}, loss={loss}", flush=True)
            continue

        loss.backward()

        for layer_idx, projection_name, _, module in iter_scored_modules(model):
            grad = module.weight.grad
            weight = module.weight.data
            if grad is None:
                continue

            score = torch.mean(torch.abs(weight * grad)).detach().float().item()
            layer_score_sums[layer_idx] += score
            layer_projection_sums[layer_idx][projection_name] += score
            layer_projection_counts[layer_idx][projection_name] += 1

        processed_samples += 1

    if processed_samples == 0:
        raise RuntimeError("没有可用于统计的有效样本，无法生成 layer_scores.json。")

    layers_payload: list[dict[str, Any]] = []
    for layer_idx in sorted(layer_score_sums.keys()):
        projection_scores = {}
        for projection_name in sorted(layer_projection_sums[layer_idx].keys()):
            count = layer_projection_counts[layer_idx][projection_name]
            projection_scores[projection_name] = (
                layer_projection_sums[layer_idx][projection_name] / max(1, count)
            )

        layers_payload.append(
            {
                "layer": layer_idx,
                "score": layer_score_sums[layer_idx] / processed_samples,
                "projection_scores": projection_scores,
            }
        )

    ranking = sorted(layers_payload, key=lambda item: float(item["score"]), reverse=True)
    summary = {
        "num_layers_scored": len(layers_payload),
        "processed_samples": processed_samples,
        "top5_layers": ranking[:5],
        "bottom5_layers": ranking[-5:],
    }
    payload = {
        "meta": {
            "model_path": str(args.model_path),
            "data_path": str(args.data_path),
            "sample_size": args.sample_size,
            "processed_samples": processed_samples,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device),
            "dtype": str(dtype),
        },
        "summary": summary,
        "layers": layers_payload,
        "ranking": ranking,
    }
    args.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[layer_scoring] 已写入: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
