#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
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

# 仅统计和结构化生成最相关的投影层。
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
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokenized
    else:
        # 保守回退：若 tokenizer 没有 chat template，则直接拼接成 instruction-following 文本。
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

        # 每个样本独立清空梯度，保证最后得到的是“样本级 Taylor 分数”的平均值，
        # 而不是被前一个样本污染的累计梯度。
        model.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            print(f"[layer_scoring] 跳过异常样本 sample={sample_idx}, loss={loss}", flush=True)
            continue

        loss.backward()

        # 一阶泰勒展开近似中，参数重要性可写为 |w * grad|。
        # 直觉上，它衡量“如果把这个权重微小扰动为 0，损失会变化多少”。
        # 这里我们对每个目标线性层的整个权重矩阵求 mean(abs(w * grad))，
        # 再把同一 Transformer block 内的 attention / MLP 投影聚合为 layer score。
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
        if sample_idx % 10 == 0 or sample_idx == len(sampled_rows):
            print(
                f"[layer_scoring] progress: {sample_idx}/{len(sampled_rows)} "
                f"processed={processed_samples} loss={loss.detach().float().item():.6f}",
                flush=True,
            )

    if processed_samples == 0:
        raise RuntimeError("没有成功处理任何样本，无法生成层敏感度结果。")

    result_layers: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for layer_idx in sorted(layer_score_sums):
        projection_scores: dict[str, float] = {}
        for projection_name, score_sum in sorted(layer_projection_sums[layer_idx].items()):
            count = layer_projection_counts[layer_idx][projection_name]
            projection_scores[projection_name] = score_sum / max(1, count)

        # 聚合层分数时取各投影平均值，避免某一层因为统计到的子模块数量更多而被放大。
        layer_score = sum(projection_scores.values()) / max(1, len(projection_scores))
        layer_key = f"layer_{layer_idx}"
        result_layers[layer_key] = {
            "layer_index": layer_idx,
            "score": layer_score,
            "projection_scores": projection_scores,
        }
        ranking.append({"layer": layer_key, "score": layer_score})

    ranking.sort(key=lambda item: item["score"], reverse=True)

    score_values = [item["score"] for item in ranking]
    payload = {
        "meta": {
            "model_path": str(args.model_path),
            "data_path": str(args.data_path),
            "output_path": str(args.output_path),
            "sample_size_requested": args.sample_size,
            "sample_size_used": processed_samples,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device),
            "dtype": str(dtype),
        },
        "summary": {
            "num_layers_scored": len(result_layers),
            "max_score": max(score_values) if score_values else 0.0,
            "min_score": min(score_values) if score_values else 0.0,
            "mean_score": (sum(score_values) / len(score_values)) if score_values else 0.0,
            "top_5_layers": ranking[:5],
            "bottom_5_layers": ranking[-5:] if len(ranking) >= 5 else ranking,
        },
        "layers": result_layers,
        "ranking": ranking,
    }

    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"[layer_scoring] 已完成，输出 {len(result_layers)} 层分数，结果保存到: {args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[layer_scoring] 用户中断。", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"[layer_scoring] 失败: {exc}", file=sys.stderr)
        raise
