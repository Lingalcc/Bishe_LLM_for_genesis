#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .generate_genesis_franka_dataset import build_tools_json, validate_payload
except ImportError:
    from generate_genesis_franka_dataset import build_tools_json, validate_payload


AUG_SYSTEM_PROMPT = (
    "你是机器人控制数据增强专家。"
    "你的任务是：在保持目标 JSON 控制指令语义完全不变的前提下，"
    "生成高质量、多样化、可执行导向的中文用户指令。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment Franka NL->JSON SFT dataset using advanced model API."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data_prepare/genesis_franka_toolcall_alpaca.json"),
        help="Input alpaca dataset path.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data_prepare/genesis_franka_toolcall_alpaca_augmented.json"),
        help="Output augmented alpaca dataset path.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("data_prepare/genesis_franka_toolcall_augment_stats.json"),
        help="Augmentation stats path.",
    )
    parser.add_argument(
        "--output-sharegpt-file",
        type=Path,
        default=Path("data_prepare/genesis_franka_toolcall_sharegpt_augmented.json"),
        help="Output augmented sharegpt dataset path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-source",
        type=int,
        default=800,
        help="Number of source samples selected for augmentation.",
    )
    parser.add_argument(
        "--aug-per-sample",
        type=int,
        default=2,
        help="How many augmented instructions to generate per source sample.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-5"),
        help="Model name for augmentation.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name that stores API key.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key string. If set, it overrides --api-key-env.",
    )
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max completion tokens.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per API call.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between successful calls to avoid rate bursts.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def extract_first_json_from_text(text: str) -> Any:
    payload = text.strip()
    if not payload:
        raise ValueError("empty response")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", payload, flags=re.IGNORECASE)
    if code_match:
        inner = code_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(payload):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(payload[idx:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("no valid JSON found")


def parse_output_payload(text: str) -> Any:
    obj = extract_first_json_from_text(text)
    validate_payload(obj)
    return obj


def parse_augmented_instructions(text: str, expected_n: int) -> list[str]:
    obj = extract_first_json_from_text(text)

    if isinstance(obj, dict) and "instructions" in obj:
        arr = obj["instructions"]
    elif isinstance(obj, list):
        arr = obj
    else:
        raise ValueError("augmented response must be list or {'instructions': [...]} format")

    if not isinstance(arr, list):
        raise ValueError("instructions must be a list")
    cleaned = []
    for x in arr:
        if isinstance(x, str):
            t = x.strip()
            if t:
                cleaned.append(t)
    if len(cleaned) < expected_n:
        raise ValueError(f"instructions count < expected ({len(cleaned)} < {expected_n})")
    return cleaned[:expected_n]


def build_aug_user_prompt(
    source_instruction: str,
    payload: Any,
    n: int,
) -> str:
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "请基于下面的目标 JSON 指令，生成多条“不同表达但语义等价”的中文用户自然语言请求。\n\n"
        f"原始指令（仅供参考语气，不要照抄）：\n{source_instruction}\n\n"
        f"目标 JSON（语义必须完全一致）：\n{payload_text}\n\n"
        f"要求：\n"
        f"1. 生成 {n} 条中文指令，必须与目标 JSON 一一对应，不引入新动作或新参数。\n"
        "2. 指令风格要多样：口语化、正式、简洁、步骤化各占一部分。\n"
        "3. 至少 40% 的样本不出现“JSON”字样，模拟真实用户表达。\n"
        "4. 不要出现解释、答案或代码块，只返回 JSON。\n"
        "5. 输出格式严格为：{\"instructions\": [\"...\", \"...\"]}\n"
    )


def call_chat_completions(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    obj = json.loads(raw)
    choices = obj.get("choices", [])
    if not choices:
        raise ValueError("empty choices from API")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content from API")
    return content


def augment_once(
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    source_instruction: str,
    payload: Any,
    aug_per_sample: int,
) -> list[str]:
    prompt = build_aug_user_prompt(source_instruction, payload, aug_per_sample)
    messages = [
        {"role": "system", "content": AUG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            content = call_chat_completions(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return parse_augmented_instructions(content, aug_per_sample)
        except (ValueError, json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError) as e:
            last_err = e
            wait_s = min(8.0, 0.7 * (2**i))
            time.sleep(wait_s)

    if last_err is None:
        raise RuntimeError("augmentation failed without explicit error")
    raise RuntimeError(f"augmentation failed after retries: {last_err}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    api_key = args.api_key.strip()
    if not api_key:
        api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Provide --api-key or set env var: {args.api_key_env}"
        )

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    data = json.loads(args.input_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("input dataset must be a non-empty JSON list")

    valid_indices: list[int] = []
    payload_cache: dict[int, Any] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        ins = item.get("instruction")
        out = item.get("output")
        if not isinstance(ins, str) or not ins.strip():
            continue
        if not isinstance(out, str) or not out.strip():
            continue
        try:
            payload_cache[i] = parse_output_payload(out)
            valid_indices.append(i)
        except Exception:
            continue

    if not valid_indices:
        raise RuntimeError("No valid source samples found.")

    num_source = min(args.num_source, len(valid_indices))
    selected = rng.sample(valid_indices, k=num_source)

    dedup: set[tuple[str, str]] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        ins = item.get("instruction", "")
        out = item.get("output", "")
        if isinstance(ins, str) and isinstance(out, str):
            dedup.add((normalize_text(ins), normalize_text(out)))

    augmented_rows: list[dict[str, Any]] = []
    stats = {
        "input_count": len(data),
        "valid_source_count": len(valid_indices),
        "selected_source_count": num_source,
        "aug_per_sample": args.aug_per_sample,
        "api_model": args.model,
        "api_base": args.api_base,
        "api_success_calls": 0,
        "api_failed_calls": 0,
        "new_rows": 0,
        "dedup_skipped": 0,
        "skipped_source_errors": 0,
        "seed": args.seed,
    }

    for idx in selected:
        item = data[idx]
        source_instruction = item["instruction"]
        output_text = item["output"]
        input_text = item.get("input", "")
        system_text = item.get("system", "")

        payload = payload_cache[idx]
        try:
            new_instructions = augment_once(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                source_instruction=source_instruction,
                payload=payload,
                aug_per_sample=args.aug_per_sample,
            )
            stats["api_success_calls"] += 1
        except Exception:
            stats["api_failed_calls"] += 1
            stats["skipped_source_errors"] += 1
            continue

        for ins in new_instructions:
            key = (normalize_text(ins), normalize_text(output_text))
            if key in dedup:
                stats["dedup_skipped"] += 1
                continue
            dedup.add(key)
            augmented_rows.append(
                {
                    "instruction": ins,
                    "input": input_text if isinstance(input_text, str) else "",
                    "output": output_text,
                    "system": system_text if isinstance(system_text, str) else "",
                }
            )
        time.sleep(max(0.0, args.sleep_seconds))

    merged = data + augmented_rows

    tools_json = build_tools_json()
    merged_sharegpt: list[dict[str, Any]] = []
    for row in merged:
        if not isinstance(row, dict):
            continue
        ins = row.get("instruction", "")
        out = row.get("output", "")
        if not isinstance(ins, str) or not isinstance(out, str):
            continue
        merged_sharegpt.append(
            {
                "conversations": [
                    {"from": "human", "value": ins},
                    {"from": "gpt", "value": out},
                ],
                "system": row.get("system", "") if isinstance(row.get("system", ""), str) else "",
                "tools": tools_json,
            }
        )
    stats["new_rows"] = len(augmented_rows)
    stats["output_count"] = len(merged)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_sharegpt_file.parent.mkdir(parents=True, exist_ok=True)
    args.stats_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_sharegpt_file.write_text(json.dumps(merged_sharegpt, ensure_ascii=False, indent=2), encoding="utf-8")
    args.stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] input     : {args.input_file} ({len(data)} rows)")
    print(f"[ok] augmented : +{len(augmented_rows)} rows")
    print(f"[ok] output    : {args.output_file} ({len(merged)} rows)")
    print(f"[ok] sharegpt  : {args.output_sharegpt_file} ({len(merged_sharegpt)} rows)")
    print(f"[ok] stats     : {args.stats_file}")
    print("[stats]")
    for k in (
        "selected_source_count",
        "api_success_calls",
        "api_failed_calls",
        "dedup_skipped",
        "new_rows",
        "output_count",
    ):
        print(f"  - {k}: {stats[k]}")


if __name__ == "__main__":
    main()
