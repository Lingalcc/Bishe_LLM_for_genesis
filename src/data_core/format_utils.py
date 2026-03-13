"""Dataset format conversion and sample validation utilities.

Supports two output formats used by LLaMA Factory:
  - Alpaca:   [{"instruction": ..., "input": "", "output": ...}]
  - ShareGPT: [{"conversations": [{"from": "system", ...}, ...]}]
"""
from __future__ import annotations

from typing import Any

from src.protocols.toolcall import extract_first_json, validate_payload

# System prompt embedded into ShareGPT training data
SYSTEM_PROMPT_FOR_DATASET = (
    "你是 Franka 机械臂控制指令生成器。"
    "请把用户自然语言转换为可执行的 JSON action。"
    "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，"
    "你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
    "只输出 JSON，不要输出解释。"
)


# ── Validation ────────────────────────────────────────────────────────────

def validate_sample(sample: dict[str, Any]) -> bool:
    """Return True if *sample* has valid instruction + parseable action JSON."""
    instruction = sample.get("instruction")
    output_str = sample.get("output")
    if not isinstance(instruction, str) or not instruction.strip():
        return False
    if not isinstance(output_str, str) or not output_str.strip():
        return False
    try:
        payload = extract_first_json(output_str)
        validate_payload(payload)
        return True
    except Exception:
        return False


# ── Alpaca format ─────────────────────────────────────────────────────────

def to_alpaca_format(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert raw samples to Alpaca format (LLaMA Factory compatible).

    Each row: {"instruction": ..., "input": "", "output": ...}
    """
    return [
        {
            "instruction": s["instruction"],
            "input": "",
            "output": s["output"],
        }
        for s in samples
    ]


# ── ShareGPT format ──────────────────────────────────────────────────────

def to_sharegpt_format(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw samples to ShareGPT multi-turn format (LLaMA Factory compatible).

    Each row: {"conversations": [system, human, gpt]}
    """
    return [
        {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT_FOR_DATASET},
                {"from": "human", "value": s["instruction"]},
                {"from": "gpt", "value": s["output"]},
            ]
        }
        for s in samples
    ]
