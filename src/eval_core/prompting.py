from __future__ import annotations

from typing import Any

DEFAULT_EVAL_SYSTEM_PROMPT = (
    "你是 Franka 机械臂控制指令生成器。"
    "请把用户自然语言转换为可执行的 JSON action。"
    "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，"
    "你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
    "只输出 JSON，不要输出解释。"
)


def _normalize_prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def resolve_system_prompt(
    *,
    cfg_system_prompt: str | None,
    sample_system_prompt: str | None,
    default_system_prompt: str = DEFAULT_EVAL_SYSTEM_PROMPT,
) -> str:
    """Resolve system prompt with priority: cfg > sample > default."""
    return (
        _normalize_prompt_text(cfg_system_prompt)
        or _normalize_prompt_text(sample_system_prompt)
        or _normalize_prompt_text(default_system_prompt)
    )


def build_eval_messages(
    *,
    instruction: str,
    cfg_system_prompt: str | None,
    sample_system_prompt: str | None,
    default_system_prompt: str = DEFAULT_EVAL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    system_prompt = resolve_system_prompt(
        cfg_system_prompt=cfg_system_prompt,
        sample_system_prompt=sample_system_prompt,
        default_system_prompt=default_system_prompt,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
