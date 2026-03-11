"""Shared API client utilities for data_core modules.

Provides a reusable OpenAI-compatible chat-completions caller and
a robust JSON extractor for handling messy LLM outputs.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


def resolve_api_key(api_key: str, api_key_env: str) -> str:
    """Resolve API key from explicit value or environment variable."""
    key = api_key.strip()
    if key:
        return key
    env_name = api_key_env.strip() or "OPENAI_API_KEY"
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"API key is empty. Set config api_key or env var `{env_name}`."
        )
    return key


def call_chat_api(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.9,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    """Call an OpenAI-compatible chat/completions endpoint and return the
    assistant content string."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
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
        raise ValueError("API returned empty choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API returned empty content")
    return content


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract the first JSON array from potentially messy LLM output.

    Handles: raw JSON, ```json code blocks, and leading prose before `[`.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Code-block extraction
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if code_match:
        try:
            result = json.loads(code_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Scan for array start
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("no valid JSON array found in API response")
