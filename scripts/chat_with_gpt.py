#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.secrets import MissingSecretError, resolve_api_key_from_env


DEFAULT_BASE_URL = "https://api.yescode.cloud"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


def normalize_base_url(base_url: str) -> str:
    """将兼容服务地址规范化到 OpenAI SDK 常用的 /v1 根路径。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def normalize_reasoning_effort(reasoning_effort: str) -> str | None:
    """把本地配置中的推理强度映射到 Responses API 更常见的取值。"""
    effort = reasoning_effort.strip().lower()
    if not effort:
        return None
    mapping = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }
    return mapping.get(effort)


def extract_output_text(response: Any) -> str:
    """兼容不同网关实现，尽量提取文本输出。"""
    if not isinstance(response, dict):
        return ""

    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = response.get("output")
    if not output:
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content_list = item.get("content") or []
        for content in content_list:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                chunk = content.get("text", "")
                if isinstance(chunk, str) and chunk:
                    parts.append(chunk)
    return "".join(parts).strip()


def resolve_api_key(api_key_env: str) -> str:
    """基于环境变量获取 API Key。"""
    try:
        return resolve_api_key_from_env(
            api_key="",
            api_key_env=api_key_env,
            default_env=DEFAULT_API_KEY_ENV,
            source_name="GPT 对话脚本",
        )
    except MissingSecretError as exc:
        raise RuntimeError(str(exc)) from exc

def create_response(
    *,
    base_url: str,
    api_key: str,
    model: str,
    history: list[dict[str, str]],
    reasoning_effort: str | None,
) -> str:
    """调用 Responses API，并返回纯文本结果。"""
    request_args: dict[str, Any] = {
        "model": model,
        "input": history,
    }
    if reasoning_effort:
        request_args["reasoning"] = {"effort": reasoning_effort}

    data = json.dumps(request_args, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=f"{normalize_base_url(base_url)}/responses",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"网络请求失败：{exc.reason}") from exc

    response = json.loads(raw)
    text = extract_output_text(response)
    if not text:
        raise RuntimeError("模型已返回响应，但没有解析到可显示的文本内容。")
    return text


def interactive_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    reasoning_effort: str | None,
) -> None:
    """启动命令行多轮对话。"""
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    print(f"已连接模型：{model}")
    print("输入内容后回车发送，输入 /exit 退出，/clear 清空上下文。", flush=True)

    while True:
        try:
            user_text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。", flush=True)
            return

        if not user_text:
            continue
        if user_text == "/exit":
            print("已退出。", flush=True)
            return
        if user_text == "/clear":
            history = []
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})
            print("上下文已清空。", flush=True)
            continue

        history.append({"role": "user", "content": user_text})
        try:
            answer = create_response(
                base_url=base_url,
                api_key=api_key,
                model=model,
                history=history,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001
            history.pop()
            print(f"请求失败：{exc}", flush=True)
            continue

        history.append({"role": "assistant", "content": answer})
        print(f"\nGPT：{answer}", flush=True)


def single_turn_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    reasoning_effort: str | None,
) -> None:
    """执行单轮提问。"""
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": user_message})

    answer = create_response(
        base_url=base_url,
        api_key=api_key,
        model=model,
        history=history,
        reasoning_effort=reasoning_effort,
    )
    print(answer, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 OpenAI 兼容 Responses API 与 GPT 进行命令行对话。",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API 根地址，默认值：{DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名称，默认值：{DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help="推理强度，支持 minimal/low/medium/high；xhigh 会自动降级为 high。",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"保存 API Key 的环境变量名，默认值：{DEFAULT_API_KEY_ENV}",
    )
    parser.add_argument(
        "--system",
        default="你是一个有帮助的中文助手。",
        help="系统提示词。",
    )
    parser.add_argument(
        "--message",
        help="单轮提问内容；不传时进入交互模式。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reasoning_effort = normalize_reasoning_effort(args.reasoning_effort)
    if args.reasoning_effort and not reasoning_effort:
        print(
            "警告：当前 --reasoning-effort 不在兼容映射中，已忽略该参数。",
            file=sys.stderr,
            flush=True,
        )

    try:
        api_key = resolve_api_key(args.api_key_env)
        if args.message:
            single_turn_chat(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                system_prompt=args.system,
                user_message=args.message,
                reasoning_effort=reasoning_effort,
            )
            return 0

        interactive_chat(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=args.system,
            reasoning_effort=reasoning_effort,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"执行失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
