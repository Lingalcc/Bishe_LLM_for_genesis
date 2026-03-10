#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.app.unified_config import get_section


DEFAULT_APP_SYSTEM_PROMPT = (
    "你是 Franka 机械臂控制指令生成器。"
    "请把用户自然语言转换为可执行的 JSON action。"
    "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
    "只输出 JSON，不要输出解释。"
)

DEFAULT_APP_EXAMPLE_JSON = """{
  "commands": [
    {"action": "open_gripper", "position": 0.04},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.18], "quat": [0, 1, 0, 0]},
    {"action": "close_gripper", "position": 0.0},
    {"action": "wait", "steps": 20},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.30], "quat": [0, 1, 0, 0]}
  ]
}"""


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


def normalize_action_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "commands" in payload:
        commands = payload["commands"]
    elif isinstance(payload, list):
        commands = payload
    elif isinstance(payload, dict) and "action" in payload:
        commands = [payload]
    else:
        raise ValueError("payload must be command object, command list, or {'commands':[...]} format")

    if not isinstance(commands, list) or not commands:
        raise ValueError("commands must be non-empty list")
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            raise TypeError(f"command at index {i} must be object")
        if "action" not in cmd:
            raise ValueError(f"command at index {i} missing 'action'")
    return {"commands": commands}


def collect_scene_state(manager: Any) -> dict[str, Any]:
    entities = manager.get_entities()
    snapshots: list[dict[str, Any]] = []
    for name in sorted(entities.keys()):
        try:
            params = manager.get_entity_params(name, include_state=False)
        except Exception:
            params = {"name": name}
        try:
            runtime_state = manager.get_entity_state(name)
        except Exception:
            runtime_state = {}
        snapshots.append(
            {
                "name": name,
                "category": params.get("category", ""),
                "entity_class": params.get("entity_class", ""),
                "state": runtime_state,
            }
        )
    return {"entities": snapshots}


def build_state_context_text(scene_state: dict[str, Any]) -> str:
    state_text = json.dumps(scene_state, ensure_ascii=False, separators=(",", ":"))
    return f"[STATE_CONTEXT]{state_text}[/STATE_CONTEXT]"


def inject_state_into_instruction(instruction: str, scene_state: dict[str, Any] | None) -> str:
    if not scene_state:
        return instruction
    return f"{build_state_context_text(scene_state)}\n用户指令: {instruction}"


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


def predict_actions_from_instruction(
    instruction: str,
    cfg: dict[str, Any],
    *,
    scene_state: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    model_cfg = get_section(cfg, "app", "model")
    api_base = str(model_cfg.get("api_base", "https://api.openai.com/v1"))
    model = str(model_cfg.get("model", "gpt-5"))
    api_key = str(model_cfg.get("api_key", "")).strip()
    temperature = float(model_cfg.get("temperature", 0.0))
    max_tokens = int(model_cfg.get("max_tokens", 1200))
    timeout = int(model_cfg.get("timeout", 120))
    max_retries = int(model_cfg.get("max_retries", 3))
    sleep_seconds = float(model_cfg.get("sleep_seconds", 0.0))
    system_prompt = str(model_cfg.get("system_prompt", DEFAULT_APP_SYSTEM_PROMPT))

    if not api_key:
        raise RuntimeError("app.model.api_key is empty. Please set it in configs/default.yaml")

    prompt_text = inject_state_into_instruction(instruction, scene_state)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            raw = call_chat_completions(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            payload = normalize_action_payload(extract_first_json_from_text(raw))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return raw, payload
        except (ValueError, json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError) as err:
            last_err = err
            time.sleep(min(8.0, 0.8 * (2**i)))

    raise RuntimeError(f"model prediction failed: {last_err}")


def build_interactive_env(show_viewer: bool = True):
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    local_genesis_path = repo_root / "Genesis"
    if str(local_genesis_path) not in sys.path:
        sys.path.insert(0, str(local_genesis_path))

    import genesis as gs
    from genesis_example.genesis_tools import GenesisManager, GenesisRobot

    manager = GenesisManager(
        backend="gpu",
        init_kwargs={"precision": "32", "logging_level": "warning"},
        scene_kwargs={
            "show_viewer": show_viewer,
            "sim_options": gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0)),
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(2.5, -1.6, 1.4),
                camera_lookat=(0.5, 0.0, 0.3),
                camera_fov=35,
                res=(1280, 720),
                max_FPS=60,
            ),
        },
        auto_init=True,
        auto_create_scene=True,
        force_reinit=True,
    )

    manager.add_plane("ground")
    manager.add_robot_from_file(
        name="franka",
        file="xml/franka_emika_panda/panda.xml",
        robot_type="mjcf",
    )
    manager.add_object(
        name="cube",
        morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
    )
    manager.build_scene()
    manager.step(1)

    robot = GenesisRobot(
        manager=manager,
        robot_name="franka",
        ee_link_name="hand",
        arm_dofs_idx_local=list(range(7)),
        gripper_dofs_idx_local=[7, 8],
        default_interp_steps=120,
        default_command_steps=120,
        default_init_qpos=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
        default_init_steps=150,
    )
    robot.move_to_default_pose()
    return manager, robot


def print_state(robot: Any) -> None:
    state = robot.manager.get_entity_state(robot.robot_name)
    keys = list(state.keys())
    print(f"[state] keys: {keys}")
    if "qpos" in state:
        qpos = state["qpos"]
        if isinstance(qpos, list) and qpos:
            if isinstance(qpos[0], list):
                print(f"[state] qpos[0][:7] = {qpos[0][:7]}")
            else:
                print(f"[state] qpos[:7] = {qpos[:7]}")
