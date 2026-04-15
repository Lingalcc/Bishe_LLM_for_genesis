from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class ActionSchema:
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationPolicy:
    name: str = "default"
    allow_steps: bool = False
    allow_wait_steps: bool = False


DEFAULT_POLICY = ValidationPolicy()

POLICIES: dict[str, ValidationPolicy] = {
    "default": DEFAULT_POLICY,
    # Alias kept for clearer callsites in execution path.
    "execution": DEFAULT_POLICY,
    # Alias kept for clearer callsites in evaluation path.
    "evaluation": DEFAULT_POLICY,
}


ACTION_SCHEMAS: dict[str, ActionSchema] = {
    "wait": ActionSchema(),
    "step": ActionSchema(),
    "reset_scene": ActionSchema(),
    "get_state": ActionSchema(),
    "open_gripper": ActionSchema(optional_fields=("position",)),
    "close_gripper": ActionSchema(optional_fields=("position",)),
    "move_ee": ActionSchema(required_fields=("pos", "quat")),
    "set_qpos": ActionSchema(required_fields=("qpos",)),
    "set_dofs_position": ActionSchema(required_fields=("values",), optional_fields=("dofs_idx_local",)),
    "control_dofs_position": ActionSchema(required_fields=("values",), optional_fields=("dofs_idx_local",)),
    "control_dofs_velocity": ActionSchema(required_fields=("values",), optional_fields=("dofs_idx_local",)),
    "control_dofs_force": ActionSchema(required_fields=("values",), optional_fields=("dofs_idx_local",)),
}


def extract_first_json(text: str) -> Any:
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

    partial = _recover_partial_commands_payload(payload)
    if partial is not None:
        return partial

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


def _recover_partial_commands_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r'"commands"\s*:\s*\[', text)
    if match is None:
        return None

    decoder = json.JSONDecoder()
    idx = match.end()
    recovered: list[dict[str, Any]] = []

    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(text) or text[idx] == "]":
            break
        if text[idx] != "{":
            break
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            break
        if not isinstance(obj, dict):
            break
        action = obj.get("action")
        if not isinstance(action, str) or not action.strip():
            break
        recovered.append(obj)
        idx += end

    if not recovered:
        return None
    return {"commands": recovered}


def normalize_payload(payload: Any) -> dict[str, Any]:
    raw = payload
    if isinstance(payload, str):
        raw = extract_first_json(payload)

    if isinstance(raw, dict) and "commands" in raw:
        commands = raw["commands"]
    elif isinstance(raw, list):
        commands = raw
    elif isinstance(raw, dict) and "action" in raw:
        commands = [raw]
    else:
        raise ValueError("payload must be command object, command list, or {'commands':[...]} format")

    if not isinstance(commands, list) or len(commands) == 0:
        raise ValueError("commands must be non-empty list")

    normalized: list[dict[str, Any]] = []
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            raise TypeError(f"command index={i} must be object")
        action = cmd.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"command index={i} missing valid 'action'")
        normalized_cmd = dict(cmd)
        normalized_cmd["action"] = action.strip()
        normalized.append(normalized_cmd)

    return {"commands": normalized}


def _ensure_number_list(value: Any, key: str, *, exact_len: int | None = None, min_len: int = 1) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be number list")
    if exact_len is not None and len(value) != exact_len:
        raise ValueError(f"{key} must be list[{exact_len}] numbers")
    if exact_len is None and len(value) < min_len:
        raise ValueError(f"{key} must be non-empty number list")
    if not all(is_num(v) for v in value):
        raise ValueError(f"{key} must be number list")


def _ensure_int_list(value: Any, key: str) -> None:
    if not isinstance(value, list) or not value or not all(isinstance(v, int) for v in value):
        raise ValueError(f"{key} must be int list")


def _resolve_policy(policy: ValidationPolicy | str | None) -> ValidationPolicy:
    if policy is None:
        return DEFAULT_POLICY
    if isinstance(policy, ValidationPolicy):
        return policy
    if isinstance(policy, str) and policy in POLICIES:
        return POLICIES[policy]
    raise ValueError(f"unknown validation policy: {policy}")


def _validate_command(cmd: dict[str, Any], *, policy: ValidationPolicy) -> None:
    action = cmd["action"]
    if action not in ACTION_SCHEMAS:
        raise ValueError(f"unsupported action: {action}")

    schema = ACTION_SCHEMAS[action]
    for field in schema.required_fields:
        if field not in cmd:
            raise ValueError(f"{action}.{field} is required")

    steps = cmd.get("steps")
    if steps is not None:
        if action in {"wait", "step"} and not policy.allow_wait_steps:
            raise ValueError(f"{action}.steps is not supported")
        if not policy.allow_steps:
            raise ValueError("`steps` is not supported")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be int >= 1")

    if action == "move_ee":
        _ensure_number_list(cmd.get("pos"), "move_ee.pos", exact_len=3)
        _ensure_number_list(cmd.get("quat"), "move_ee.quat", exact_len=4)
        return

    if action in {"open_gripper", "close_gripper"}:
        if "position" in cmd and not is_num(cmd["position"]):
            raise ValueError(f"{action}.position must be number")
        return

    if action == "set_qpos":
        _ensure_number_list(cmd.get("qpos"), "set_qpos.qpos")
        return

    if action in {
        "set_dofs_position",
        "control_dofs_position",
        "control_dofs_velocity",
        "control_dofs_force",
    }:
        values = cmd.get("values")
        _ensure_number_list(values, f"{action}.values")
        dofs = cmd.get("dofs_idx_local")
        if dofs is not None:
            _ensure_int_list(dofs, f"{action}.dofs_idx_local")
            if len(dofs) != len(values):
                raise ValueError(f"{action}.dofs_idx_local length must match values")
        return


def validate_payload(
    payload: Any,
    policy: ValidationPolicy | str | None = None,
) -> list[dict[str, Any]]:
    active_policy = _resolve_policy(policy)
    normalized = normalize_payload(payload)
    commands = normalized["commands"]
    for cmd in commands:
        _validate_command(cmd, policy=active_policy)
    return commands
