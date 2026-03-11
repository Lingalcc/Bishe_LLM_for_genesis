#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_command(cmd: dict[str, Any]) -> None:
    if not isinstance(cmd, dict):
        raise ValueError("command must be a dict")
    action = cmd.get("action")
    if not isinstance(action, str):
        raise ValueError("command.action must be a string")

    if action in {"step", "wait"}:
        steps = cmd.get("steps")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(f"{action}.steps must be int >= 1")
        return

    if action in {"reset_scene", "get_state"}:
        return

    if action in {"open_gripper", "close_gripper"}:
        if "position" in cmd and not is_num(cmd["position"]):
            raise ValueError(f"{action}.position must be number")
        if "steps" in cmd and (not isinstance(cmd["steps"], int) or cmd["steps"] < 1):
            raise ValueError(f"{action}.steps must be int >= 1")
        return

    if action == "move_ee":
        pos = cmd.get("pos")
        if not isinstance(pos, list) or len(pos) != 3 or not all(is_num(v) for v in pos):
            raise ValueError("move_ee.pos must be list[3] numbers")
        if "quat" in cmd:
            quat = cmd["quat"]
            if not isinstance(quat, list) or len(quat) != 4 or not all(is_num(v) for v in quat):
                raise ValueError("move_ee.quat must be list[4] numbers")
        if "steps" in cmd and (not isinstance(cmd["steps"], int) or cmd["steps"] < 1):
            raise ValueError("move_ee.steps must be int >= 1")
        return

    if action == "set_qpos":
        qpos = cmd.get("qpos")
        if not isinstance(qpos, list) or len(qpos) < 1 or not all(is_num(v) for v in qpos):
            raise ValueError("set_qpos.qpos must be non-empty number list")
        if "steps" in cmd and (not isinstance(cmd["steps"], int) or cmd["steps"] < 1):
            raise ValueError("set_qpos.steps must be int >= 1")
        return

    if action in {"set_dofs_position", "control_dofs_position", "control_dofs_velocity", "control_dofs_force"}:
        values = cmd.get("values")
        if not isinstance(values, list) or len(values) < 1 or not all(is_num(v) for v in values):
            raise ValueError(f"{action}.values must be non-empty number list")
        dofs = cmd.get("dofs_idx_local")
        if dofs is not None:
            if not isinstance(dofs, list) or len(dofs) < 1 or not all(isinstance(v, int) for v in dofs):
                raise ValueError(f"{action}.dofs_idx_local must be int list")
            if len(dofs) != len(values):
                raise ValueError(f"{action}.dofs_idx_local length must match values")
        if "steps" in cmd and (not isinstance(cmd["steps"], int) or cmd["steps"] < 1):
            raise ValueError(f"{action}.steps must be int >= 1")
        return

    raise ValueError(f"unsupported action: {action}")


def validate_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        commands = payload
    elif isinstance(payload, dict) and "commands" in payload:
        commands = payload["commands"]
    elif isinstance(payload, dict):
        commands = [payload]
    else:
        raise ValueError("payload must be object/list")

    if not isinstance(commands, list) or len(commands) == 0:
        raise ValueError("commands must be non-empty list")
    for cmd in commands:
        validate_command(cmd)
    return commands
