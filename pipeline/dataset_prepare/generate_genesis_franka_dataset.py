#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SYSTEM_PROMPT = (
    "你是 Franka 机械臂控制指令生成器。"
    "你的任务是把用户自然语言转换为可执行 JSON 指令。"
    "如果输入中包含 [STATE_CONTEXT]...[/STATE_CONTEXT]，你必须利用其中的场景状态进行决策。"
    "只输出 JSON，不要输出解释、注释或 Markdown。"
)

JSON_OUTPUT_HINTS = [
    "请只返回可执行 JSON，不要解释。",
    "只输出 JSON 指令。",
    "输出必须是 JSON，不能有多余文本。",
    "返回 JSON 格式命令即可。",
]


@dataclass
class Sample:
    instruction: str
    commands: list[dict[str, Any]]
    category: str


def r3(x: float) -> float:
    return round(float(x), 3)


def choose(rng: random.Random, items: list[Any]) -> Any:
    return items[rng.randrange(0, len(items))]


def sample_pos(rng: random.Random) -> list[float]:
    return [
        r3(rng.uniform(0.50, 0.75)),
        r3(rng.uniform(-0.22, 0.22)),
        r3(rng.uniform(0.12, 0.36)),
    ]


def sample_quat(rng: random.Random) -> list[float]:
    # 预定义一组稳定常用姿态
    quats = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.924, 0.383, 0.0],
        [0.0, 0.707, 0.707, 0.0],
        [0.271, 0.653, 0.653, 0.271],
    ]
    return choose(rng, quats)


def sample_home_like_qpos(rng: random.Random) -> list[float]:
    # Franka 9 维: 7 个手臂 + 2 个夹爪
    base = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
    q = []
    for i, val in enumerate(base):
        if i < 7:
            q.append(r3(val + rng.uniform(-0.25, 0.25)))
        else:
            q.append(r3(max(0.0, min(0.04, val + rng.uniform(-0.015, 0.0)))))
    return q


def maybe_add_steps(cmd: dict[str, Any], rng: random.Random, *, prob: float, lo: int, hi: int) -> None:
    if rng.random() < prob:
        cmd["steps"] = rng.randint(lo, hi)


def gen_open_gripper(rng: random.Random) -> Sample:
    pos = r3(rng.uniform(0.03, 0.04))
    cmd: dict[str, Any] = {"action": "open_gripper"}
    if rng.random() < 0.7:
        cmd["position"] = pos
    maybe_add_steps(cmd, rng, prob=0.5, lo=20, hi=80)

    templates = [
        f"把夹爪打开到大约 {pos} 米。",
        f"请张开机械爪，目标开度 {pos}。",
        "先把夹爪打开，保持可抓取状态。",
        "将末端夹爪打开。",
    ]
    return Sample(choose(rng, templates), [cmd], "open_gripper")


def gen_close_gripper(rng: random.Random) -> Sample:
    pos = r3(rng.uniform(0.0, 0.01))
    cmd: dict[str, Any] = {"action": "close_gripper"}
    if rng.random() < 0.7:
        cmd["position"] = pos
    maybe_add_steps(cmd, rng, prob=0.5, lo=20, hi=80)

    templates = [
        f"把夹爪闭合到 {pos} 米左右。",
        f"请闭合机械爪，目标开度 {pos}。",
        "将夹爪夹紧。",
        "关闭夹爪。",
    ]
    return Sample(choose(rng, templates), [cmd], "close_gripper")


def gen_wait(rng: random.Random) -> Sample:
    steps = rng.randint(8, 80)
    cmd = {"action": "wait", "steps": steps}
    templates = [
        f"保持当前状态等待 {steps} 步。",
        f"暂停一下，仿真推进 {steps} 个 step。",
        f"先别动，等待 {steps} 步。",
    ]
    return Sample(choose(rng, templates), [cmd], "wait")


def gen_step(rng: random.Random) -> Sample:
    steps = rng.randint(1, 30)
    cmd = {"action": "step", "steps": steps}
    templates = [
        f"推进仿真 {steps} 步。",
        f"执行 {steps} 个 step。",
        f"向前走 {steps} 个仿真步。",
    ]
    return Sample(choose(rng, templates), [cmd], "step")


def gen_get_state(rng: random.Random) -> Sample:
    cmd = {"action": "get_state"}
    templates = [
        "读取当前 franka 状态。",
        "给我机器人当前状态。",
        "查询机械臂最新状态。",
    ]
    return Sample(choose(rng, templates), [cmd], "get_state")


def gen_reset_scene(rng: random.Random) -> Sample:
    cmd = {"action": "reset_scene"}
    templates = [
        "重置整个场景。",
        "把仿真场景恢复到初始状态。",
        "执行场景重置。",
    ]
    return Sample(choose(rng, templates), [cmd], "reset_scene")


def gen_move_ee(rng: random.Random) -> Sample:
    pos = sample_pos(rng)
    cmd: dict[str, Any] = {"action": "move_ee", "pos": pos}
    use_quat = rng.random() < 0.8
    if use_quat:
        quat = sample_quat(rng)
        cmd["quat"] = quat
    maybe_add_steps(cmd, rng, prob=0.45, lo=40, hi=180)

    if use_quat:
        text = choose(
            rng,
            [
                f"把末端移动到 {pos}，四元数姿态设为 {cmd['quat']}。",
                f"请将手爪移动到坐标 {pos}，并使用 quat={cmd['quat']}。",
                f"移动 ee 到 {pos}，姿态 {cmd['quat']}。",
            ],
        )
    else:
        text = choose(
            rng,
            [
                f"把末端执行器移动到 {pos}。",
                f"移动机械臂 ee 到位置 {pos}。",
                f"将手爪平移到 {pos}。",
            ],
        )
    return Sample(text, [cmd], "move_ee")


def gen_set_qpos(rng: random.Random) -> Sample:
    qpos = sample_home_like_qpos(rng)
    cmd: dict[str, Any] = {"action": "set_qpos", "qpos": qpos}
    maybe_add_steps(cmd, rng, prob=0.6, lo=30, hi=150)

    templates = [
        f"将 Franka 的 9 维 qpos 设置为 {qpos}。",
        f"直接设定关节位置 qpos={qpos}。",
        f"把机器人关节状态切换到 {qpos}。",
    ]
    return Sample(choose(rng, templates), [cmd], "set_qpos")


def sample_dofs_and_values_for_position(rng: random.Random) -> tuple[list[int], list[float]]:
    dofs = sorted(rng.sample(list(range(9)), k=rng.randint(2, 5)))
    values: list[float] = []
    for idx in dofs:
        if idx < 7:
            values.append(r3(rng.uniform(-2.2, 2.2)))
        else:
            values.append(r3(rng.uniform(0.0, 0.04)))
    return dofs, values


def gen_set_dofs_position(rng: random.Random) -> Sample:
    dofs, values = sample_dofs_and_values_for_position(rng)
    cmd: dict[str, Any] = {"action": "set_dofs_position", "dofs_idx_local": dofs, "values": values}
    maybe_add_steps(cmd, rng, prob=0.5, lo=20, hi=120)

    templates = [
        f"把局部自由度 {dofs} 的位置直接设为 {values}。",
        f"请执行 set_dofs_position，dofs={dofs}，values={values}。",
        f"直接改写 dof 位置：索引 {dofs} -> {values}。",
    ]
    return Sample(choose(rng, templates), [cmd], "set_dofs_position")


def gen_control_dofs_position(rng: random.Random) -> Sample:
    dofs, values = sample_dofs_and_values_for_position(rng)
    cmd: dict[str, Any] = {"action": "control_dofs_position", "dofs_idx_local": dofs, "values": values}
    maybe_add_steps(cmd, rng, prob=0.6, lo=30, hi=180)

    templates = [
        f"用位置控制让 dof {dofs} 追踪到 {values}。",
        f"控制局部关节 {dofs} 到目标位置 {values}。",
        f"执行 position 控制：idx={dofs}, target={values}。",
    ]
    return Sample(choose(rng, templates), [cmd], "control_dofs_position")


def gen_control_dofs_velocity(rng: random.Random) -> Sample:
    dofs = sorted(rng.sample(list(range(7)), k=rng.randint(1, 3)))
    values = [r3(rng.uniform(-0.5, 0.5)) for _ in dofs]
    cmd: dict[str, Any] = {"action": "control_dofs_velocity", "dofs_idx_local": dofs, "values": values}
    maybe_add_steps(cmd, rng, prob=0.65, lo=20, hi=100)

    templates = [
        f"对关节 {dofs} 施加速度控制，目标速度 {values}。",
        f"执行速度控制：dofs={dofs}, values={values}。",
        f"把这些关节速度设为 {values}（索引 {dofs}）。",
    ]
    return Sample(choose(rng, templates), [cmd], "control_dofs_velocity")


def gen_control_dofs_force(rng: random.Random) -> Sample:
    dofs = sorted(rng.sample(list(range(7)), k=rng.randint(1, 3)))
    values = [r3(rng.uniform(-8.0, 8.0)) for _ in dofs]
    cmd: dict[str, Any] = {"action": "control_dofs_force", "dofs_idx_local": dofs, "values": values}
    maybe_add_steps(cmd, rng, prob=0.7, lo=10, hi=80)

    templates = [
        f"对关节 {dofs} 施加力控，目标力 {values}。",
        f"请发送 force 控制命令：idx={dofs}, force={values}。",
        f"设置 dof {dofs} 的控制力为 {values}。",
    ]
    return Sample(choose(rng, templates), [cmd], "control_dofs_force")


def gen_seq_grasp_lift(rng: random.Random) -> Sample:
    x = r3(rng.uniform(0.60, 0.70))
    y = r3(rng.uniform(-0.08, 0.08))
    z_top = r3(rng.uniform(0.20, 0.28))
    z_down = r3(max(0.10, z_top - rng.uniform(0.06, 0.1)))
    z_up = r3(z_top + rng.uniform(0.08, 0.14))

    cmds = [
        {"action": "open_gripper", "position": 0.04},
        {"action": "move_ee", "pos": [x, y, z_top], "quat": [0.0, 1.0, 0.0, 0.0]},
        {"action": "move_ee", "pos": [x, y, z_down], "quat": [0.0, 1.0, 0.0, 0.0]},
        {"action": "close_gripper", "position": 0.0},
        {"action": "wait", "steps": rng.randint(10, 30)},
        {"action": "move_ee", "pos": [x, y, z_up], "quat": [0.0, 1.0, 0.0, 0.0]},
    ]
    text = (
        f"执行抓取并抬升：先张开夹爪，移动到 [{x}, {y}, {z_top}]，"
        f"下探到 [{x}, {y}, {z_down}] 后闭合夹爪，再抬升到 [{x}, {y}, {z_up}]。"
    )
    return Sample(text, cmds, "seq_grasp_lift")


def gen_seq_place(rng: random.Random) -> Sample:
    x = r3(rng.uniform(0.55, 0.72))
    y = r3(rng.uniform(-0.16, 0.16))
    z_place = r3(rng.uniform(0.11, 0.2))
    z_retreat = r3(z_place + rng.uniform(0.08, 0.15))

    cmds = [
        {"action": "move_ee", "pos": [x, y, z_place], "quat": [0.0, 1.0, 0.0, 0.0]},
        {"action": "open_gripper", "position": 0.04},
        {"action": "wait", "steps": rng.randint(10, 25)},
        {"action": "move_ee", "pos": [x, y, z_retreat], "quat": [0.0, 1.0, 0.0, 0.0]},
    ]
    text = (
        f"执行放置动作：移动到 [{x}, {y}, {z_place}]，打开夹爪释放物体，"
        f"等待后撤离到 [{x}, {y}, {z_retreat}]。"
    )
    return Sample(text, cmds, "seq_place")


def gen_seq_reset_and_check(rng: random.Random) -> Sample:
    pos = sample_pos(rng)
    cmds = [
        {"action": "reset_scene"},
        {"action": "move_ee", "pos": pos, "quat": sample_quat(rng)},
        {"action": "wait", "steps": rng.randint(15, 40)},
        {"action": "get_state"},
    ]
    text = f"先重置场景，再把末端移动到 {pos}，稍等后返回当前状态。"
    return Sample(text, cmds, "seq_reset_check")


def gen_seq_joint_then_state(rng: random.Random) -> Sample:
    qpos = sample_home_like_qpos(rng)
    cmds = [
        {"action": "set_qpos", "qpos": qpos},
        {"action": "wait", "steps": rng.randint(20, 80)},
        {"action": "get_state"},
    ]
    text = f"把 qpos 设为 {qpos}，等待稳定后读取状态。"
    return Sample(text, cmds, "seq_joint_state")


def with_json_hint(text: str, rng: random.Random) -> str:
    return f"{text}\n{choose(rng, JSON_OUTPUT_HINTS)}"


def build_payload(commands: list[dict[str, Any]], rng: random.Random) -> Any:
    # 兼容多种输出格式：单命令对象 / {"commands": [...]}。
    if len(commands) == 1 and rng.random() < 0.35:
        return commands[0]
    return {"commands": commands}


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


def payload_to_text(payload: Any, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def sample_scene_state_for_commands(commands: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    cube_pos = [r3(rng.uniform(0.56, 0.72)), r3(rng.uniform(-0.16, 0.16)), 0.02]
    ee_pos = [r3(rng.uniform(0.50, 0.76)), r3(rng.uniform(-0.24, 0.24)), r3(rng.uniform(0.12, 0.36))]
    ee_quat = sample_quat(rng)
    franka_qpos = sample_home_like_qpos(rng)

    for cmd in commands:
        action = str(cmd.get("action", "")).strip().lower()
        if action == "move_ee" and isinstance(cmd.get("pos"), list) and len(cmd["pos"]) == 3:
            ee_pos = [r3(v) for v in cmd["pos"]]
            if isinstance(cmd.get("quat"), list) and len(cmd["quat"]) == 4:
                ee_quat = [r3(v) for v in cmd["quat"]]
        elif action == "set_qpos" and isinstance(cmd.get("qpos"), list) and len(cmd["qpos"]) >= 9:
            franka_qpos = [r3(v) for v in cmd["qpos"][:9]]
        elif action in {"seq_grasp_lift", "seq_place"}:
            pass

    return {
        "entities": [
            {
                "name": "franka",
                "category": "robot",
                "state": {
                    "qpos": franka_qpos,
                    "ee_pos": ee_pos,
                    "ee_quat": ee_quat,
                },
            },
            {
                "name": "cube",
                "category": "object",
                "state": {
                    "pos": cube_pos,
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "name": "ground",
                "category": "object",
                "state": {
                    "plane": True,
                },
            },
        ]
    }


def build_instruction_with_state_context(instruction: str, scene_state: dict[str, Any]) -> str:
    state_text = json.dumps(scene_state, ensure_ascii=False, separators=(",", ":"))
    return f"[STATE_CONTEXT]{state_text}[/STATE_CONTEXT]\n用户指令: {instruction}"


def build_tools_json() -> str:
    tools = [
        {
            "name": "execute_robot_json",
            "description": "执行 Franka 机械臂 JSON 指令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "array",
                        "description": "动作列表，每个元素至少包含 action 字段。",
                        "items": {"type": "object"},
                    }
                },
                "required": ["commands"],
            },
        }
    ]
    return json.dumps(tools, ensure_ascii=False)


def generate_dataset(
    num_samples: int,
    seed: int,
    *,
    state_context_ratio: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    generators: list[tuple[Callable[[random.Random], Sample], int]] = [
        (gen_move_ee, 22),
        (gen_open_gripper, 8),
        (gen_close_gripper, 8),
        (gen_wait, 6),
        (gen_step, 4),
        (gen_get_state, 5),
        (gen_reset_scene, 4),
        (gen_set_qpos, 9),
        (gen_set_dofs_position, 8),
        (gen_control_dofs_position, 10),
        (gen_control_dofs_velocity, 6),
        (gen_control_dofs_force, 6),
        (gen_seq_grasp_lift, 8),
        (gen_seq_place, 7),
        (gen_seq_reset_and_check, 5),
        (gen_seq_joint_then_state, 6),
    ]
    weighted_funcs: list[Callable[[random.Random], Sample]] = []
    for fn, w in generators:
        weighted_funcs.extend([fn] * w)

    dedup: set[tuple[str, str]] = set()
    alpaca: list[dict[str, Any]] = []
    sharegpt: list[dict[str, Any]] = []
    action_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    state_context_count = 0

    tools_json = build_tools_json()

    max_trials = num_samples * 30
    trials = 0
    while len(alpaca) < num_samples and trials < max_trials:
        trials += 1
        sample = choose(rng, weighted_funcs)(rng)
        instruction = with_json_hint(sample.instruction, rng)
        payload = build_payload(sample.commands, rng)
        commands = validate_payload(payload)
        if rng.random() < state_context_ratio:
            scene_state = sample_scene_state_for_commands(commands, rng)
            instruction = build_instruction_with_state_context(instruction, scene_state)
            state_context_count += 1
        response = payload_to_text(payload, rng)

        key = (normalize_text(instruction), normalize_text(response))
        if key in dedup:
            continue
        dedup.add(key)

        alpaca.append(
            {
                "instruction": instruction,
                "input": "",
                "output": response,
                "system": SYSTEM_PROMPT,
            }
        )
        sharegpt.append(
            {
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": response},
                ],
                "system": SYSTEM_PROMPT,
                "tools": tools_json,
            }
        )

        for cmd in commands:
            action_counter[cmd["action"]] += 1
        category_counter[sample.category] += 1

    if len(alpaca) < num_samples:
        raise RuntimeError(
            f"Only generated {len(alpaca)} samples (< {num_samples}). "
            "Increase max_trials or add templates."
        )

    stats = {
        "num_samples": len(alpaca),
        "seed": seed,
        "action_counter": dict(action_counter),
        "category_counter": dict(category_counter),
        "state_context_count": state_context_count,
        "state_context_ratio": (state_context_count / len(alpaca)) if alpaca else 0.0,
    }
    return alpaca, sharegpt, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate high-quality Franka natural-language -> JSON SFT datasets."
    )
    parser.add_argument("--num-samples", type=int, default=4000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--state-context-ratio",
        type=float,
        default=0.7,
        help="Ratio of samples that include [STATE_CONTEXT] state injection.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_prepare"),
        help="Output directory.",
    )
    parser.add_argument(
        "--alpaca-file",
        type=str,
        default="genesis_franka_toolcall_alpaca.json",
        help="Output alpaca-format filename.",
    )
    parser.add_argument(
        "--sharegpt-file",
        type=str,
        default="genesis_franka_toolcall_sharegpt.json",
        help="Output sharegpt-format filename.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="genesis_franka_toolcall_stats.json",
        help="Output stats filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= args.state_context_ratio <= 1.0:
        raise ValueError("--state-context-ratio must be within [0.0, 1.0]")

    alpaca, sharegpt, stats = generate_dataset(
        args.num_samples,
        args.seed,
        state_context_ratio=args.state_context_ratio,
    )

    alpaca_path = out_dir / args.alpaca_file
    sharegpt_path = out_dir / args.sharegpt_file
    stats_path = out_dir / args.stats_file

    alpaca_path.write_text(json.dumps(alpaca, ensure_ascii=False, indent=2), encoding="utf-8")
    sharegpt_path.write_text(json.dumps(sharegpt, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] alpaca  : {alpaca_path} ({len(alpaca)} samples)")
    print(f"[ok] sharegpt: {sharegpt_path} ({len(sharegpt)} samples)")
    print(f"[ok] stats   : {stats_path}")
    print("[stats] action coverage:")
    for action, cnt in sorted(stats["action_counter"].items(), key=lambda kv: kv[0]):
        print(f"  - {action}: {cnt}")


if __name__ == "__main__":
    main()
