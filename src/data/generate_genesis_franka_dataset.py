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

DEFAULT_ACTION_WEIGHTS: dict[str, int] = {
    "move_ee": 22,
    "open_gripper": 8,
    "close_gripper": 8,
    "wait": 6,
    "step": 4,
    "get_state": 5,
    "reset_scene": 4,
    "set_qpos": 9,
    "set_dofs_position": 8,
    "control_dofs_position": 10,
    "control_dofs_velocity": 6,
    "control_dofs_force": 6,
    "seq_grasp_lift": 8,
    "seq_place": 7,
    "seq_reset_check": 5,
    "seq_joint_state": 6,
}


@dataclass
class Sample:
    instruction: str
    commands: list[dict[str, Any]]
    category: str
    scene_state: dict[str, Any] | None = None
    force_state_context: bool = False


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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clamp_workspace_pos(pos: list[float]) -> list[float]:
    return [
        r3(clamp(pos[0], 0.45, 0.78)),
        r3(clamp(pos[1], -0.26, 0.26)),
        r3(clamp(pos[2], 0.08, 0.42)),
    ]


def sample_cube_pos(rng: random.Random) -> list[float]:
    return [
        r3(rng.uniform(0.56, 0.72)),
        r3(rng.uniform(-0.16, 0.16)),
        0.02,
    ]


def sample_cube_color(rng: random.Random) -> str:
    return choose(rng, ["红色", "蓝色", "绿色", "黄色", "橙色", "白色"])


def offset_pos_from_target(
    target_pos: list[float],
    rng: random.Random,
    *,
    xy_delta: float = 0.05,
    z_delta: float = 0.05,
) -> list[float]:
    """
    基于“目标位置”构造“动作发生前的当前位置”，避免状态穿越。
    """
    start = clamp_workspace_pos(
        [
            target_pos[0] + rng.uniform(-xy_delta, xy_delta),
            target_pos[1] + rng.uniform(-xy_delta, xy_delta),
            target_pos[2] + rng.uniform(-z_delta, z_delta),
        ]
    )
    # 避免恰好等于目标（状态穿越）
    if all(abs(start[i] - target_pos[i]) < 1e-6 for i in range(3)):
        start = clamp_workspace_pos([target_pos[0] + 0.02, target_pos[1] - 0.015, target_pos[2] + 0.02])
    return start


def offset_qpos_from_target(target_qpos: list[float], rng: random.Random) -> list[float]:
    """
    基于目标 qpos 采样一个动作前 qpos，确保不与目标完全一致。
    """
    start: list[float] = []
    for i, v in enumerate(target_qpos[:9]):
        if i < 7:
            nv = r3(clamp(v + rng.uniform(-0.20, 0.20), -2.75, 2.75))
        else:
            nv = r3(clamp(v + rng.uniform(-0.012, 0.012), 0.0, 0.04))
        start.append(nv)
    if all(abs(start[i] - target_qpos[i]) < 1e-6 for i in range(min(9, len(target_qpos)))):
        start[0] = r3(clamp(start[0] + 0.08, -2.75, 2.75))
    return start


def build_scene_state(
    *,
    cube_pos: list[float],
    cube_color: str,
    ee_pos: list[float],
    ee_quat: list[float],
    franka_qpos: list[float],
) -> dict[str, Any]:
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
                    "color": cube_color,
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
        "打开夹爪。",
        "夹爪先松开一点。",
        "把手爪张开，准备后续动作。",
        "先开爪，别夹住东西。",
        "打开 gripper 到可抓取状态。",
        "机械爪打开，准备执行下一步。",
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
        "把手爪收紧。",
        "夹爪闭合，执行抓取。",
        "现在合拢夹爪。",
        "收爪，保持抓取。",
        "把 gripper 关上。",
        "请把末端夹爪夹住目标。",
    ]
    return Sample(choose(rng, templates), [cmd], "close_gripper")


def gen_wait(rng: random.Random) -> Sample:
    steps = rng.randint(8, 80)
    cmd = {"action": "wait", "steps": steps}
    templates = [
        f"保持当前状态等待 {steps} 步。",
        f"暂停一下，仿真推进 {steps} 个 step。",
        f"先别动，等待 {steps} 步。",
        f"原地等待 {steps} 个仿真步。",
        f"保持姿态不变，持续 {steps} 步。",
        f"停一下，推进 {steps} 步再继续。",
        f"等待 {steps} steps。",
        f"不要执行新动作，等 {steps} 步。",
        f"hold 住，仿真走 {steps} 步。",
        f"静止 {steps} 步。",
    ]
    return Sample(choose(rng, templates), [cmd], "wait")


def gen_step(rng: random.Random) -> Sample:
    steps = rng.randint(1, 30)
    cmd = {"action": "step", "steps": steps}
    templates = [
        f"推进仿真 {steps} 步。",
        f"执行 {steps} 个 step。",
        f"向前走 {steps} 个仿真步。",
        f"仅推进模拟器 {steps} 步，不做其它动作。",
        f"把环境 step {steps} 次。",
        f"跑 {steps} 步仿真。",
        f"前进 {steps} step。",
        f"模拟器向前推进 {steps} 帧。",
        f"执行 step={steps}。",
        f"只做 step，次数 {steps}。",
    ]
    return Sample(choose(rng, templates), [cmd], "step")


def gen_get_state(rng: random.Random) -> Sample:
    cmd = {"action": "get_state"}
    templates = [
        "读取当前 franka 状态。",
        "给我机器人当前状态。",
        "查询机械臂最新状态。",
        "拉取当前状态。",
        "返回当前关节和末端状态。",
        "现在读取一次 state。",
        "把机器人状态发出来。",
        "获取当前位姿信息。",
        "读一下当前状态快照。",
        "执行 get_state。",
    ]
    return Sample(choose(rng, templates), [cmd], "get_state")


def gen_reset_scene(rng: random.Random) -> Sample:
    cmd = {"action": "reset_scene"}
    templates = [
        "重置整个场景。",
        "把仿真场景恢复到初始状态。",
        "执行场景重置。",
        "场景回到初始配置。",
        "重开这一局场景。",
        "请 reset scene。",
        "把环境清回初始。",
        "恢复默认场景。",
        "重新初始化仿真场景。",
        "先执行 reset_scene。",
    ]
    return Sample(choose(rng, templates), [cmd], "reset_scene")


def gen_move_ee(rng: random.Random) -> Sample:
    cube_pos = sample_cube_pos(rng)
    cube_color = sample_cube_color(rng)
    relation = choose(rng, ["上方", "前方", "后方", "左侧", "右侧"])

    if relation == "上方":
        dx, dy, dz = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(0.13, 0.22)
    elif relation == "前方":
        dx, dy, dz = rng.uniform(0.06, 0.12), rng.uniform(-0.03, 0.03), rng.uniform(0.11, 0.18)
    elif relation == "后方":
        dx, dy, dz = rng.uniform(-0.12, -0.06), rng.uniform(-0.03, 0.03), rng.uniform(0.11, 0.18)
    elif relation == "左侧":
        dx, dy, dz = rng.uniform(-0.03, 0.03), rng.uniform(0.07, 0.13), rng.uniform(0.11, 0.18)
    else:
        dx, dy, dz = rng.uniform(-0.03, 0.03), rng.uniform(-0.13, -0.07), rng.uniform(0.11, 0.18)

    pos = clamp_workspace_pos([cube_pos[0] + dx, cube_pos[1] + dy, cube_pos[2] + dz])
    quat = choose(rng, [[0.0, 1.0, 0.0, 0.0], [0.0, 0.924, 0.383, 0.0]])
    cmd: dict[str, Any] = {"action": "move_ee", "pos": pos, "quat": quat}
    maybe_add_steps(cmd, rng, prob=0.55, lo=40, hi=180)

    obj_ref = choose(rng, [f"{cube_color}方块", f"{cube_color}积木", "方块", "小块", "那个块"])
    templates = [
        f"把末端移到{obj_ref}{relation}。",
        f"移动到{obj_ref}{relation}位置。",
        f"手爪去到{obj_ref}{relation}，准备下一步。",
        f"把 ee 放到{obj_ref}{relation}。",
        f"靠近{obj_ref}，停在它{relation}。",
        f"到{obj_ref}{relation}待命。",
        f"把机械臂末端对准{obj_ref}{relation}区域。",
        f"移过去，位置在{obj_ref}{relation}就行。",
        f"把手爪挪到{obj_ref}{relation}。",
        f"就位到{obj_ref}{relation}。",
    ]

    ee_pos = offset_pos_from_target(pos, rng, xy_delta=0.08, z_delta=0.08)
    ee_quat = sample_quat(rng)
    franka_qpos = sample_home_like_qpos(rng)
    scene_state = build_scene_state(
        cube_pos=cube_pos,
        cube_color=cube_color,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        franka_qpos=franka_qpos,
    )
    return Sample(
        choose(rng, templates),
        [cmd],
        "move_ee",
        scene_state=scene_state,
        force_state_context=True,
    )


def gen_set_qpos(rng: random.Random) -> Sample:
    qpos = sample_home_like_qpos(rng)
    cmd: dict[str, Any] = {"action": "set_qpos", "qpos": qpos}
    maybe_add_steps(cmd, rng, prob=0.6, lo=30, hi=150)

    templates = [
        f"将 Franka 的 9 维 qpos 设置为 {qpos}。",
        f"直接设定关节位置 qpos={qpos}。",
        f"把机器人关节状态切换到 {qpos}。",
        f"把机械臂 qpos 改成 {qpos}。",
        f"执行 set_qpos，目标是 {qpos}。",
        f"更新关节配置到 {qpos}。",
        f"设置机器人到该关节状态：{qpos}。",
        f"关节重定位到 {qpos}。",
        f"将当前关节置为 {qpos}。",
        f"把整机 qpos 设为 {qpos}。",
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
        f"将 dof {dofs} 位置更新为 {values}。",
        f"按索引 {dofs} 写入位置 {values}。",
        f"把这些关节直接拨到 {values}（idx={dofs}）。",
        f"执行关节位置覆盖：{dofs} -> {values}。",
        f"set_dofs_position 一下，目标 {values}。",
        f"局部关节位置重设为 {values}，索引 {dofs}。",
        f"直接设置 dofs={dofs} 的位置为 {values}。",
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
        f"关节位置控制到 {values}（索引 {dofs}）。",
        f"让这些 dof 做位置闭环到 {values}。",
        f"把位置控制目标设成 {values}，作用在 {dofs}。",
        f"执行 dof 位置控制，idx={dofs}。",
        f"控制关节位置：{dofs}->{values}。",
        f"对局部关节做 position tracking：{values}。",
        f"位置控制指令，下发到 {dofs}，目标 {values}。",
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
        f"对 {dofs} 下发速度目标 {values}。",
        f"做速度闭环，目标是 {values}。",
        f"关节速度控制：idx={dofs}，v={values}。",
        f"执行 velocity control 到 {values}。",
        f"把 dof {dofs} 的速度控制到 {values}。",
        f"对指定关节施加速度命令 {values}。",
        f"给关节 {dofs} 设置速度目标 {values}。",
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
        f"对 {dofs} 下发力控制 {values}。",
        f"做力控，作用关节 {dofs}，目标 {values}。",
        f"施加关节控制力 {values}（idx={dofs}）。",
        f"force control 到 {values}，关节索引 {dofs}。",
        f"将关节 {dofs} 的力设定为 {values}。",
        f"对这些 dof 输出力命令 {values}。",
        f"执行关节力控制：{dofs}->{values}。",
    ]
    return Sample(choose(rng, templates), [cmd], "control_dofs_force")


def gen_seq_grasp_lift(rng: random.Random) -> Sample:
    cube_pos = sample_cube_pos(rng)
    cube_color = sample_cube_color(rng)
    hover_pos = clamp_workspace_pos(
        [
            cube_pos[0] + rng.uniform(-0.02, 0.02),
            cube_pos[1] + rng.uniform(-0.02, 0.02),
            cube_pos[2] + rng.uniform(0.16, 0.22),
        ]
    )
    approach_pos = clamp_workspace_pos(
        [
            cube_pos[0] + rng.uniform(-0.015, 0.015),
            cube_pos[1] + rng.uniform(-0.015, 0.015),
            cube_pos[2] + rng.uniform(0.07, 0.10),
        ]
    )
    grasp_pos = clamp_workspace_pos(
        [
            cube_pos[0] + rng.uniform(-0.01, 0.01),
            cube_pos[1] + rng.uniform(-0.01, 0.01),
            cube_pos[2] + rng.uniform(0.035, 0.055),
        ]
    )
    lift_pos = clamp_workspace_pos([grasp_pos[0], grasp_pos[1], grasp_pos[2] + rng.uniform(0.10, 0.16)])
    quat = [0.0, 1.0, 0.0, 0.0]

    cmds = [
        {"action": "open_gripper", "position": 0.04},
        {"action": "move_ee", "pos": hover_pos, "quat": quat},
        {"action": "move_ee", "pos": approach_pos, "quat": quat},
        {"action": "move_ee", "pos": grasp_pos, "quat": quat},
        {"action": "close_gripper", "position": 0.0},
        {"action": "wait", "steps": rng.randint(10, 30)},
        {"action": "move_ee", "pos": lift_pos, "quat": quat},
    ]

    obj_ref = choose(rng, [f"桌上的{cube_color}方块", f"那个{cube_color}块", "桌上的方块", "那个小方块"])
    templates = [
        f"抓住{obj_ref}并抬起来。",
        f"把{obj_ref}抓起来再举高一点。",
        f"执行抓取抬升，把{obj_ref}拿起。",
        f"去夹住{obj_ref}，然后向上提。",
        f"把{obj_ref}拿起来。",
        f"对{obj_ref}做一次抓取并上提。",
        f"先抓住{obj_ref}，再抬升末端。",
        f"抓取{obj_ref}后上提。",
        f"把{obj_ref}捏住并提离桌面。",
        f"完成{obj_ref}的抓取抬升动作。",
    ]

    scene_state = build_scene_state(
        cube_pos=cube_pos,
        cube_color=cube_color,
        ee_pos=offset_pos_from_target(hover_pos, rng, xy_delta=0.07, z_delta=0.06),
        ee_quat=sample_quat(rng),
        franka_qpos=sample_home_like_qpos(rng),
    )
    return Sample(
        choose(rng, templates),
        cmds,
        "seq_grasp_lift",
        scene_state=scene_state,
        force_state_context=True,
    )


def gen_seq_place(rng: random.Random) -> Sample:
    cube_pos = sample_cube_pos(rng)
    cube_color = sample_cube_color(rng)
    place_dx, place_dy = choose(
        rng,
        [
            (rng.uniform(0.08, 0.14), rng.uniform(-0.03, 0.03)),
            (rng.uniform(-0.14, -0.08), rng.uniform(-0.03, 0.03)),
            (rng.uniform(-0.03, 0.03), rng.uniform(0.08, 0.14)),
            (rng.uniform(-0.03, 0.03), rng.uniform(-0.14, -0.08)),
        ],
    )
    place_down = clamp_workspace_pos([cube_pos[0] + place_dx, cube_pos[1] + place_dy, cube_pos[2] + 0.06])
    place_up = clamp_workspace_pos([place_down[0], place_down[1], place_down[2] + rng.uniform(0.09, 0.15)])
    quat = [0.0, 1.0, 0.0, 0.0]

    cmds = [
        {"action": "move_ee", "pos": place_up, "quat": quat},
        {"action": "move_ee", "pos": place_down, "quat": quat},
        {"action": "open_gripper", "position": 0.04},
        {"action": "wait", "steps": rng.randint(10, 25)},
        {"action": "move_ee", "pos": place_up, "quat": quat},
    ]

    obj_ref = choose(rng, [f"{cube_color}方块", f"{cube_color}块", "方块", "这个块"])
    templates = [
        f"把手里的{obj_ref}放到旁边位置。",
        f"执行放置，把{obj_ref}放下后撤离。",
        f"把{obj_ref}放到桌面另一侧。",
        f"放置当前抓取物，然后抬手离开。",
        f"把{obj_ref}轻放到目标区域。",
        f"完成一次放置动作：放下并退开。",
        f"将{obj_ref}放好，松爪后撤。",
        f"把这个物体放到新位置。",
        f"执行 place：落下、开爪、撤离。",
        f"把{obj_ref}放下并抬起末端。",
    ]

    scene_state = build_scene_state(
        cube_pos=cube_pos,
        cube_color=cube_color,
        ee_pos=offset_pos_from_target(place_up, rng, xy_delta=0.07, z_delta=0.06),
        ee_quat=sample_quat(rng),
        franka_qpos=sample_home_like_qpos(rng),
    )
    return Sample(
        choose(rng, templates),
        cmds,
        "seq_place",
        scene_state=scene_state,
        force_state_context=True,
    )


def gen_seq_reset_and_check(rng: random.Random) -> Sample:
    pos = sample_pos(rng)
    cmds = [
        {"action": "reset_scene"},
        {"action": "move_ee", "pos": pos, "quat": sample_quat(rng)},
        {"action": "wait", "steps": rng.randint(15, 40)},
        {"action": "get_state"},
    ]
    templates = [
        f"先重置场景，再把末端移动到 {pos}，稍等后返回当前状态。",
        f"reset 场景后移动到 {pos}，等待一下并读取状态。",
        f"执行 reset_scene，然后到 {pos}，最后 get_state。",
        f"先清场，再把 ee 挪到 {pos}，接着查状态。",
        f"重置后移动到 {pos}，等待稳定并输出状态。",
        f"场景恢复初始，末端到 {pos}，然后读取一次状态。",
        f"先 reset，再 move_ee 到 {pos}，最后看 state。",
        f"重置环境后就位到 {pos}，随后查询当前状态。",
        f"把场景重置并移动到 {pos}，等待后获取状态。",
        f"执行 reset->move->wait->get_state，目标点 {pos}。",
    ]
    return Sample(choose(rng, templates), cmds, "seq_reset_check")


def gen_seq_joint_then_state(rng: random.Random) -> Sample:
    qpos = sample_home_like_qpos(rng)
    cmds = [
        {"action": "set_qpos", "qpos": qpos},
        {"action": "wait", "steps": rng.randint(20, 80)},
        {"action": "get_state"},
    ]
    templates = [
        f"把 qpos 设为 {qpos}，等待稳定后读取状态。",
        f"设置关节到 {qpos}，稍等后 get_state。",
        f"先 set_qpos={qpos}，再等一会儿查看状态。",
        f"把机械臂调到 qpos {qpos} 并回传状态。",
        f"执行关节设定 {qpos}，等待后输出当前状态。",
        f"关节重配置到 {qpos}，随后读取机器人状态。",
        f"把姿态切到 {qpos}，稳定后查 state。",
        f"设置 qpos 后等待，再返回状态。",
        f"按 qpos={qpos} 就位，然后 get_state。",
        f"先设关节目标 {qpos}，再查询当前状态。",
    ]
    return Sample(choose(rng, templates), cmds, "seq_joint_state")


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
    """
    构造动作执行前场景状态。

    关键修复：
    - 不能把当前 ee_pos / qpos 直接设成指令目标。
    - 需要在目标附近随机偏移，模拟“动作发生前”的真实状态。
    """
    cube_pos = sample_cube_pos(rng)
    cube_color = sample_cube_color(rng)
    ee_pos = [r3(rng.uniform(0.50, 0.76)), r3(rng.uniform(-0.24, 0.24)), r3(rng.uniform(0.12, 0.36))]
    ee_quat = sample_quat(rng)
    franka_qpos = sample_home_like_qpos(rng)

    move_target: list[float] | None = None
    move_quat: list[float] | None = None
    qpos_target: list[float] | None = None

    for cmd in commands:
        action = str(cmd.get("action", "")).strip().lower()
        if move_target is None and action == "move_ee" and isinstance(cmd.get("pos"), list) and len(cmd["pos"]) == 3:
            move_target = [r3(v) for v in cmd["pos"]]
            if isinstance(cmd.get("quat"), list) and len(cmd["quat"]) == 4:
                move_quat = [r3(v) for v in cmd["quat"]]
        if qpos_target is None and action == "set_qpos" and isinstance(cmd.get("qpos"), list) and len(cmd["qpos"]) >= 9:
            qpos_target = [r3(v) for v in cmd["qpos"][:9]]
        if move_target is not None and qpos_target is not None:
            break

    if move_target is not None:
        ee_pos = offset_pos_from_target(move_target, rng, xy_delta=0.07, z_delta=0.08)
        if move_quat is not None:
            ee_quat = move_quat
    if qpos_target is not None:
        franka_qpos = offset_qpos_from_target(qpos_target, rng)

    return build_scene_state(
        cube_pos=cube_pos,
        cube_color=cube_color,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        franka_qpos=franka_qpos,
    )


def build_instruction_with_state_context(instruction: str, scene_state: dict[str, Any]) -> str:
    state_text = json.dumps(scene_state, ensure_ascii=False, separators=(",", ":"))
    return f"[STATE_CONTEXT]{state_text}[/STATE_CONTEXT]\n用户指令: {instruction}"


def _build_action_generator_map() -> dict[str, Callable[[random.Random], Sample]]:
    return {
        "move_ee": gen_move_ee,
        "open_gripper": gen_open_gripper,
        "close_gripper": gen_close_gripper,
        "wait": gen_wait,
        "step": gen_step,
        "get_state": gen_get_state,
        "reset_scene": gen_reset_scene,
        "set_qpos": gen_set_qpos,
        "set_dofs_position": gen_set_dofs_position,
        "control_dofs_position": gen_control_dofs_position,
        "control_dofs_velocity": gen_control_dofs_velocity,
        "control_dofs_force": gen_control_dofs_force,
        "seq_grasp_lift": gen_seq_grasp_lift,
        "seq_place": gen_seq_place,
        "seq_reset_check": gen_seq_reset_and_check,
        "seq_joint_state": gen_seq_joint_then_state,
    }


def load_action_weights_from_file(action_map_file: Path) -> dict[str, int]:
    if not action_map_file.exists():
        raise FileNotFoundError(f"action map file not found: {action_map_file}")

    obj = json.loads(action_map_file.read_text(encoding="utf-8"))
    return parse_action_weights_obj(obj)


def parse_action_weights_obj(obj: Any) -> dict[str, int]:
    if isinstance(obj, dict) and isinstance(obj.get("action_weights"), dict):
        obj = obj["action_weights"]

    if not isinstance(obj, dict):
        raise ValueError("action map must be a JSON object or {'action_weights': {...}}")

    parsed: dict[str, int] = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            raise ValueError("action map keys must be strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"action weight for '{key}' must be int")
        parsed[key] = value
    return parsed


def build_weighted_generators(
    action_weights: dict[str, int] | None,
) -> tuple[list[Callable[[random.Random], Sample]], dict[str, int]]:
    action_to_generator = _build_action_generator_map()
    resolved_weights = dict(DEFAULT_ACTION_WEIGHTS)

    if action_weights is not None:
        unknown_actions = sorted(set(action_weights.keys()) - set(action_to_generator.keys()))
        if unknown_actions:
            raise ValueError(f"unknown actions in action map: {unknown_actions}")
        for action, weight in action_weights.items():
            if weight < 0:
                raise ValueError(f"action weight must be >= 0 for action: {action}")
            resolved_weights[action] = weight

    weighted_funcs: list[Callable[[random.Random], Sample]] = []
    for action, generator in action_to_generator.items():
        weight = resolved_weights.get(action, 0)
        if weight > 0:
            weighted_funcs.extend([generator] * weight)

    if not weighted_funcs:
        raise ValueError("all action weights are zero; at least one action must have positive weight")
    return weighted_funcs, resolved_weights


def generate_dataset(
    num_samples: int,
    seed: int,
    *,
    state_context_ratio: float = 0.7,
    action_weights: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    weighted_funcs, resolved_action_weights = build_weighted_generators(action_weights)

    dedup: set[tuple[str, str]] = set()
    alpaca: list[dict[str, Any]] = []
    sharegpt: list[dict[str, Any]] = []
    action_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    state_context_count = 0

    max_trials = num_samples * 30
    trials = 0
    while len(alpaca) < num_samples and trials < max_trials:
        trials += 1
        sample = choose(rng, weighted_funcs)(rng)
        instruction = with_json_hint(sample.instruction, rng)
        payload = build_payload(sample.commands, rng)
        commands = validate_payload(payload)
        use_state_context = sample.force_state_context or (rng.random() < state_context_ratio)
        if use_state_context:
            scene_state = sample.scene_state if sample.scene_state is not None else sample_scene_state_for_commands(commands, rng)
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
        "action_weight_map": resolved_action_weights,
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
    parser.add_argument(
        "--action-map-file",
        type=Path,
        default=None,
        help="JSON file for action sampling weights. Format: {'action_weights': {...}} or flat object.",
    )
    parser.add_argument(
        "--action-map-json",
        type=str,
        default=None,
        help="JSON string for action sampling weights. Format: {'action_weights': {...}} or flat object.",
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

    action_weights: dict[str, int] | None = None
    if args.action_map_file is not None and args.action_map_json is not None:
        raise ValueError("use only one of --action-map-file or --action-map-json")

    if args.action_map_file is not None:
        action_weights = load_action_weights_from_file(args.action_map_file)
    elif args.action_map_json is not None:
        action_weights = parse_action_weights_obj(json.loads(args.action_map_json))

    alpaca, sharegpt, stats = generate_dataset(
        args.num_samples,
        args.seed,
        state_context_ratio=args.state_context_ratio,
        action_weights=action_weights,
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
    if args.action_map_file is not None:
        print(f"[ok] action_map: {args.action_map_file}")
    if args.action_map_json is not None:
        print("[ok] action_map: inline_json")
    print("[stats] action coverage:")
    for action, cnt in sorted(stats["action_counter"].items(), key=lambda kv: kv[0]):
        print(f"  - {action}: {cnt}")


if __name__ == "__main__":
    main()
