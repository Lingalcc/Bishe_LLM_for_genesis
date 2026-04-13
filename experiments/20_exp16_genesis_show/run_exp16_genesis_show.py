#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "20_exp16_genesis_show" / "configs" / "genesis_show.yaml"
DEFAULT_REPORTS_DIR = REPO_ROOT / "experiments" / "20_exp16_genesis_show" / "reports"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_section, load_merged_config
from src.utils.run_meta import record_run_meta
from src.utils.secrets import safe_json_dumps


SUPPORTED_ACTIONS: list[dict[str, Any]] = [
    {
        "action": "move_ee",
        "description": "移动末端执行器到目标位置与姿态。",
        "example_json": {"commands": [{"action": "move_ee", "pos": [0.65, 0.0, 0.15], "quat": [0, 1, 0, 0]}]},
        "recommended_for_demo": True,
    },
    {
        "action": "open_gripper",
        "description": "张开夹爪。",
        "example_json": {"commands": [{"action": "open_gripper", "position": 0.04}]},
        "recommended_for_demo": True,
    },
    {
        "action": "close_gripper",
        "description": "闭合夹爪。",
        "example_json": {"commands": [{"action": "close_gripper", "position": 0.0}]},
        "recommended_for_demo": True,
    },
    {
        "action": "wait",
        "description": "推进仿真若干步，用于等待动作稳定。",
        "example_json": {"commands": [{"action": "wait"}]},
        "recommended_for_demo": True,
    },
    {
        "action": "get_state",
        "description": "读取当前机械臂状态。",
        "example_json": {"commands": [{"action": "get_state"}]},
        "recommended_for_demo": True,
    },
    {
        "action": "reset_scene",
        "description": "执行轻量级场景复位步进。",
        "example_json": {"commands": [{"action": "reset_scene"}]},
        "recommended_for_demo": False,
    },
    {
        "action": "set_qpos",
        "description": "直接设置整条机械臂与夹爪的关节位置。",
        "example_json": {
            "commands": [
                {
                    "action": "set_qpos",
                    "qpos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
                }
            ]
        },
        "recommended_for_demo": False,
    },
    {
        "action": "set_dofs_position",
        "description": "为指定关节设置目标位置。",
        "example_json": {
            "commands": [{"action": "set_dofs_position", "values": [-0.52, -1.2], "dofs_idx_local": [1, 3]}]
        },
        "recommended_for_demo": False,
    },
    {
        "action": "control_dofs_position",
        "description": "按位置控制方式驱动指定关节。",
        "example_json": {
            "commands": [{"action": "control_dofs_position", "values": [0.2, -0.2], "dofs_idx_local": [1, 3]}]
        },
        "recommended_for_demo": False,
    },
    {
        "action": "control_dofs_velocity",
        "description": "按速度控制方式驱动指定关节。",
        "example_json": {
            "commands": [{"action": "control_dofs_velocity", "values": [0.1, -0.1], "dofs_idx_local": [1, 3]}]
        },
        "recommended_for_demo": False,
    },
    {
        "action": "control_dofs_force",
        "description": "对指定自由度施加力或力矩。",
        "example_json": {
            "commands": [{"action": "control_dofs_force", "values": [0.0, 0.0, -2.0], "dofs_idx_local": [0, 1, 2]}]
        },
        "recommended_for_demo": False,
    },
]


EXAMPLE_INSTRUCTIONS: list[dict[str, str]] = [
    {
        "title": "方块上方定位",
        "instruction": "移动到方块上方10厘米处",
        "notes": "最稳的展示口令之一，依赖场景中的 cube 状态注入。",
    },
    {
        "title": "打开夹爪并就位",
        "instruction": "先张开夹爪，再移动到工作台中央上方30厘米处",
        "notes": "适合先展示指令到 JSON 的可解释性。",
    },
    {
        "title": "抓取并抬起",
        "instruction": "移动到方块上方，下降到方块高度后闭合夹爪，再抬起到更高位置",
        "notes": "适合展示多步复合动作。",
    },
    {
        "title": "读取状态",
        "instruction": "读取当前机械臂状态",
        "notes": "通常会触发 get_state，便于调试。",
    },
    {
        "title": "末端定位",
        "instruction": "将机械臂移动到x=0.5米、y=0.0米、z=0.3米的位置，保持朝下姿态",
        "notes": "适合验证绝对坐标控制。",
    },
    {
        "title": "夹爪闭合",
        "instruction": "移动到方块附近后关闭夹爪",
        "notes": "适合快速观察夹爪执行结果。",
    },
    {
        "title": "等待稳定",
        "instruction": "保持当前姿态并等待一下",
        "notes": "适合配合前后状态对比。",
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp16 Genesis Show：最终策略的交互式执行展示。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="实验覆盖配置。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG, help="基础配置文件。")
    parser.add_argument("--instruction", type=str, default="", help="单次执行的自然语言指令。")
    parser.add_argument("--interactive", action="store_true", help="强制进入交互模式。")
    parser.add_argument("--print-raw", action="store_true", help="打印模型原始输出。")
    parser.add_argument("--disable-sim-state", action="store_true", help="关闭场景状态注入。")
    parser.add_argument("--hide-viewer", action="store_true", help="关闭 Genesis viewer，仅打印终端结果。")
    parser.add_argument("--list-examples", action="store_true", help="打印推荐指令与动作示例后退出。")
    parser.add_argument(
        "--strict-vllm-compat-check",
        action="store_true",
        help="不要沿用 exp15 的跳检策略，强制执行 vLLM/compressed-tensors 版本检查。",
    )
    return parser.parse_args(argv)


def resolve_demo_profile(merged_config: dict[str, Any]) -> dict[str, Any]:
    inference_cfg = get_section(merged_config, "app", "inference")
    local_cfg = inference_cfg.get("local", {}) if isinstance(inference_cfg.get("local"), dict) else {}
    exp_cfg = get_section(merged_config, "exp16_genesis_show")
    return {
        "mode": str(inference_cfg.get("mode", "api")).strip().lower(),
        "model_path": str(local_cfg.get("model_path", "")).strip(),
        "backend": str(local_cfg.get("backend", "auto")).strip().lower(),
        "quantization": str(local_cfg.get("quantization", "")).strip(),
        "max_model_len": int(local_cfg.get("max_model_len", 0) or 0),
        "gpu_memory_utilization": float(local_cfg.get("gpu_memory_utilization", 0.0) or 0.0),
        "reports_dir": str(exp_cfg.get("reports_dir", DEFAULT_REPORTS_DIR)),
        "show_examples_on_start": bool(exp_cfg.get("show_examples_on_start", True)),
        "skip_vllm_compat_check": bool(exp_cfg.get("skip_vllm_compat_check", False)),
    }


def _load_demo_config(args: argparse.Namespace) -> dict[str, Any]:
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )
    if args.hide_viewer:
        merged_config.setdefault("app", {}).setdefault("interactive", {})["show_viewer"] = False
    return merged_config


def _resolve_reports_dir(merged_config: dict[str, Any]) -> Path:
    exp_cfg = get_section(merged_config, "exp16_genesis_show")
    reports_dir = exp_cfg.get("reports_dir", DEFAULT_REPORTS_DIR)
    path = Path(str(reports_dir))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _make_session_dir(reports_dir: Path) -> Path:
    session_dir = reports_dir / "sessions" / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _resolve_skip_vllm_compat_check(merged_config: dict[str, Any], args: argparse.Namespace) -> bool:
    profile = resolve_demo_profile(merged_config)
    skip_check = bool(profile.get("skip_vllm_compat_check", False))
    if args.strict_vllm_compat_check:
        skip_check = False
    return skip_check


def _apply_vllm_compat_policy(skip_check: bool) -> None:
    if skip_check:
        os.environ["LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK"] = "1"
    else:
        os.environ.pop("LLM_GENESIS_SKIP_VLLM_COMPAT_CHECK", None)


def _print_supported_actions() -> None:
    print("可执行动作示例：")
    for item in SUPPORTED_ACTIONS:
        suffix = "（推荐现场演示）" if item.get("recommended_for_demo") else ""
        print(f"- {item['action']}{suffix}: {item['description']}")
        print(json.dumps(item["example_json"], ensure_ascii=False))


def _print_instruction_examples() -> None:
    print("推荐自然语言指令示例：")
    for idx, item in enumerate(EXAMPLE_INSTRUCTIONS, start=1):
        print(f"{idx}. {item['title']}：{item['instruction']}")
        print(f"   说明：{item['notes']}")


def _print_banner(merged_config: dict[str, Any], *, session_dir: Path) -> None:
    profile = resolve_demo_profile(merged_config)
    print("Exp16 Genesis Show")
    print("默认策略：Top18Rank8 + vLLM + AWQ（compressed-tensors）")
    print(f"模型路径：{profile['model_path']}")
    print(f"推理后端：{profile['backend']} | 量化：{profile['quantization'] or 'none'}")
    print(f"max_model_len：{profile['max_model_len']} | gpu_memory_utilization：{profile['gpu_memory_utilization']:.2f}")
    if profile["backend"] == "vllm":
        if profile.get("skip_vllm_compat_check", False):
            print("vLLM 版本检查：默认跳过（与 exp15 已验证口径保持一致）")
        else:
            print("vLLM 版本检查：严格模式")
    print(f"会话目录：{session_dir}")
    print("内置命令：/help /state /scene /example /examples /actions /raw on|off /quit")
    if profile["show_examples_on_start"]:
        _print_instruction_examples()


def _build_turn_report(
    *,
    turn_index: int,
    instruction: str,
    scene_state: dict[str, Any] | None,
    model_raw: str,
    payload: dict[str, Any],
    results: list[dict[str, Any]],
    elapsed_sec: float,
    error: str | None = None,
) -> dict[str, Any]:
    ok_count = sum(1 for item in results if item.get("status") == "ok")
    return {
        "turn_index": turn_index,
        "instruction": instruction,
        "scene_state": scene_state,
        "model_raw": model_raw,
        "payload": payload,
        "results": results,
        "elapsed_sec": elapsed_sec,
        "num_commands": len(results),
        "num_ok_commands": ok_count,
        "execution_success": error is None and bool(results) and ok_count == len(results),
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _save_turn_report(session_dir: Path, report: dict[str, Any]) -> Path:
    filename = f"turn_{int(report['turn_index']):03d}.json"
    path = session_dir / filename
    path.write_text(safe_json_dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_execution_results(results: list[dict[str, Any]]) -> None:
    print("[execution_results]")
    for item in results:
        if item.get("status") == "ok":
            result_text = json.dumps(item.get("result", {}), ensure_ascii=False)
            print(f"  - idx={item.get('index')} action={item.get('action')} status=ok result={result_text}")
        else:
            print(
                f"  - idx={item.get('index')} action={item.get('action')} "
                f"status=error error={item.get('error', '')}"
            )


def run_single_demo(
    *,
    merged_config: dict[str, Any],
    instruction: str,
    print_raw: bool,
    disable_sim_state: bool,
    session_dir: Path,
) -> Path:
    from src.sim_core.runtime import SimRuntimeConfig, run_instruction_to_motion

    started_at = time.perf_counter()
    result = run_instruction_to_motion(
        SimRuntimeConfig(
            instruction=instruction,
            print_raw=print_raw,
            disable_sim_state=disable_sim_state,
        ),
        merged_config=copy.deepcopy(merged_config),
    )
    elapsed_sec = time.perf_counter() - started_at
    model_raw = str(result.get("raw", ""))
    payload = result.get("payload", {})
    scene_state = result.get("scene_state")
    results = result.get("results", [])
    if not isinstance(payload, dict):
        raise TypeError("payload 必须是对象。")
    if not isinstance(results, list):
        raise TypeError("results 必须是列表。")

    if print_raw:
        print("[model_raw]")
        print(model_raw)
    if scene_state is not None:
        print("[scene_state]")
        print(json.dumps(scene_state, ensure_ascii=False, indent=2))
    print("[action_json]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    _print_execution_results(results)

    report = _build_turn_report(
        turn_index=1,
        instruction=instruction,
        scene_state=scene_state if isinstance(scene_state, dict) else None,
        model_raw=model_raw,
        payload=payload,
        results=results,
        elapsed_sec=elapsed_sec,
    )
    report_path = _save_turn_report(session_dir, report)
    print(f"[exp16] 单次执行耗时：{elapsed_sec:.3f}s")
    print(f"[exp16] 结果已保存：{report_path}")
    return report_path


def run_interactive_demo(
    *,
    merged_config: dict[str, Any],
    print_raw: bool,
    disable_sim_state: bool,
    session_dir: Path,
) -> None:
    from src.app.app_common import (
        DEFAULT_APP_EXAMPLE_JSON,
        build_interactive_env,
        collect_scene_state,
        preload_local_engine,
        predict_actions_from_instruction,
        print_state,
    )

    app_cfg = get_section(merged_config, "app", "interactive")
    if not bool(app_cfg.get("enabled", True)):
        print("[app] disabled by config. Set app.interactive.enabled=true to run.")
        return

    show_viewer = bool(app_cfg.get("show_viewer", True))
    state_cfg = get_section(merged_config, "app", "state_injection")
    use_sim_state = bool(state_cfg.get("enable_instruction_to_motion", True))
    if disable_sim_state:
        use_sim_state = False

    inference_cfg = get_section(merged_config, "app", "inference")
    inference_mode = str(inference_cfg.get("mode", "api")).strip().lower()

    manager = None
    raw_enabled = print_raw
    turn_index = 0

    try:
        if inference_mode == "local":
            print("[exp16] 预加载本地推理引擎...")
            try:
                preload_local_engine(merged_config, warmup=True)
                print("[exp16] 本地推理引擎已就绪。")
            except Exception as exc:
                print(f"[exp16] 预热失败，将在首轮推理时重试：{type(exc).__name__}: {exc}")

        manager, robot = build_interactive_env(show_viewer=show_viewer, cfg=merged_config)
        _print_banner(merged_config, session_dir=session_dir)
        print("输入自然语言后，模型会生成 action JSON 并立即在 Genesis 中执行。")
        print("[example_action_json]")
        print(DEFAULT_APP_EXAMPLE_JSON)

        while True:
            raw = input("\nexp16>>> ").strip()
            if not raw:
                continue
            if raw == "/quit":
                break
            if raw == "/help":
                print("内置命令：/help /state /scene /example /examples /actions /raw on|off /quit")
                continue
            if raw == "/state":
                print_state(robot)
                continue
            if raw == "/scene":
                entities = robot.manager.get_entities()
                print("[scene_entities]", sorted(entities.keys()))
                for name in sorted(entities.keys()):
                    try:
                        state = robot.manager.get_entity_state(name)
                    except Exception:
                        state = {}
                    print(f"- {name}: {json.dumps(state, ensure_ascii=False)}")
                continue
            if raw == "/example":
                print(DEFAULT_APP_EXAMPLE_JSON)
                continue
            if raw == "/examples":
                _print_instruction_examples()
                continue
            if raw == "/actions":
                _print_supported_actions()
                continue
            if raw.startswith("/raw"):
                parts = raw.split(maxsplit=1)
                if len(parts) == 2 and parts[1] in {"on", "off"}:
                    raw_enabled = parts[1] == "on"
                    print(f"[exp16] 模型原始输出已{'开启' if raw_enabled else '关闭'}。")
                else:
                    print("[exp16] 用法：/raw on|off")
                continue

            turn_index += 1
            scene_state: dict[str, Any] | None = None
            model_raw = ""
            payload: dict[str, Any] = {}
            results: list[dict[str, Any]] = []
            try:
                started_at = time.perf_counter()
                scene_state = collect_scene_state(robot.manager) if use_sim_state else None
                model_raw, payload = predict_actions_from_instruction(
                    raw,
                    merged_config,
                    scene_state=scene_state,
                )
                results = robot.execute_json(payload)
                elapsed_sec = time.perf_counter() - started_at

                if raw_enabled:
                    print("[model_raw]")
                    print(model_raw)
                print("[action_json]")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                _print_execution_results(results)

                report = _build_turn_report(
                    turn_index=turn_index,
                    instruction=raw,
                    scene_state=scene_state,
                    model_raw=model_raw,
                    payload=payload,
                    results=results,
                    elapsed_sec=elapsed_sec,
                )
                report_path = _save_turn_report(session_dir, report)
                print(
                    f"[exp16] 第 {turn_index} 轮耗时 {elapsed_sec:.3f}s，"
                    f"执行{'成功' if report['execution_success'] else '存在失败'}。"
                )
                print(f"[exp16] 已保存：{report_path}")
            except Exception as exc:
                elapsed_sec = 0.0
                error_text = f"{type(exc).__name__}: {exc}"
                print(f"[exp16] 本轮执行失败：{error_text}")
                report = _build_turn_report(
                    turn_index=turn_index,
                    instruction=raw,
                    scene_state=scene_state,
                    model_raw=model_raw,
                    payload=payload,
                    results=results,
                    elapsed_sec=elapsed_sec,
                    error=error_text,
                )
                report_path = _save_turn_report(session_dir, report)
                print(f"[exp16] 已保存失败记录：{report_path}")
    except KeyboardInterrupt:
        print("\n[exp16] 用户中断，会话结束。")
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_examples:
        _print_instruction_examples()
        _print_supported_actions()
        return

    merged_config = _load_demo_config(args)
    skip_vllm_compat_check = _resolve_skip_vllm_compat_check(merged_config, args)
    _apply_vllm_compat_policy(skip_vllm_compat_check)
    reports_dir = _resolve_reports_dir(merged_config)
    session_dir = _make_session_dir(reports_dir)
    record_run_meta(
        session_dir,
        merged_config=merged_config,
        cli_args=vars(args),
        argv=sys.argv,
        extra_meta={
            "entry": "experiments/20_exp16_genesis_show/run_exp16_genesis_show.py",
            "experiment": "exp16_genesis_show",
            "default_strategy": "Top18Rank8_vLLM_AWQ",
            "skip_vllm_compat_check": skip_vllm_compat_check,
        },
    )

    if args.instruction and not args.interactive:
        run_single_demo(
            merged_config=merged_config,
            instruction=args.instruction,
            print_raw=args.print_raw,
            disable_sim_state=args.disable_sim_state,
            session_dir=session_dir,
        )
        return

    run_interactive_demo(
        merged_config=merged_config,
        print_raw=args.print_raw,
        disable_sim_state=args.disable_sim_state,
        session_dir=session_dir,
    )


if __name__ == "__main__":
    main()
