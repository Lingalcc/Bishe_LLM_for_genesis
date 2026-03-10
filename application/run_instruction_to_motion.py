#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.app_common import (
    DEFAULT_APP_EXAMPLE_JSON,
    build_interactive_env,
    collect_scene_state,
    predict_actions_from_instruction,
    print_state,
)
from application.unified_config import DEFAULT_CONFIG_PATH, get_section, load_config


HELP_TEXT = """
Instruction -> Action -> Franka运动 模式
--------------------------------------
输入自然语言指令，模型先输出 action JSON，再由仿真引擎执行。

Special commands:
  /help      Show this help
  /state     Print current robot state
  /scene     Print all scene entities and states
  /example   Show action JSON example
  /quit      Exit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end instruction->action->motion simulation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Unified config JSON path.")
    parser.add_argument("--instruction", type=str, default="", help="One-shot instruction.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw model response.")
    parser.add_argument(
        "--disable-sim-state",
        action="store_true",
        help="Disable simulation state injection before model prediction.",
    )
    return parser.parse_args()


def execute_instruction(robot, instruction: str, cfg: dict, print_raw: bool, use_sim_state: bool) -> None:
    scene_state = collect_scene_state(robot.manager) if use_sim_state else None
    raw, payload = predict_actions_from_instruction(instruction, cfg, scene_state=scene_state)
    if print_raw:
        print("[model_raw]")
        print(raw)
    if scene_state is not None:
        print("[scene_state]")
        print(json.dumps(scene_state, ensure_ascii=False, indent=2))
    print("[action_json]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    results = robot.execute_json(payload)
    print("[ok] Executed commands:")
    for item in results:
        print(f"  - idx={item.get('index')} action={item.get('action')} status={item.get('status')}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    app_cfg = get_section(cfg, "app", "interactive")
    enabled = bool(app_cfg.get("enabled", True))
    show_viewer = bool(app_cfg.get("show_viewer", True))
    state_cfg = get_section(cfg, "app", "state_injection")
    use_sim_state = bool(state_cfg.get("enable_instruction_to_motion", True))
    if args.disable_sim_state:
        use_sim_state = False
    if not enabled:
        print("[app] disabled by config. Set app.interactive.enabled=true to run.")
        return

    manager = None
    robot = None
    try:
        manager, robot = build_interactive_env(show_viewer=show_viewer)

        if args.instruction.strip():
            execute_instruction(robot, args.instruction.strip(), cfg, args.print_raw, use_sim_state)
            return

        print(HELP_TEXT.strip())
        print("\nAction JSON Example:")
        print(DEFAULT_APP_EXAMPLE_JSON)
        while True:
            text = input("\ninstruction >>> ").strip()
            if not text:
                continue
            if text == "/quit":
                print("Exiting.")
                break
            if text == "/help":
                print(HELP_TEXT.strip())
                continue
            if text == "/example":
                print(DEFAULT_APP_EXAMPLE_JSON)
                continue
            if text == "/state":
                print_state(robot)
                continue
            if text == "/scene":
                entities = robot.manager.get_entities()
                print("[scene_entities]", sorted(entities.keys()))
                for name in sorted(entities.keys()):
                    try:
                        st = robot.manager.get_entity_state(name)
                    except Exception:
                        st = {}
                    print(f"- {name}: {json.dumps(st, ensure_ascii=False)}")
                continue
            try:
                execute_instruction(robot, text, cfg, args.print_raw, use_sim_state)
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


if __name__ == "__main__":
    main()
