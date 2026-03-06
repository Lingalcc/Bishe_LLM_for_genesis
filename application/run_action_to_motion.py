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

from pipeline.app.app_common import DEFAULT_APP_EXAMPLE_JSON, build_interactive_env, print_state
from pipeline.unified_config import DEFAULT_CONFIG_PATH, get_section, load_config


HELP_TEXT = """
Action -> Franka运动 模式
------------------------
输入 action JSON，仿真引擎执行并驱动 Franka 运动。

输入格式:
1) {"action": "open_gripper"}
2) {"commands": [{"action":"move_ee","pos":[0.65,0,0.18],"quat":[0,1,0,0]}]}

Special commands:
  /help      Show this help
  /state     Print current robot state
  /scene     Print all scene entities and states
  /example   Show action JSON example
  /quit      Exit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute action JSON in Genesis simulation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Unified config JSON path.")
    parser.add_argument("--action", type=str, default="", help="One-shot action JSON text.")
    parser.add_argument("--action-file", type=Path, default=None, help="One-shot action JSON file.")
    return parser.parse_args()


def load_action_text(args: argparse.Namespace) -> str:
    if args.action_file is not None:
        return args.action_file.read_text(encoding="utf-8").strip()
    return args.action.strip()


def execute_and_print(robot, raw_action: str) -> None:
    results = robot.execute_json(raw_action)
    print("[ok] Executed commands:")
    for item in results:
        print(f"  - idx={item.get('index')} action={item.get('action')} status={item.get('status')}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    app_cfg = get_section(cfg, "app", "interactive")
    enabled = bool(app_cfg.get("enabled", True))
    show_viewer = bool(app_cfg.get("show_viewer", True))
    if not enabled:
        print("[app] disabled by config. Set app.interactive.enabled=true to run.")
        return

    manager = None
    robot = None
    try:
        manager, robot = build_interactive_env(show_viewer=show_viewer)
        one_shot = load_action_text(args)
        if one_shot:
            execute_and_print(robot, one_shot)
            return

        print(HELP_TEXT.strip())
        print("\nExample action JSON:")
        print(DEFAULT_APP_EXAMPLE_JSON)
        while True:
            raw = input("\naction >>> ").strip()
            if not raw:
                continue
            if raw == "/quit":
                print("Exiting.")
                break
            if raw == "/help":
                print(HELP_TEXT.strip())
                continue
            if raw == "/example":
                print(DEFAULT_APP_EXAMPLE_JSON)
                continue
            if raw == "/state":
                print_state(robot)
                continue
            if raw == "/scene":
                scene_state = robot.manager.get_entities()
                print("[scene_entities]", sorted(scene_state.keys()))
                for name in sorted(scene_state.keys()):
                    try:
                        st = robot.manager.get_entity_state(name)
                    except Exception:
                        st = {}
                    print(f"- {name}: {json.dumps(st, ensure_ascii=False)}")
                continue
            try:
                execute_and_print(robot, raw)
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
