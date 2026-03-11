#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GENESIS_PATH = REPO_ROOT / "Genesis"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_GENESIS_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_GENESIS_PATH))

from src.genesis.interactive_env import GenesisInteractiveTestEnv


def main() -> None:
    env = GenesisInteractiveTestEnv(show_viewer=True)
    try:
        env.setup()
        if env.robot is None:
            raise RuntimeError("Robot controller init failed.")

        print("[cube_positions]")
        print(json.dumps(env.get_cube_positions(), ensure_ascii=False, indent=2))

        print("[interactive]")
        print("输入 action JSON 控制 Franka；空行表示保持当前姿态不动。")
        print("命令: /help /cube /state /home /quit")
        print('示例: {"commands":[{"action":"move_ee","pos":[0.65,0.0,0.18],"quat":[0,1,0,0]}]}')

        while True:
            raw = input(">>> ").strip()
            if not raw:
                continue
            if raw == "/quit":
                break
            if raw == "/help":
                print("命令: /help /cube /state /home /quit")
                print("动作不支持 steps 字段；每条命令会自动执行固定步进。")
                continue
            if raw == "/cube":
                print(json.dumps(env.get_cube_positions(), ensure_ascii=False, indent=2))
                continue
            if raw == "/state":
                if env.manager is None:
                    print("{}")
                else:
                    print(json.dumps(env.manager.get_entity_state("franka"), ensure_ascii=False, indent=2))
                continue
            if raw == "/home":
                env.move_robot_to_init_pose_with_ik()
                print("[ok] moved to init IK pose")
                continue

            results = env.execute_action_json(raw)
            print("[action_results]")
            for item in results:
                msg = f"idx={item.get('index')} action={item.get('action')} status={item.get('status')}"
                if item.get("status") == "error":
                    msg += f" error={item.get('error')}"
                print(msg)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
