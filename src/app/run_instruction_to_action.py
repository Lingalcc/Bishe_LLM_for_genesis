#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.app_common import (
    build_interactive_env,
    collect_scene_state,
    predict_actions_from_instruction,
)
from src.app.unified_config import DEFAULT_CONFIG_PATH, get_section, load_config


HELP_TEXT = """
Instruction -> Action 模式
-------------------------
输入自然语言指令，模型输出 JSON action。

Special commands:
  /help      Show this help
  /quit      Exit program
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert instruction to action JSON via model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Unified config YAML path.")
    parser.add_argument("--instruction", type=str, default="", help="One-shot instruction.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw model response.")
    parser.add_argument(
        "--disable-sim-state",
        action="store_true",
        help="Disable simulation state injection before model prediction.",
    )
    return parser.parse_args(argv)


def run_once(instruction: str, cfg: dict, print_raw: bool, scene_state: dict | None) -> None:
    raw, payload = predict_actions_from_instruction(instruction, cfg, scene_state=scene_state)
    if print_raw:
        print("[model_raw]")
        print(raw)
    if scene_state is not None:
        print("[scene_state]")
        print(json.dumps(scene_state, ensure_ascii=False, indent=2))
    print("[action_json]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_instruction_to_action(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    instruction: str = "",
    print_raw: bool = False,
    disable_sim_state: bool = False,
) -> None:
    cfg = load_config(config_path)
    app_cfg = get_section(cfg, "app", "interactive")
    show_viewer = bool(app_cfg.get("show_viewer", True))

    state_cfg = get_section(cfg, "app", "state_injection")
    use_sim_state = bool(state_cfg.get("enable_instruction_to_action", True))
    if disable_sim_state:
        use_sim_state = False

    manager = None
    robot = None
    try:
        if use_sim_state:
            manager, robot = build_interactive_env(show_viewer=show_viewer)

        if instruction.strip():
            scene_state = collect_scene_state(manager) if manager is not None else None
            run_once(instruction.strip(), cfg, print_raw, scene_state)
            return

        print(HELP_TEXT.strip())
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
            try:
                scene_state = collect_scene_state(manager) if manager is not None else None
                run_once(text, cfg, print_raw, scene_state)
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_instruction_to_action(
        config_path=args.config,
        instruction=args.instruction,
        print_raw=args.print_raw,
        disable_sim_state=args.disable_sim_state,
    )


if __name__ == "__main__":
    main()
