#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sim_core.runtime import SimRuntimeConfig, run_instruction_to_motion
from src.utils.config import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E simulation experiment runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/04_sim_exp/configs/e2e_sim.yaml"),
        help="Experiment-local override config YAML path.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Global base config YAML path.",
    )
    parser.add_argument("--instruction", type=str, required=True, help="Natural language instruction.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw model output.")
    parser.add_argument(
        "--disable-sim-state",
        action="store_true",
        help="Disable simulation state injection before model prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )

    runtime_cfg = SimRuntimeConfig(
        config_path=None,
        instruction=args.instruction,
        print_raw=args.print_raw,
        disable_sim_state=args.disable_sim_state,
    )
    result = run_instruction_to_motion(runtime_cfg, merged_config=merged_config)

    if result.get("disabled"):
        print("[app] disabled by config.")
        return

    if args.print_raw:
        print("[model_raw]")
        print(result["raw"])
    if result.get("scene_state") is not None:
        print("[scene_state]")
        print(json.dumps(result["scene_state"], ensure_ascii=False, indent=2))
    print("[action_json]")
    print(json.dumps(result["payload"], ensure_ascii=False, indent=2))
    print("[ok] executed commands:")
    for item in result["results"]:
        print(f"  - idx={item.get('index')} action={item.get('action')} status={item.get('status')}")


if __name__ == "__main__":
    main()
