#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.unified_config import DEFAULT_CONFIG_PATH, get_section, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regression tests for Genesis tool-call stack.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Unified config JSON path.",
    )
    parser.add_argument(
        "--target",
        choices=["all", "manager", "controller", "basic"],
        default=None,
        help="Test target group.",
    )
    return parser.parse_args()


def run_file(file_name: str) -> None:
    script_path = REPO_ROOT / file_name
    if not script_path.exists():
        raise FileNotFoundError(f"script not found: {script_path}")
    cmd = [sys.executable, str(script_path)]
    print("[test]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    section = get_section(cfg, "test", "regression")
    target = args.target or str(section.get("target", "all"))

    if target == "all":
        run_file("test_genesis_manager.py")
        run_file("test_robot_json_controller.py")
        run_file("test_genesis.py")
        return

    if target == "manager":
        run_file("test_genesis_manager.py")
    elif target == "controller":
        run_file("test_robot_json_controller.py")
    elif target == "basic":
        run_file("test_genesis.py")
    else:
        raise ValueError(f"invalid regression target in config: {target}")


if __name__ == "__main__":
    main()
