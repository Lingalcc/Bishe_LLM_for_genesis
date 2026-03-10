#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.unified_config import DEFAULT_CONFIG_PATH, get_section, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Genesis interactive simulation app."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Unified config JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    section = get_section(cfg, "app", "interactive")
    enabled = bool(section.get("enabled", True))
    if not enabled:
        print("[app] disabled by config. Set app.interactive.enabled=true to run.")
        raise SystemExit(0)

    from application.interactive_robot_control import main

    main()
