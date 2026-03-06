#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline_config.json"


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a JSON object")
    return data


def get_section(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    if isinstance(current, dict):
        return current
    return {}


def build_cli_args(
    section: dict[str, Any],
    option_keys: list[str],
    *,
    bool_keys: list[str] | None = None,
) -> list[str]:
    cli_args: list[str] = []
    bool_set = set(bool_keys or [])

    for key in option_keys:
        if key not in section:
            continue
        value = section[key]
        if value is None:
            continue

        flag = "--" + key.replace("_", "-")
        if key in bool_set:
            if bool(value):
                cli_args.append(flag)
            continue

        cli_args.extend([flag, str(value)])

    return cli_args
