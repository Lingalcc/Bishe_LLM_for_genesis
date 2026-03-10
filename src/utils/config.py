#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install dependency `pyyaml` first."
        ) from exc

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping object")
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
