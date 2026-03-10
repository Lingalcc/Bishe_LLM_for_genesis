#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
LEGACY_DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
# Keep this name for backward compatibility with existing imports.
DEFAULT_CONFIG_PATH = DEFAULT_BASE_CONFIG_PATH


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
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
        raise ValueError(f"config root must be a mapping object: {config_path}")
    return data


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep merge two mappings.
    - Nested dicts are merged recursively.
    - Scalar/list values are replaced by override values.
    """
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge_dicts(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load a single YAML config file without any merge behavior.
    """
    return _load_yaml_mapping(config_path)


def load_merged_config(
    *,
    override_config_path: Path | None = None,
    base_config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load base config and deep-merge optional experiment override config.

    Search order for base config:
    1) `base_config_path` argument
    2) `configs/base.yaml`
    3) fallback `configs/default.yaml` (legacy compatibility)
    """
    if base_config_path is not None:
        resolved_base = base_config_path
    elif DEFAULT_BASE_CONFIG_PATH.exists():
        resolved_base = DEFAULT_BASE_CONFIG_PATH
    else:
        resolved_base = LEGACY_DEFAULT_CONFIG_PATH

    base = _load_yaml_mapping(resolved_base)
    if override_config_path is None:
        return base

    override = _load_yaml_mapping(override_config_path)
    return deep_merge_dicts(base, override)


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
