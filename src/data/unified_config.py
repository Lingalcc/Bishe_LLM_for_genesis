#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.utils.config import (
    DEFAULT_CONFIG_PATH,
    build_cli_args,
    get_section,
    load_config,
)


def load_dataset_prepare_section(
    *,
    config_path: Path,
    section: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Load dataset_prepare.<section> config and merge into defaults.
    If config cannot be read or section is missing, defaults are returned.
    """
    merged: dict[str, Any] = dict(defaults)
    if not config_path.exists():
        return merged

    try:
        config = load_config(config_path)
    except ValueError:
        return merged

    resolved_section: dict[str, Any] = {}
    # Primary format: {"dataset_prepare": {"generate": {...}}}
    primary = get_section(config, "dataset_prepare", section)
    if primary:
        resolved_section = primary
    else:
        # Fallback format: {"generate": {...}}
        fallback = get_section(config, section)
        if fallback:
            resolved_section = fallback

    for key, value in resolved_section.items():
        if value is not None:
            merged[key] = value
    return merged


def load_dataset_prepare_runtime_config(
    *,
    config_path: Path,
    section: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Dedicated helper for runtime scripts:
    - start from hard-coded defaults
    - override by config file section if readable
    - keep defaults when config/section is missing or invalid
    """
    return load_dataset_prepare_section(
        config_path=config_path,
        section=section,
        defaults=defaults,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "get_section",
    "build_cli_args",
    "load_dataset_prepare_section",
    "load_dataset_prepare_runtime_config",
]
