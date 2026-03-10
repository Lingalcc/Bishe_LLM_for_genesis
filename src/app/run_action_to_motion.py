from __future__ import annotations

from pathlib import Path

from src.sim_core.runtime import SimRuntimeConfig, run_action_to_motion as core_run_action_to_motion


def run_action_to_motion(
    *,
    config_path: Path,
    action: str = "",
    action_file: Path | None = None,
) -> dict:
    cfg = SimRuntimeConfig(
        config_path=config_path,
        action=action,
        action_file=action_file,
    )
    return core_run_action_to_motion(cfg)
