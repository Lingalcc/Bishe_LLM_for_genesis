from __future__ import annotations

from pathlib import Path

from src.sim_core.runtime import SimRuntimeConfig, run_instruction_to_action as core_run_instruction_to_action


def run_instruction_to_action(
    *,
    config_path: Path,
    instruction: str,
    print_raw: bool = False,
    disable_sim_state: bool = False,
) -> dict:
    cfg = SimRuntimeConfig(
        config_path=config_path,
        instruction=instruction,
        print_raw=print_raw,
        disable_sim_state=disable_sim_state,
    )
    return core_run_instruction_to_action(cfg)
