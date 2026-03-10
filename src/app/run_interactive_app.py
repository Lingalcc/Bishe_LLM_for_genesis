from __future__ import annotations

from pathlib import Path

from src.sim_core.runtime import run_interactive_session


def run_interactive_app(config_path: Path) -> None:
    run_interactive_session(config_path)
