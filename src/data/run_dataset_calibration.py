from __future__ import annotations

from pathlib import Path

from src.data_core.calibration import CalibrationConfig, calibrate_dataset


def run_dataset_calibration(
    *,
    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json"),
    max_print_errors: int = 10,
    strict: bool = False,
) -> dict:
    cfg = CalibrationConfig(
        dataset_file=dataset_file,
        max_print_errors=max_print_errors,
        strict=strict,
    )
    return calibrate_dataset(cfg)
