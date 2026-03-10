from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.generate_genesis_franka_dataset import validate_payload


@dataclass(frozen=True)
class CalibrationConfig:
    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json")
    max_print_errors: int = 10
    strict: bool = False


def calibrate_dataset(cfg: CalibrationConfig) -> dict[str, Any]:
    if not cfg.dataset_file.exists():
        raise FileNotFoundError(f"dataset file not found: {cfg.dataset_file}")

    rows = json.loads(cfg.dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("dataset must be a JSON list")

    total = len(rows)
    valid = 0
    invalid = 0
    missing_instruction = 0
    missing_output = 0
    invalid_examples: list[str] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid += 1
            if len(invalid_examples) < cfg.max_print_errors:
                invalid_examples.append(f"row={idx} reason=row_not_dict")
            continue

        instruction = row.get("instruction")
        output = row.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            missing_instruction += 1
            invalid += 1
            if len(invalid_examples) < cfg.max_print_errors:
                invalid_examples.append(f"row={idx} reason=missing_instruction")
            continue
        if not isinstance(output, str) or not output.strip():
            missing_output += 1
            invalid += 1
            if len(invalid_examples) < cfg.max_print_errors:
                invalid_examples.append(f"row={idx} reason=missing_output")
            continue

        try:
            payload = json.loads(output)
            validate_payload(payload)
            valid += 1
        except Exception as err:
            invalid += 1
            if len(invalid_examples) < cfg.max_print_errors:
                invalid_examples.append(
                    f"row={idx} reason=invalid_json_or_schema err={type(err).__name__}: {err}"
                )

    report = {
        "dataset_file": str(cfg.dataset_file),
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "missing_instruction": missing_instruction,
        "missing_output": missing_output,
        "pass_rate": (valid / total) if total else 0.0,
        "errors": invalid_examples,
    }
    if cfg.strict and invalid > 0:
        raise RuntimeError(f"dataset calibration failed: invalid={invalid}")
    return report


def calibrate_from_merged_config(config: dict[str, Any]) -> dict[str, Any]:
    section = (
        config.get("dataset_prepare", {}).get("calibration", {})
        if isinstance(config.get("dataset_prepare"), dict)
        else {}
    )
    cfg = CalibrationConfig(
        dataset_file=Path(section.get("dataset_file", CalibrationConfig.dataset_file)),
        max_print_errors=int(section.get("max_print_errors", CalibrationConfig.max_print_errors)),
        strict=bool(section.get("strict", CalibrationConfig.strict)),
    )
    return calibrate_dataset(cfg)
