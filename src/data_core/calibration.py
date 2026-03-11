"""Dataset validation / calibration.

Reads an Alpaca-format dataset and validates every row:
  - instruction / output fields are present and non-empty
  - output is valid JSON with a ``commands`` array
  - each command passes the project's ``toolcall_validator``

Reports statistics and optionally raises on violations.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval_core.toolcall_validator import validate_payload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationConfig:
    """Parameters for dataset calibration / validation."""

    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json")
    max_print_errors: int = 10
    strict: bool = False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def calibrate_dataset(cfg: CalibrationConfig) -> dict[str, Any]:
    """Validate every row in *dataset_file* and return a report dict."""
    path = Path(cfg.dataset_file)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    total = len(data)
    valid_count = 0
    errors: list[dict[str, Any]] = []

    for idx, row in enumerate(data):
        instruction = row.get("instruction")
        output_str = row.get("output")

        # Check fields exist
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append({"index": idx, "error": "missing or empty instruction"})
            continue
        if not isinstance(output_str, str) or not output_str.strip():
            errors.append({"index": idx, "error": "missing or empty output"})
            continue

        # Parse JSON
        try:
            payload = json.loads(output_str)
        except json.JSONDecodeError as exc:
            errors.append({"index": idx, "error": f"invalid JSON: {exc}"})
            continue

        # Validate action structure
        try:
            validate_payload(payload)
        except Exception as exc:
            errors.append({"index": idx, "error": f"validation: {exc}"})
            continue

        valid_count += 1

    # Log errors (up to max_print_errors)
    for err in errors[: cfg.max_print_errors]:
        logger.warning("Row %d: %s", err["index"], err["error"])
    if len(errors) > cfg.max_print_errors:
        logger.warning("... and %d more errors", len(errors) - cfg.max_print_errors)

    report = {
        "dataset_file": str(path),
        "total_rows": total,
        "valid_rows": valid_count,
        "invalid_rows": len(errors),
        "valid_ratio": valid_count / total if total else 0.0,
        "errors": errors,
    }

    logger.info(
        "Calibration: %d/%d valid (%.2f%%), %d errors",
        valid_count, total, report["valid_ratio"] * 100, len(errors),
    )

    if cfg.strict and errors:
        raise ValueError(
            f"Strict calibration failed: {len(errors)} invalid rows in {path}"
        )

    return report


# ---------------------------------------------------------------------------
# Entry from merged config
# ---------------------------------------------------------------------------

def calibrate_from_merged_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build :class:`CalibrationConfig` from a merged YAML dict and run."""
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
