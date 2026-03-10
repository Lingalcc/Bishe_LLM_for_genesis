#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.generate_genesis_franka_dataset import validate_payload
from src.data.unified_config import DEFAULT_CONFIG_PATH, load_dataset_prepare_runtime_config


DEFAULT_DATASET_FILE = "data_prepare/genesis_franka_toolcall_alpaca.json"
DEFAULT_MAX_PRINT_ERRORS = 10
DEFAULT_STRICT = False

CALIBRATION_DEFAULTS: dict[str, Any] = {
    "dataset_file": DEFAULT_DATASET_FILE,
    "max_print_errors": DEFAULT_MAX_PRINT_ERRORS,
    "strict": DEFAULT_STRICT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate/validate NL->JSON dataset quality."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Unified config YAML path.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Path to alpaca-format dataset.",
    )
    parser.add_argument(
        "--max-print-errors",
        type=int,
        default=None,
        help="Max invalid row details to print.",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=None,
        help="Exit with non-zero code if invalid rows exist.",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=None,
        help="Always exit with zero even when invalid rows exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    section = load_dataset_prepare_runtime_config(
        config_path=args.config,
        section="calibration",
        defaults=CALIBRATION_DEFAULTS,
    )

    dataset_file = args.dataset_file or Path(
        str(section.get("dataset_file", DEFAULT_DATASET_FILE))
    )
    max_print_errors = args.max_print_errors
    if max_print_errors is None:
        max_print_errors = int(section.get("max_print_errors", DEFAULT_MAX_PRINT_ERRORS))
    strict = args.strict if args.strict is not None else bool(section.get("strict", DEFAULT_STRICT))

    if not dataset_file.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_file}")

    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
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
            if len(invalid_examples) < max_print_errors:
                invalid_examples.append(f"row={idx} reason=row_not_dict")
            continue

        instruction = row.get("instruction")
        output = row.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            missing_instruction += 1
            invalid += 1
            if len(invalid_examples) < max_print_errors:
                invalid_examples.append(f"row={idx} reason=missing_instruction")
            continue
        if not isinstance(output, str) or not output.strip():
            missing_output += 1
            invalid += 1
            if len(invalid_examples) < max_print_errors:
                invalid_examples.append(f"row={idx} reason=missing_output")
            continue

        try:
            payload = json.loads(output)
            validate_payload(payload)
            valid += 1
        except Exception as err:
            invalid += 1
            if len(invalid_examples) < max_print_errors:
                invalid_examples.append(f"row={idx} reason=invalid_json_or_schema err={type(err).__name__}: {err}")

    print("[calibration] config    :", args.config)
    print("[calibration] file      :", dataset_file)
    print("[calibration] total     :", total)
    print("[calibration] valid     :", valid)
    print("[calibration] invalid   :", invalid)
    print("[calibration] miss_ins  :", missing_instruction)
    print("[calibration] miss_out  :", missing_output)
    if total > 0:
        print("[calibration] pass_rate :", f"{(valid / total) * 100:.2f}%")

    if invalid_examples:
        print("[calibration] sample_errors:")
        for line in invalid_examples:
            print("  -", line)

    if strict and invalid > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
