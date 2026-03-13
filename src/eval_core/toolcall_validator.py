#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from src.protocols.toolcall import (
    ValidationPolicy,
    is_num as protocol_is_num,
    validate_payload as protocol_validate_payload,
)


def is_num(x: Any) -> bool:
    return protocol_is_num(x)


def validate_command(cmd: dict[str, Any], policy: ValidationPolicy | str | None = None) -> None:
    protocol_validate_payload({"commands": [cmd]}, policy=policy)


def validate_payload(
    payload: Any,
    policy: ValidationPolicy | str | None = None,
) -> list[dict[str, Any]]:
    return protocol_validate_payload(payload, policy=policy)
