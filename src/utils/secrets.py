from __future__ import annotations

import json
import os
import re
from typing import Any


class MissingSecretError(RuntimeError):
    """Raised when a required secret is not available from environment variables."""


_DEFAULT_MASK = "[REDACTED]"
_PLACEHOLDER_VALUES = {
    "",
    "your_api_key",
    "your-openai-api-key",
    "your-deepseek-api-key",
    "<your_api_key>",
    "<your-openai-api-key>",
    "<your-deepseek-api-key>",
    "replace_me",
    _DEFAULT_MASK.lower(),
}
_SECRET_NAME_TOKENS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
)


def _normalize_secret_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_placeholder_secret(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    lower = text.lower()
    if lower in _PLACEHOLDER_VALUES:
        return True
    if re.fullmatch(r"\$\{[A-Z0-9_]+\}", text):
        return True
    return False


def _extract_env_from_placeholder(value: str) -> str | None:
    matched = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", value.strip())
    if not matched:
        return None
    return matched.group(1)


def resolve_api_key_from_env(
    *,
    api_key: str = "",
    api_key_env: str = "",
    default_env: str = "OPENAI_API_KEY",
    source_name: str = "API",
) -> str:
    """Resolve API key strictly from environment variables.

    `api_key` is preserved for backward-compatible signatures, but is no longer
    used as a source of truth for security reasons.
    """
    explicit = _normalize_secret_text(api_key)
    inferred_env = _extract_env_from_placeholder(explicit) if explicit else None
    env_name = _normalize_secret_text(api_key_env) or inferred_env or default_env
    key = _normalize_secret_text(os.environ.get(env_name, ""))

    if _is_placeholder_secret(key):
        raise MissingSecretError(
            f"{source_name} key is missing. Set environment variable `{env_name}`. "
            "Config field `api_key` is not used for real secrets."
        )
    return key


def _is_sensitive_field_name(field_name: Any) -> bool:
    text = str(field_name).strip().lower()
    if not text:
        return False
    if text.endswith("_env") or text.endswith("env"):
        return False
    return any(token in text for token in _SECRET_NAME_TOKENS)


def redact_text(text: str) -> str:
    """Redact common secret patterns from free text."""
    redacted = re.sub(
        r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]+",
        r"\1 " + _DEFAULT_MASK,
        text,
    )
    redacted = re.sub(
        r"\bsk-[A-Za-z0-9_\-]{6,}\b",
        "sk-" + _DEFAULT_MASK,
        redacted,
    )
    return redacted


def redact_secrets(obj: Any) -> Any:
    """Deep-redact sensitive fields in mapping/list structures."""
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            if _is_sensitive_field_name(key):
                if str(key).strip().lower() == "authorization":
                    out[key] = f"Bearer {_DEFAULT_MASK}"
                else:
                    out[key] = _DEFAULT_MASK
            else:
                out[key] = redact_secrets(value)
        return out

    if isinstance(obj, list):
        return [redact_secrets(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(redact_secrets(item) for item in obj)

    if isinstance(obj, str):
        return redact_text(obj)

    return obj


def safe_json_dumps(data: Any, *, ensure_ascii: bool = False, indent: int | None = None, sort_keys: bool = False) -> str:
    """Serialize data as JSON after applying secret redaction."""
    return json.dumps(
        redact_secrets(data),
        ensure_ascii=ensure_ascii,
        indent=indent,
        sort_keys=sort_keys,
    )
