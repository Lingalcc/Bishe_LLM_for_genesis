from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from src.protocols.toolcall import validate_payload

_NUMBER_TOKEN_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class _Issue:
    kind: str
    position: int
    message: str
    expected_token: str | None = None
    stage: str = "json_syntax"


class _JsonSyntaxParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)
        self.pos = 0

    def parse(self) -> _Issue | None:
        self._skip_ws()
        if self._eof():
            return _Issue(kind="其他", position=0, message="输出为空。")
        issue = self._parse_value()
        if issue is not None:
            return issue
        self._skip_ws()
        if not self._eof():
            return _Issue(kind="其他", position=self.pos, message="JSON 末尾存在多余内容。")
        return None

    def _parse_value(self) -> _Issue | None:
        self._skip_ws()
        if self._eof():
            return _Issue(kind="value 类型错", position=self.pos, message="值缺失。")

        ch = self.text[self.pos]
        if ch == "{":
            return self._parse_object()
        if ch == "[":
            return self._parse_array()
        if ch == '"':
            return self._parse_string()
        if ch in "-0123456789":
            return self._parse_number()
        if self.text.startswith("true", self.pos):
            self.pos += 4
            return None
        if self.text.startswith("false", self.pos):
            self.pos += 5
            return None
        if self.text.startswith("null", self.pos):
            self.pos += 4
            return None
        return _Issue(kind="value 类型错", position=self.pos, message="遇到非法 JSON value。")

    def _parse_object(self) -> _Issue | None:
        self.pos += 1
        self._skip_ws()
        if self._eof():
            return _Issue(kind="缺右括号", position=self.pos, message="对象缺少右花括号。", expected_token="}")
        if self.text[self.pos] == "}":
            self.pos += 1
            return None

        while True:
            self._skip_ws()
            if self._eof():
                return _Issue(kind="缺右括号", position=self.pos, message="对象缺少右花括号。", expected_token="}")
            if self.text[self.pos] != '"':
                if self.text[self.pos].isalpha() or self.text[self.pos] == "_":
                    return _Issue(kind="非法键名", position=self.pos, message="对象键名必须使用双引号。")
                return _Issue(kind="非法键名", position=self.pos, message="对象中出现非法键名。")

            issue = self._parse_string()
            if issue is not None:
                return issue

            self._skip_ws()
            if self._eof():
                return _Issue(kind="其他", position=self.pos, message="对象键后缺少冒号。", expected_token=":")
            if self.text[self.pos] != ":":
                return _Issue(kind="其他", position=self.pos, message="对象键后缺少冒号。", expected_token=":")
            self.pos += 1

            issue = self._parse_value()
            if issue is not None:
                return issue

            self._skip_ws()
            if self._eof():
                return _Issue(kind="缺右括号", position=self.pos, message="对象缺少右花括号。", expected_token="}")
            current = self.text[self.pos]
            if current == ",":
                self.pos += 1
                self._skip_ws()
                if self._eof():
                    return _Issue(kind="缺右括号", position=self.pos, message="对象缺少右花括号。", expected_token="}")
                if self.text[self.pos] == "}":
                    return _Issue(kind="其他", position=self.pos, message="对象尾部存在多余逗号。")
                continue
            if current == "}":
                self.pos += 1
                return None
            if current in '"{[-0123456789' or current.isalpha():
                return _Issue(kind="少逗号", position=self.pos, message="对象成员之间缺少逗号。", expected_token=",")
            return _Issue(kind="其他", position=self.pos, message="对象中出现非法结构。")

    def _parse_array(self) -> _Issue | None:
        self.pos += 1
        self._skip_ws()
        if self._eof():
            return _Issue(kind="缺右中括号", position=self.pos, message="数组缺少右中括号。", expected_token="]")
        if self.text[self.pos] == "]":
            self.pos += 1
            return None

        while True:
            issue = self._parse_value()
            if issue is not None:
                return issue

            self._skip_ws()
            if self._eof():
                return _Issue(kind="缺右中括号", position=self.pos, message="数组缺少右中括号。", expected_token="]")
            current = self.text[self.pos]
            if current == ",":
                self.pos += 1
                self._skip_ws()
                if self._eof():
                    return _Issue(kind="缺右中括号", position=self.pos, message="数组缺少右中括号。", expected_token="]")
                if self.text[self.pos] == "]":
                    return _Issue(kind="其他", position=self.pos, message="数组尾部存在多余逗号。")
                continue
            if current == "]":
                self.pos += 1
                return None
            if current in '"{[-0123456789' or current.isalpha():
                return _Issue(kind="少逗号", position=self.pos, message="数组元素之间缺少逗号。", expected_token=",")
            return _Issue(kind="其他", position=self.pos, message="数组中出现非法结构。")

    def _parse_string(self) -> _Issue | None:
        self.pos += 1
        escaped = False
        while not self._eof():
            ch = self.text[self.pos]
            if escaped:
                escaped = False
                self.pos += 1
                continue
            if ch == "\\":
                escaped = True
                self.pos += 1
                continue
            if ch == '"':
                self.pos += 1
                return None
            self.pos += 1
        return _Issue(kind="引号未闭合", position=self.pos, message="字符串缺少闭合双引号。", expected_token='"')

    def _parse_number(self) -> _Issue | None:
        match = re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", self.text[self.pos :])
        if match is None:
            return _Issue(kind="value 类型错", position=self.pos, message="数字 value 非法。")
        self.pos += len(match.group(0))
        return None

    def _skip_ws(self) -> None:
        while not self._eof() and self.text[self.pos].isspace():
            self.pos += 1

    def _eof(self) -> bool:
        return self.pos >= self.length


def _line_and_column(text: str, position: int) -> tuple[int, int]:
    safe_pos = max(0, min(position, len(text)))
    line = text.count("\n", 0, safe_pos) + 1
    last_newline = text.rfind("\n", 0, safe_pos)
    column = safe_pos + 1 if last_newline < 0 else safe_pos - last_newline
    return line, column


def _context_excerpt(text: str, position: int, *, radius: int = 24) -> str:
    if not text:
        return ""
    safe_pos = max(0, min(position, len(text)))
    start = max(0, safe_pos - radius)
    end = min(len(text), safe_pos + radius)
    return text[start:end]


def _sort_histogram_keys(item: tuple[str, int]) -> tuple[int, int | str]:
    key = item[0]
    if key == "unknown":
        return (1, key)
    try:
        return (0, int(key))
    except ValueError:
        return (0, key)


def _to_python_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    return list(value)


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _build_trace_token_spans(text: str, *, trace: dict[str, Any] | None, tokenizer: Any | None) -> list[dict[str, Any]]:
    if tokenizer is None or not isinstance(trace, dict):
        return []
    raw_token_traces = trace.get("token_traces")
    if not isinstance(raw_token_traces, list) or not raw_token_traces:
        return []

    token_items: list[dict[str, Any]] = []
    token_ids: list[int] = []
    for index, item in enumerate(raw_token_traces):
        if not isinstance(item, dict) or "token_id" not in item:
            return []
        token_id = int(item["token_id"])
        token_ids.append(token_id)
        token_items.append(
            {
                "token_index": index,
                "token_id": token_id,
                "exit_layer": item.get("exit_layer"),
            }
        )

    raw_text = _decode_token_ids(tokenizer, token_ids)
    if raw_text == text:
        visible_start = 0
        visible_end = len(raw_text)
    elif raw_text.strip() == text:
        visible_start = len(raw_text) - len(raw_text.lstrip())
        visible_end = visible_start + len(text)
    else:
        return []

    spans: list[dict[str, Any]] = []
    prefix_text = ""
    for index, token_info in enumerate(token_items):
        next_text = _decode_token_ids(tokenizer, token_ids[: index + 1])
        if not next_text.startswith(prefix_text):
            return []
        raw_start = len(prefix_text)
        raw_end = len(next_text)
        clipped_start = max(raw_start, visible_start)
        clipped_end = min(raw_end, visible_end)
        start = clipped_start - visible_start
        end = clipped_end - visible_start
        spans.append(
            {
                **token_info,
                "start": start,
                "end": end,
                "text": text[start:end] if end > start else "",
                "source": "trace_decode",
            }
        )
        prefix_text = next_text
    return spans


def _build_offset_token_spans(
    text: str,
    *,
    tokenize_with_offsets: Callable[[str], Any] | None,
    tokenizer: Any | None,
) -> list[dict[str, Any]]:
    raw_tokens: Any = None
    if callable(tokenize_with_offsets):
        try:
            raw_tokens = tokenize_with_offsets(text)
        except Exception:
            raw_tokens = None

    if isinstance(raw_tokens, list) and raw_tokens:
        spans: list[dict[str, Any]] = []
        for index, item in enumerate(raw_tokens):
            if not isinstance(item, dict):
                continue
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", start) or start)
            token_text = item.get("text")
            if not isinstance(token_text, str):
                token_text = text[start:end] if end > start else ""
            spans.append(
                {
                    "token_index": int(item.get("token_index", index) or index),
                    "token_id": item.get("token_id"),
                    "exit_layer": item.get("exit_layer"),
                    "start": start,
                    "end": end,
                    "text": token_text,
                    "source": "engine_offsets",
                }
            )
        if spans:
            return spans

    if tokenizer is None or not hasattr(tokenizer, "__call__"):
        return []

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except TypeError:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception:
            return []
    except Exception:
        return []

    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
        offset_mapping = encoded.get("offset_mapping")
    else:
        input_ids = getattr(encoded, "input_ids", None)
        offset_mapping = getattr(encoded, "offset_mapping", None)

    token_ids = _to_python_list(input_ids)
    offsets = _to_python_list(offset_mapping)
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    if not token_ids or not offsets or len(token_ids) != len(offsets):
        return []

    spans = []
    for index, (token_id, offset) in enumerate(zip(token_ids, offsets)):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        start = int(offset[0])
        end = int(offset[1])
        spans.append(
            {
                "token_index": index,
                "token_id": int(token_id),
                "exit_layer": None,
                "start": start,
                "end": end,
                "text": text[start:end] if end > start else "",
                "source": "tokenizer_offsets",
            }
        )
    return spans


def _classify_token_category(token_text: str) -> str:
    normalized = token_text.strip()
    if not normalized:
        return "未知 token"
    if normalized in {"{", "}", "[", "]", ":", ",", '"'}:
        return "结构 token"
    if all(ch in "{}[]:,\"" for ch in normalized):
        return "结构 token"
    if _NUMBER_TOKEN_RE.match(normalized):
        return "数值 token"
    return "动作 token"


def _find_anchor_token(position: int, token_spans: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    if not token_spans:
        return None, "unavailable"

    for token in token_spans:
        start = int(token.get("start", 0) or 0)
        end = int(token.get("end", start) or start)
        if start <= position < end:
            return token, "generated_token"

    previous_visible = None
    next_visible = None
    for token in token_spans:
        start = int(token.get("start", 0) or 0)
        end = int(token.get("end", start) or start)
        if end <= position and end > start:
            previous_visible = token
        if next_visible is None and start >= position and end > start:
            next_visible = token

    if previous_visible is not None:
        return previous_visible, "anchor_previous_token"
    if next_visible is not None:
        return next_visible, "anchor_next_token"
    return token_spans[-1], "anchor_last_token"


def _first_non_ws(text: str, start: int) -> int:
    position = max(0, start)
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _scan_string_content_span(text: str, start_quote: int) -> tuple[int, int]:
    if start_quote >= len(text) or text[start_quote] != '"':
        return start_quote, min(len(text), start_quote + 1)
    cursor = start_quote + 1
    escaped = False
    while cursor < len(text):
        ch = text[cursor]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return start_quote + 1, cursor
        cursor += 1
    return start_quote + 1, len(text)


def _find_key_occurrence(text: str, key: str) -> tuple[int, int] | None:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:')
    match = pattern.search(text)
    if match is None:
        return None
    key_start = match.start() + 1
    key_end = key_start + len(key)
    return key_start, key_end


def _find_value_span_for_key(text: str, key: str) -> tuple[int, int] | None:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:')
    match = pattern.search(text)
    if match is None:
        return None
    value_start = _first_non_ws(text, match.end())
    if value_start >= len(text):
        return len(text), len(text)
    ch = text[value_start]
    if ch == '"':
        return _scan_string_content_span(text, value_start)
    if ch in "-0123456789":
        number_match = re.match(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", text[value_start:])
        if number_match is not None:
            return value_start, value_start + len(number_match.group(0))
    if ch.isalpha():
        literal_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[value_start:])
        if literal_match is not None:
            return value_start, value_start + len(literal_match.group(0))
    return value_start, min(len(text), value_start + 1)


def _find_action_value_span(text: str, action_name: str | None = None) -> tuple[int, int] | None:
    pattern = re.compile(r'"action"\s*:\s*"([^"]*)"')
    for match in pattern.finditer(text):
        value = match.group(1)
        if action_name is not None and value != action_name:
            continue
        value_start = match.start(1)
        value_end = match.end(1)
        return value_start, value_end
    return None


def _issue_from_schema_error(text: str, error_message: str) -> _Issue:
    unsupported_action = re.search(r"unsupported action:\s*([A-Za-z0-9_]+)", error_message)
    if unsupported_action is not None:
        action_name = unsupported_action.group(1)
        span = _find_action_value_span(text, action_name=action_name)
        position = span[0] if span is not None else 0
        return _Issue(
            kind="非法动作名",
            position=position,
            message=error_message,
            expected_token=action_name,
            stage="schema_validation",
        )

    missing_action = re.search(r"missing valid 'action'", error_message)
    if missing_action is not None:
        span = _find_key_occurrence(text, "action")
        position = span[0] if span is not None else 0
        return _Issue(
            kind="非法键名",
            position=position,
            message=error_message,
            expected_token="action",
            stage="schema_validation",
        )

    missing_field = re.search(r"([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) is required", error_message)
    if missing_field is not None:
        action_name, field_name = missing_field.group(1), missing_field.group(2)
        span = _find_action_value_span(text, action_name=action_name)
        position = span[0] if span is not None else len(text)
        return _Issue(
            kind="非法键名",
            position=position,
            message=error_message,
            expected_token=field_name,
            stage="schema_validation",
        )

    unsupported_key = re.search(r"`([A-Za-z0-9_]+)` is not supported", error_message)
    if unsupported_key is not None:
        field_name = unsupported_key.group(1)
        span = _find_key_occurrence(text, field_name)
        position = span[0] if span is not None else 0
        return _Issue(
            kind="非法键名",
            position=position,
            message=error_message,
            expected_token=field_name,
            stage="schema_validation",
        )

    value_field = re.search(r"([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) (?:must be|length must match)", error_message)
    if value_field is not None:
        _, field_name = value_field.group(1), value_field.group(2)
        span = _find_value_span_for_key(text, field_name)
        position = span[0] if span is not None else 0
        return _Issue(
            kind="value 类型错",
            position=position,
            message=error_message,
            stage="schema_validation",
        )

    if "steps must be int" in error_message:
        span = _find_value_span_for_key(text, "steps")
        position = span[0] if span is not None else 0
        return _Issue(kind="value 类型错", position=position, message=error_message, stage="schema_validation")

    return _Issue(kind="其他", position=0, message=error_message, stage="schema_validation")


def _format_issue(
    *,
    text: str,
    issue: _Issue,
    token_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    line, column = _line_and_column(text, issue.position)
    anchor_token, exit_layer_source = _find_anchor_token(issue.position, token_spans)

    use_expected_token = (
        issue.kind in {"缺右括号", "缺右中括号", "引号未闭合"}
        or (issue.stage == "schema_validation" and issue.expected_token is not None)
    )
    failing_token_text = issue.expected_token if use_expected_token else None
    if failing_token_text is None and anchor_token is not None:
        token_start = int(anchor_token.get("start", 0) or 0)
        token_end = int(anchor_token.get("end", token_start) or token_start)
        failing_token_text = text[token_start:token_end]
    if failing_token_text is None:
        failing_token_text = ""

    token_category = _classify_token_category(failing_token_text)
    anchor_token_text = ""
    token_index: int | None = None
    exit_layer: int | None = None
    if anchor_token is not None:
        anchor_token_text = str(anchor_token.get("text", ""))
        raw_index = anchor_token.get("token_index")
        token_index = int(raw_index) if raw_index is not None else None
        raw_exit_layer = anchor_token.get("exit_layer")
        exit_layer = int(raw_exit_layer) if raw_exit_layer is not None else None

    return {
        "failure_stage": issue.stage,
        "first_error_kind": issue.kind,
        "first_error_message": issue.message,
        "char_position": int(issue.position),
        "line": int(line),
        "column": int(column),
        "token_text": failing_token_text,
        "token_category": token_category,
        "expected_token": issue.expected_token,
        "anchor_token_text": anchor_token_text,
        "anchor_token_index": token_index,
        "exit_layer": exit_layer,
        "exit_layer_source": exit_layer_source,
        "context_excerpt": _context_excerpt(text, issue.position),
    }


def diagnose_parse_failure(
    prediction_text: str,
    *,
    error_message: str | None = None,
    trace: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    tokenize_with_offsets: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    text = str(prediction_text or "")
    token_spans = _build_trace_token_spans(text, trace=trace, tokenizer=tokenizer)
    if not token_spans:
        token_spans = _build_offset_token_spans(
            text,
            tokenize_with_offsets=tokenize_with_offsets,
            tokenizer=tokenizer,
        )

    syntax_issue = _JsonSyntaxParser(text).parse()
    if syntax_issue is not None:
        return _format_issue(text=text, issue=syntax_issue, token_spans=token_spans)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        issue = _Issue(kind="其他", position=int(exc.pos), message=str(exc))
        return _format_issue(text=text, issue=issue, token_spans=token_spans)

    validation_error = error_message
    if not validation_error:
        try:
            validate_payload(payload, policy="evaluation")
        except Exception as exc:
            validation_error = str(exc)

    if validation_error:
        schema_issue = _issue_from_schema_error(text, validation_error)
        return _format_issue(text=text, issue=schema_issue, token_spans=token_spans)

    return {
        "failure_stage": "unknown",
        "first_error_kind": "其他",
        "first_error_message": "未能复现 parse failure，但原评测阶段判定为失败。",
        "char_position": 0,
        "line": 1,
        "column": 1,
        "token_text": "",
        "token_category": "未知 token",
        "expected_token": None,
        "anchor_token_text": "",
        "anchor_token_index": None,
        "exit_layer": None,
        "exit_layer_source": "unavailable",
        "context_excerpt": _context_excerpt(text, 0),
    }


def summarize_parse_failures(
    details: list[dict[str, Any]],
    *,
    diagnostic_key: str = "parse_failure_diagnostic",
) -> dict[str, Any] | None:
    failure_details: list[dict[str, Any]] = []
    kind_hist = Counter()
    token_category_hist = Counter()
    exit_layer_hist = Counter()
    kind_by_token_category: dict[str, Counter[str]] = defaultdict(Counter)
    kind_by_exit_layer: dict[str, Counter[str]] = defaultdict(Counter)
    token_category_by_exit_layer: dict[str, Counter[str]] = defaultdict(Counter)

    for detail in details:
        diagnostic = detail.get(diagnostic_key)
        if not isinstance(diagnostic, dict):
            continue
        failure_details.append(
            {
                "dataset_index": detail.get("dataset_index"),
                "first_error_kind": diagnostic.get("first_error_kind"),
                "token_category": diagnostic.get("token_category"),
                "exit_layer": diagnostic.get("exit_layer"),
                "char_position": diagnostic.get("char_position"),
                "token_text": diagnostic.get("token_text"),
                "context_excerpt": diagnostic.get("context_excerpt"),
                "exit_layer_source": diagnostic.get("exit_layer_source"),
            }
        )
        kind = str(diagnostic.get("first_error_kind", "其他"))
        token_category = str(diagnostic.get("token_category", "未知 token"))
        exit_layer_key = (
            str(int(diagnostic.get("exit_layer")))
            if diagnostic.get("exit_layer") is not None
            else "unknown"
        )
        kind_hist[kind] += 1
        token_category_hist[token_category] += 1
        exit_layer_hist[exit_layer_key] += 1
        kind_by_token_category[kind][token_category] += 1
        kind_by_exit_layer[kind][exit_layer_key] += 1
        token_category_by_exit_layer[token_category][exit_layer_key] += 1

    if not failure_details:
        return None

    return {
        "failure_count": len(failure_details),
        "first_error_kind_histogram": {key: int(value) for key, value in sorted(kind_hist.items())},
        "first_error_token_category_histogram": {
            key: int(value) for key, value in sorted(token_category_hist.items())
        },
        "first_error_exit_layer_histogram": {
            key: int(value)
            for key, value in sorted(
                exit_layer_hist.items(),
                key=_sort_histogram_keys,
            )
        },
        "first_error_kind_by_token_category": {
            kind: {token_category: int(count) for token_category, count in sorted(counter.items())}
            for kind, counter in sorted(kind_by_token_category.items())
        },
        "first_error_kind_by_exit_layer": {
            kind: {layer: int(count) for layer, count in sorted(counter.items(), key=_sort_histogram_keys)}
            for kind, counter in sorted(kind_by_exit_layer.items())
        },
        "first_error_token_category_by_exit_layer": {
            token_category: {layer: int(count) for layer, count in sorted(counter.items(), key=_sort_histogram_keys)}
            for token_category, counter in sorted(token_category_by_exit_layer.items())
        },
        "samples": failure_details,
    }
