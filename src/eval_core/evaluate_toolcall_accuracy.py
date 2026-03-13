#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.eval_core.toolcall_validator import validate_payload
from src.utils.secrets import MissingSecretError, redact_text, resolve_api_key_from_env, safe_json_dumps


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def extract_first_json_from_text(text: str) -> Any:
    payload = text.strip()
    if not payload:
        raise ValueError("empty response")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", payload, flags=re.IGNORECASE)
    if code_match:
        inner = code_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(payload):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(payload[idx:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("no valid JSON found")


def payload_to_commands(payload_like: Any) -> list[dict[str, Any]]:
    payload_obj = payload_like
    if isinstance(payload_like, str):
        payload_obj = extract_first_json_from_text(payload_like)
    commands = validate_payload(payload_obj)
    return commands


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        rounded = round(value, 8)
        if abs(rounded - round(rounded)) < 1e-8:
            return int(round(rounded))
        return rounded
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(value[k]) for k in sorted(value.keys())}
    return str(value)


def canonicalize_commands(commands: list[dict[str, Any]]) -> str:
    normalized = normalize_value(commands)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def call_chat_completions(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    obj = json.loads(raw)
    choices = obj.get("choices", [])
    if not choices:
        raise ValueError("empty choices from API")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content from API")
    return content


def predict_once(
    *,
    api_base: str,
    api_key: str,
    model: str,
    instruction: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            return call_chat_completions(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except (ValueError, json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError) as err:
            last_err = err
            time.sleep(min(8.0, 0.8 * (2**i)))

    if last_err is None:
        raise RuntimeError("prediction failed without explicit error")
    raise RuntimeError(f"prediction failed after retries: {redact_text(str(last_err))}")


def load_predictions_file(predictions_path: Path) -> Any:
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")
    return json.loads(predictions_path.read_text(encoding="utf-8"))


def evaluate_toolcall_accuracy(
    *,
    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json"),
    predictions_file: Path | None = None,
    report_file: Path = Path("experiments/03_eval_exp/reports/accuracy_report.json"),
    num_samples: int = 200,
    seed: int = 42,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-5",
    api_key_env: str = "OPENAI_API_KEY",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout: int = 120,
    max_retries: int = 3,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    if not dataset_file.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_file}")

    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must be a non-empty list")

    valid_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        instruction = row.get("instruction")
        output_text = row.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        if not isinstance(output_text, str) or not output_text.strip():
            continue
        try:
            gt_commands = payload_to_commands(output_text)
        except Exception:
            continue
        valid_rows.append(
            {
                "dataset_index": i,
                "instruction": instruction,
                "system": row.get("system", "") if isinstance(row.get("system", ""), str) else "",
                "gt_output": output_text,
                "gt_commands": gt_commands,
            }
        )

    if not valid_rows:
        raise RuntimeError("no valid rows found in dataset")

    rng = random.Random(seed)
    selected_count = min(num_samples, len(valid_rows))
    selected_rows = rng.sample(valid_rows, k=selected_count)

    predictions_blob: Any = None
    if predictions_file is not None:
        predictions_blob = load_predictions_file(predictions_file)

    if predictions_file is None:
        try:
            api_key = resolve_api_key_from_env(
                api_key=api_key,
                api_key_env=api_key_env,
                default_env="OPENAI_API_KEY",
                source_name="Accuracy evaluation API",
            )
        except MissingSecretError as exc:
            raise RuntimeError(
                f"{exc} Or provide `predictions_file` to run offline evaluation."
            ) from exc
    else:
        api_key = ""

    total = len(selected_rows)
    parse_ok = 0
    exact_match = 0
    action_match = 0
    online_call_fail = 0

    details: list[dict[str, Any]] = []

    for i, sample in enumerate(selected_rows):
        prediction_text = ""
        online_error = None

        if predictions_blob is not None:
            if isinstance(predictions_blob, list):
                if i >= len(predictions_blob):
                    online_error = "predictions list length < selected samples"
                else:
                    item = predictions_blob[i]
                    prediction_text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            elif isinstance(predictions_blob, dict):
                key = str(sample["dataset_index"])
                if key not in predictions_blob:
                    online_error = f"predictions dict missing dataset index key: {key}"
                else:
                    item = predictions_blob[key]
                    prediction_text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            else:
                raise TypeError("predictions-file must be JSON list or dict")
        else:
            try:
                prediction_text = predict_once(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    instruction=sample["instruction"],
                    system_prompt=sample["system"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            except Exception as err:
                online_error = str(err)
                online_call_fail += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        result: dict[str, Any] = {
            "dataset_index": sample["dataset_index"],
            "instruction": sample["instruction"],
            "online_error": online_error,
            "prediction_preview": normalize_text(prediction_text)[:200],
            "parse_ok": False,
            "exact_match": False,
            "action_match": False,
            "error": None,
        }

        if online_error is None:
            try:
                pred_commands = payload_to_commands(prediction_text)
                gt_commands = sample["gt_commands"]

                pred_sig = [str(cmd.get("action", "")) for cmd in pred_commands]
                gt_sig = [str(cmd.get("action", "")) for cmd in gt_commands]

                pred_canonical = canonicalize_commands(pred_commands)
                gt_canonical = canonicalize_commands(gt_commands)

                result["parse_ok"] = True
                result["exact_match"] = pred_canonical == gt_canonical
                result["action_match"] = pred_sig == gt_sig

                parse_ok += 1
                if result["exact_match"]:
                    exact_match += 1
                if result["action_match"]:
                    action_match += 1
            except Exception as err:
                result["error"] = str(err)

        details.append(result)

    report = {
        "dataset_file": str(dataset_file),
        "predictions_file": str(predictions_file) if predictions_file is not None else None,
        "model": model if predictions_file is None else None,
        "api_base": api_base if predictions_file is None else None,
        "seed": seed,
        "num_samples_requested": num_samples,
        "num_samples_evaluated": total,
        "num_valid_rows_in_dataset": len(valid_rows),
        "online_call_failures": online_call_fail,
        "parse_ok": parse_ok,
        "parse_ok_rate": (parse_ok / total) if total else 0.0,
        "exact_match": exact_match,
        "exact_match_rate": (exact_match / total) if total else 0.0,
        "exact_match_rate_on_parse_ok": (exact_match / parse_ok) if parse_ok else 0.0,
        "action_match": action_match,
        "action_match_rate": (action_match / total) if total else 0.0,
        "details": details,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(safe_json_dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
