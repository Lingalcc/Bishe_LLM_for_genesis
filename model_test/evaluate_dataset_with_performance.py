#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_test.performance_monitor import monitor_inference_performance


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _canonicalize_output(text: str) -> str:
    payload = text.strip()
    if not payload:
        return ""
    try:
        obj = json.loads(payload)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return _normalize_text(payload)


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_dataset(
    dataset: Iterable[dict[str, Any]],
    engine: Any,
    *,
    report_file: str | Path = "model_test/accuracy_report.json",
) -> dict[str, Any]:
    """
    Evaluate a dataset with exact-match accuracy and system-level performance metrics.

    Required fields in each sample:
      - instruction: str
      - output: str (ground truth)

    Engine contract:
      - engine.generate(prompt: str) -> str
    """
    if not hasattr(engine, "generate"):
        raise TypeError("`engine` must provide a callable `generate(prompt)` method.")
    if not callable(engine.generate):
        raise TypeError("`engine.generate` is not callable.")

    rows = list(dataset)
    total = len(rows)

    exact_match = 0
    latency_values: list[float] = []
    throughput_values: list[float] = []
    peak_vram_values: list[float] = []
    failures = 0
    details: list[dict[str, Any]] = []

    for idx, sample in enumerate(rows):
        instruction = sample.get("instruction", "")
        gt_output = sample.get("output", "")

        if not isinstance(instruction, str) or not instruction.strip():
            failures += 1
            details.append(
                {
                    "index": idx,
                    "error": "invalid or empty `instruction`",
                    "exact_match": False,
                    "latency_sec": 0.0,
                    "throughput_tps": 0.0,
                    "peak_vram_mb": 0.0,
                }
            )
            continue

        if not isinstance(gt_output, str):
            gt_output = json.dumps(gt_output, ensure_ascii=False)

        monitor = monitor_inference_performance(input_text=instruction)
        pred_output = ""
        infer_error: str | None = None

        try:
            with monitor:
                pred_output = engine.generate(instruction)
                if not isinstance(pred_output, str):
                    pred_output = json.dumps(pred_output, ensure_ascii=False)
                monitor.set_output_text(pred_output)
        except Exception as exc:
            failures += 1
            infer_error = f"{type(exc).__name__}: {exc}"

        try:
            metrics = monitor.metrics
        except Exception:
            metrics = {"latency_sec": 0.0, "throughput_tps": 0.0, "peak_vram_mb": 0.0}

        latency_values.append(float(metrics["latency_sec"]))
        throughput_values.append(float(metrics["throughput_tps"]))
        peak_vram_values.append(float(metrics["peak_vram_mb"]))

        is_exact = _canonicalize_output(pred_output) == _canonicalize_output(gt_output)
        if is_exact:
            exact_match += 1

        details.append(
            {
                "index": idx,
                "exact_match": is_exact,
                "error": infer_error,
                "latency_sec": float(metrics["latency_sec"]),
                "throughput_tps": float(metrics["throughput_tps"]),
                "peak_vram_mb": float(metrics["peak_vram_mb"]),
            }
        )

    report = {
        "num_samples_evaluated": total,
        "exact_match": exact_match,
        "exact_match_rate": (exact_match / total) if total else 0.0,
        "avg_latency_sec": _safe_mean(latency_values),
        "avg_throughput_tps": _safe_mean(throughput_values),
        "avg_peak_vram_mb": _safe_mean(peak_vram_values),
        "num_failures": failures,
        "details": details,
    }

    out_path = Path(report_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


class _DummyEngine:
    """Demo-only engine. Replace this with your real engine."""

    def generate(self, prompt: str) -> str:
        return json.dumps({"action": "echo", "prompt": prompt}, ensure_ascii=False)


def _demo() -> None:
    dataset = [
        {
            "instruction": "Open the gripper.",
            "output": json.dumps({"action": "echo", "prompt": "Open the gripper."}, ensure_ascii=False),
        },
        {
            "instruction": "Move to home pose.",
            "output": json.dumps({"action": "echo", "prompt": "Move to home pose."}, ensure_ascii=False),
        },
    ]
    engine = _DummyEngine()
    report = evaluate_dataset(dataset, engine)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()
