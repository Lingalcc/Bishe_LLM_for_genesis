from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.evaluate_toolcall_accuracy import evaluate_toolcall_accuracy


@dataclass(frozen=True)
class AccuracyEvalConfig:
    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json")
    predictions_file: Path | None = None
    report_file: Path = Path("src/eval/accuracy_report.json")
    num_samples: int = 200
    seed: int = 42
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-5"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout: int = 120
    max_retries: int = 3
    sleep_seconds: float = 0.0


def run_accuracy_eval(cfg: AccuracyEvalConfig) -> dict[str, Any]:
    return evaluate_toolcall_accuracy(
        dataset_file=cfg.dataset_file,
        predictions_file=cfg.predictions_file,
        report_file=cfg.report_file,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        api_base=cfg.api_base,
        model=cfg.model,
        api_key=cfg.api_key,
        api_key_env=cfg.api_key_env,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        sleep_seconds=cfg.sleep_seconds,
    )


def run_accuracy_from_merged_config(config: dict[str, Any]) -> dict[str, Any]:
    section = (
        config.get("test", {}).get("accuracy_eval", {})
        if isinstance(config.get("test"), dict)
        else {}
    )
    cfg = AccuracyEvalConfig(
        dataset_file=Path(section.get("dataset_file", AccuracyEvalConfig.dataset_file)),
        predictions_file=Path(section["predictions_file"]) if section.get("predictions_file") else None,
        report_file=Path(section.get("report_file", AccuracyEvalConfig.report_file)),
        num_samples=int(section.get("num_samples", AccuracyEvalConfig.num_samples)),
        seed=int(section.get("seed", AccuracyEvalConfig.seed)),
        api_base=str(section.get("api_base", AccuracyEvalConfig.api_base)),
        model=str(section.get("model", AccuracyEvalConfig.model)),
        api_key=str(section.get("api_key", AccuracyEvalConfig.api_key)),
        api_key_env=str(section.get("api_key_env", AccuracyEvalConfig.api_key_env)),
        temperature=float(section.get("temperature", AccuracyEvalConfig.temperature)),
        max_tokens=int(section.get("max_tokens", AccuracyEvalConfig.max_tokens)),
        timeout=int(section.get("timeout", AccuracyEvalConfig.timeout)),
        max_retries=int(section.get("max_retries", AccuracyEvalConfig.max_retries)),
        sleep_seconds=float(section.get("sleep_seconds", AccuracyEvalConfig.sleep_seconds)),
    )
    return run_accuracy_eval(cfg)
