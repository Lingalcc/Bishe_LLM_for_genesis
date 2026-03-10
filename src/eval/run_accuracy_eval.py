from __future__ import annotations

from pathlib import Path

from src.eval_core.accuracy import AccuracyEvalConfig, run_accuracy_eval as core_run_accuracy_eval


def run_accuracy_eval(
    *,
    dataset_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json"),
    predictions_file: Path | None = None,
    report_file: Path = Path("src/eval/accuracy_report.json"),
    num_samples: int = 200,
    seed: int = 42,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-5",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.0,
    max_tokens: int = 1200,
    timeout: int = 120,
    max_retries: int = 3,
    sleep_seconds: float = 0.0,
) -> dict:
    cfg = AccuracyEvalConfig(
        dataset_file=dataset_file,
        predictions_file=predictions_file,
        report_file=report_file,
        num_samples=num_samples,
        seed=seed,
        api_base=api_base,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    return core_run_accuracy_eval(cfg)


run_accuracy = run_accuracy_eval
