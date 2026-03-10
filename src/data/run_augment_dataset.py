from __future__ import annotations

from pathlib import Path

from src.data_core.augment import AugmentDatasetConfig, run_augment_dataset


def run_augment(
    *,
    input_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json"),
    output_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca_augmented.json"),
    stats_file: Path = Path("data_prepare/genesis_franka_toolcall_augment_stats.json"),
    output_sharegpt_file: Path = Path("data_prepare/genesis_franka_toolcall_sharegpt_augmented.json"),
    seed: int = 42,
    num_source: int = 800,
    aug_per_sample: int = 2,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-5",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.9,
    max_tokens: int = 1200,
    timeout: int = 120,
    max_retries: int = 5,
    sleep_seconds: float = 0.2,
) -> dict:
    cfg = AugmentDatasetConfig(
        input_file=input_file,
        output_file=output_file,
        stats_file=stats_file,
        output_sharegpt_file=output_sharegpt_file,
        seed=seed,
        num_source=num_source,
        aug_per_sample=aug_per_sample,
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
    return run_augment_dataset(cfg)
