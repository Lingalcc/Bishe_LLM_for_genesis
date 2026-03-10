from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.augment_genesis_franka_dataset_with_api import augment_dataset_with_api


@dataclass(frozen=True)
class AugmentDatasetConfig:
    input_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca.json")
    output_file: Path = Path("data_prepare/genesis_franka_toolcall_alpaca_augmented.json")
    stats_file: Path = Path("data_prepare/genesis_franka_toolcall_augment_stats.json")
    output_sharegpt_file: Path = Path("data_prepare/genesis_franka_toolcall_sharegpt_augmented.json")
    seed: int = 42
    num_source: int = 800
    aug_per_sample: int = 2
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-5"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.9
    max_tokens: int = 1200
    timeout: int = 120
    max_retries: int = 5
    sleep_seconds: float = 0.2


def run_augment_dataset(cfg: AugmentDatasetConfig) -> dict[str, Any]:
    return augment_dataset_with_api(
        input_file=cfg.input_file,
        output_file=cfg.output_file,
        stats_file=cfg.stats_file,
        output_sharegpt_file=cfg.output_sharegpt_file,
        seed=cfg.seed,
        num_source=cfg.num_source,
        aug_per_sample=cfg.aug_per_sample,
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


def run_augment_from_merged_config(config: dict[str, Any]) -> dict[str, Any]:
    section = (
        config.get("dataset_prepare", {}).get("augment", {})
        if isinstance(config.get("dataset_prepare"), dict)
        else {}
    )
    cfg = AugmentDatasetConfig(
        input_file=Path(section.get("input_file", AugmentDatasetConfig.input_file)),
        output_file=Path(section.get("output_file", AugmentDatasetConfig.output_file)),
        stats_file=Path(section.get("stats_file", AugmentDatasetConfig.stats_file)),
        output_sharegpt_file=Path(
            section.get("output_sharegpt_file", AugmentDatasetConfig.output_sharegpt_file)
        ),
        seed=int(section.get("seed", AugmentDatasetConfig.seed)),
        num_source=int(section.get("num_source", AugmentDatasetConfig.num_source)),
        aug_per_sample=int(section.get("aug_per_sample", AugmentDatasetConfig.aug_per_sample)),
        api_base=str(section.get("api_base", AugmentDatasetConfig.api_base)),
        model=str(section.get("model", AugmentDatasetConfig.model)),
        api_key=str(section.get("api_key", AugmentDatasetConfig.api_key)),
        api_key_env=str(section.get("api_key_env", AugmentDatasetConfig.api_key_env)),
        temperature=float(section.get("temperature", AugmentDatasetConfig.temperature)),
        max_tokens=int(section.get("max_tokens", AugmentDatasetConfig.max_tokens)),
        timeout=int(section.get("timeout", AugmentDatasetConfig.timeout)),
        max_retries=int(section.get("max_retries", AugmentDatasetConfig.max_retries)),
        sleep_seconds=float(section.get("sleep_seconds", AugmentDatasetConfig.sleep_seconds)),
    )
    return run_augment_dataset(cfg)
