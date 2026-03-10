from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.generate_genesis_franka_dataset import generate_and_save_dataset


@dataclass(frozen=True)
class GenerateDatasetConfig:
    num_samples: int = 4000
    seed: int = 42
    state_context_ratio: float = 0.7
    out_dir: Path = Path("data_prepare")
    alpaca_file: str = "genesis_franka_toolcall_alpaca.json"
    sharegpt_file: str = "genesis_franka_toolcall_sharegpt.json"
    stats_file: str = "genesis_franka_toolcall_stats.json"
    action_map_file: Path | None = Path("src/data/configs/action_map.default.json")
    action_map_json: str | None = None


def run_generate_dataset(cfg: GenerateDatasetConfig) -> dict[str, Path]:
    if cfg.num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if not 0.0 <= cfg.state_context_ratio <= 1.0:
        raise ValueError("state_context_ratio must be in [0.0, 1.0]")

    result = generate_and_save_dataset(
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        state_context_ratio=cfg.state_context_ratio,
        out_dir=cfg.out_dir,
        alpaca_file=cfg.alpaca_file,
        sharegpt_file=cfg.sharegpt_file,
        stats_file=cfg.stats_file,
        action_map_file=cfg.action_map_file,
        action_map_json=cfg.action_map_json,
    )
    return {
        "alpaca_path": Path(result["alpaca_path"]),
        "sharegpt_path": Path(result["sharegpt_path"]),
        "stats_path": Path(result["stats_path"]),
    }


def run_generate_from_merged_config(config: dict[str, Any]) -> dict[str, Path]:
    section = (
        config.get("dataset_prepare", {}).get("generate", {})
        if isinstance(config.get("dataset_prepare"), dict)
        else {}
    )
    cfg = GenerateDatasetConfig(
        num_samples=int(section.get("num_samples", 4000)),
        seed=int(section.get("seed", 42)),
        state_context_ratio=float(section.get("state_context_ratio", 0.7)),
        out_dir=Path(section.get("out_dir", "data_prepare")),
        alpaca_file=str(section.get("alpaca_file", "genesis_franka_toolcall_alpaca.json")),
        sharegpt_file=str(section.get("sharegpt_file", "genesis_franka_toolcall_sharegpt.json")),
        stats_file=str(section.get("stats_file", "genesis_franka_toolcall_stats.json")),
        action_map_file=Path(section["action_map_file"]) if section.get("action_map_file") else None,
        action_map_json=section.get("action_map_json"),
    )
    return run_generate_dataset(cfg)
