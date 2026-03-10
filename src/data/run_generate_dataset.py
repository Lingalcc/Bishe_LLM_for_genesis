from __future__ import annotations

from pathlib import Path

from src.data_core.generate import GenerateDatasetConfig, run_generate_dataset


def run_generate(
    *,
    num_samples: int = 4000,
    seed: int = 42,
    state_context_ratio: float = 0.7,
    out_dir: Path = Path("data_prepare"),
    alpaca_file: str = "genesis_franka_toolcall_alpaca.json",
    sharegpt_file: str = "genesis_franka_toolcall_sharegpt.json",
    stats_file: str = "genesis_franka_toolcall_stats.json",
    action_map_file: Path | None = Path("src/data/configs/action_map.default.json"),
    action_map_json: str | None = None,
) -> dict[str, Path]:
    cfg = GenerateDatasetConfig(
        num_samples=num_samples,
        seed=seed,
        state_context_ratio=state_context_ratio,
        out_dir=out_dir,
        alpaca_file=alpaca_file,
        sharegpt_file=sharegpt_file,
        stats_file=stats_file,
        action_map_file=action_map_file,
        action_map_json=action_map_json,
    )
    return run_generate_dataset(cfg)
