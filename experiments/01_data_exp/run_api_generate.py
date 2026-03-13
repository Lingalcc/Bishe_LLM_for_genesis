#!/usr/bin/env python3
"""API-based dataset generation demo.

Generates natural-language instruction → action-JSON training data for
Franka robot fine-tuning via a high-level LLM API (OpenAI / DeepSeek / etc.).

Usage:
    # Use default api_generate.yaml config:
    python experiments/01_data_exp/run_api_generate.py

    # Custom config override:
    python experiments/01_data_exp/run_api_generate.py \
        --config experiments/01_data_exp/configs/api_generate.yaml

    # Quick demo (10 samples):
    python experiments/01_data_exp/run_api_generate.py --demo

Environment:
    Set your API key via environment variable before running:
        export DEEPSEEK_API_KEY="sk-..."
    Real secrets must come from environment variables.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_core.generate import (
    GenerateDatasetConfig,
    DatasetGenerator,
    run_generate_dataset,
    run_generate_from_merged_config,
    ACTION_SCHEMAS,
    DIFFICULTY_LEVELS,
)
from src.utils.config import load_merged_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_action_summary() -> None:
    """Print all supported robot actions."""
    print("=" * 60)
    print("Franka Panda 机械臂 - 可用基本动作汇总")
    print("=" * 60)
    for i, (name, schema) in enumerate(ACTION_SCHEMAS.items(), 1):
        print(f"  {i:2d}. {name:30s} — {schema['description']}")
        if schema["params"]:
            for pname, pinfo in schema["params"].items():
                opt = " (可选)" if pinfo.get("optional") else ""
                print(f"      参数 {pname}: {pinfo['desc']}{opt}")
        print(f"      示例: {json.dumps(schema['example'], ensure_ascii=False)}")
    print("=" * 60)
    print(f"共 {len(ACTION_SCHEMAS)} 个基本动作\n")


def run_demo(cfg: GenerateDatasetConfig) -> None:
    """Run a small demo generation and print samples to stdout."""
    print("\n🚀 开始 Demo 模式 — 生成少量样本并展示\n")
    print_action_summary()

    demo_cfg = GenerateDatasetConfig(
        out_dir=cfg.out_dir,
        alpaca_file="demo_alpaca.json",
        sharegpt_file="demo_sharegpt.json",
        stats_file="demo_stats.json",
        num_samples=10,
        seed=cfg.seed,
        batch_size=10,
        state_context_ratio=0.5,
        simple_ratio=0.4,
        medium_ratio=0.4,
        complex_ratio=0.2,
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

    result = run_generate_dataset(demo_cfg)
    print(f"\n✅ Demo 生成完成: {result['total_samples']} 条样本\n")

    # Print a few samples
    alpaca_path = Path(result["alpaca_path"])
    if alpaca_path.exists():
        data = json.loads(alpaca_path.read_text(encoding="utf-8"))
        print("=" * 60)
        print("样本预览 (Alpaca 格式):")
        print("=" * 60)
        for i, row in enumerate(data[:5], 1):
            print(f"\n--- 样本 {i} ---")
            print(f"  Instruction: {row['instruction'][:120]}...")
            try:
                output_obj = json.loads(row["output"])
                print(f"  Output:      {json.dumps(output_obj, ensure_ascii=False, indent=4)}")
            except Exception:
                print(f"  Output:      {row['output'][:200]}")
        print()

    sharegpt_path = Path(result["sharegpt_path"])
    if sharegpt_path.exists():
        data = json.loads(sharegpt_path.read_text(encoding="utf-8"))
        print("=" * 60)
        print("样本预览 (ShareGPT 格式 - LLaMA Factory 兼容):")
        print("=" * 60)
        for i, row in enumerate(data[:2], 1):
            print(f"\n--- 样本 {i} ---")
            print(json.dumps(row, ensure_ascii=False, indent=2))
        print()

    print(f"📁 输出文件:")
    print(f"   Alpaca   : {result['alpaca_path']}")
    print(f"   ShareGPT : {result['sharegpt_path']}")
    print(f"   Stats    : {result['stats_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="API-based dataset generation for Franka robot fine-tuning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/01_data_exp/configs/api_generate.yaml"),
        help="Experiment config YAML path (overrides base config).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Global base config YAML path.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo (10 samples) with preview output.",
    )
    parser.add_argument(
        "--show-actions",
        action="store_true",
        help="Only print available action summary, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_actions:
        print_action_summary()
        return

    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )

    if args.demo:
        # Build config from merged to get API settings, then run demo
        section = merged_config.get("dataset_prepare", {}).get("generate", {})
        cfg = GenerateDatasetConfig(
            out_dir=Path(section.get("out_dir", "data_prepare")),
            api_base=str(section.get("api_base", GenerateDatasetConfig.api_base)),
            model=str(section.get("model", GenerateDatasetConfig.model)),
            api_key=str(section.get("api_key", GenerateDatasetConfig.api_key)),
            api_key_env=str(section.get("api_key_env", GenerateDatasetConfig.api_key_env)),
            temperature=float(section.get("temperature", GenerateDatasetConfig.temperature)),
            max_tokens=int(section.get("max_tokens", GenerateDatasetConfig.max_tokens)),
            timeout=int(section.get("timeout", GenerateDatasetConfig.timeout)),
            max_retries=int(section.get("max_retries", GenerateDatasetConfig.max_retries)),
            sleep_seconds=float(section.get("sleep_seconds", GenerateDatasetConfig.sleep_seconds)),
        )
        run_demo(cfg)
        return

    # Full generation
    print_action_summary()
    logger.info("Starting full dataset generation...")
    result = run_generate_from_merged_config(merged_config)

    print(f"\n✅ 数据集生成完成!")
    print(f"   总样本数 : {result['total_samples']}")
    print(f"   Alpaca   : {result['alpaca_path']}")
    print(f"   ShareGPT : {result['sharegpt_path']}")
    print(f"   Stats    : {result['stats_path']}")


if __name__ == "__main__":
    main()
