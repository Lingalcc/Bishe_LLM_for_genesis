#!/usr/bin/env python3
"""Fine-tuning benchmark: evaluate accuracy before & after training.

Workflow:
    1. Evaluate base model accuracy (pre-finetune)
    2. Run fine-tuning with configurable method (lora/qlora/dora/galore)
    3. Evaluate fine-tuned model accuracy (post-finetune)
    4. Generate comparison report with all metrics

Outputs a unified report containing:
    - Task success rate (Accuracy): parse_ok, exact_match, action_match
    - Peak VRAM: training and inference VRAM usage
    - Loss Curve: training loss per step
    - Pre/Post comparison: improvement delta

Usage:
    # Full benchmark (base eval → train → finetuned eval):
    python experiments/02_finetune_exp/run_benchmark.py

    # Skip training (use existing finetuned model):
    python experiments/02_finetune_exp/run_benchmark.py --skip-train

    # Only evaluate pre-finetune baseline:
    python experiments/02_finetune_exp/run_benchmark.py --eval-only base

    # Only evaluate post-finetune:
    python experiments/02_finetune_exp/run_benchmark.py --eval-only finetuned

    # Dry-run (no actual training or evaluation):
    python experiments/02_finetune_exp/run_benchmark.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_merged_config
from src.utils.secrets import safe_json_dumps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _eval_model(
    dataset_file: Path,
    model_path: str,
    report_file: Path,
    *,
    num_samples: int = 100,
    seed: int = 42,
    backend: str = "transformers",
    quantization: str | None = None,
    max_new_tokens: int = 512,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Run local model accuracy evaluation with performance metrics."""
    from src.eval_core.accuracy import AccuracyEvalConfig, run_accuracy_eval

    cfg = AccuracyEvalConfig(
        dataset_file=dataset_file,
        report_file=report_file,
        num_samples=num_samples,
        seed=seed,
        mode="local",
        model_path=model_path,
        backend=backend,
        quantization=quantization,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        system_prompt=system_prompt,
    )
    return run_accuracy_eval(cfg)


def _run_training(merged_config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Run fine-tuning and collect training metrics."""
    from src.finetune_core.train import run_finetune_from_merged_config

    return run_finetune_from_merged_config(
        merged_config,
        dry_run_override=dry_run,
    )


def run_benchmark(
    merged_config: dict[str, Any],
    *,
    eval_only: str | None = None,
    skip_train: bool = False,
    skip_base_eval: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the full fine-tuning benchmark pipeline.

    Args:
        merged_config: Merged YAML configuration dict.
        eval_only: If set, only evaluate "base" or "finetuned" model.
        skip_train: Skip training step (assume finetuned model exists).
        skip_base_eval: Skip base model evaluation (train + eval finetuned only).
        dry_run: Don't actually run anything, just print what would happen.

    Returns:
        Unified report dict with pre/post accuracy, training metrics, comparison.
    """
    bench_cfg = merged_config.get("benchmark", {})
    finetune_cfg = merged_config.get("finetune", {}).get("train", {})
    eval_cfg = merged_config.get("test", {}).get("accuracy_eval", {})

    # Resolve paths and settings
    dataset_file = Path(bench_cfg.get("dataset_file", eval_cfg.get(
        "dataset_file", "data_prepare/genesis_franka_toolcall_alpaca.json")))
    base_model_path = str(bench_cfg.get("base_model_path",
        merged_config.get("app", {}).get("inference", {}).get("local", {}).get("model_path", "")))
    finetuned_model_path = str(bench_cfg.get("finetuned_model_path", "model/my_lora_merged_model"))
    report_dir = Path(bench_cfg.get("report_dir", "experiments/02_finetune_exp/reports"))
    num_samples = int(bench_cfg.get("num_samples", eval_cfg.get("num_samples", 100)))
    seed = int(bench_cfg.get("seed", 42))
    backend = str(bench_cfg.get("backend", "transformers"))
    quantization = bench_cfg.get("quantization")
    max_new_tokens = int(bench_cfg.get("max_new_tokens", 512))
    system_prompt = str(bench_cfg.get("system_prompt", eval_cfg.get("system_prompt",
        "你是 Franka 机械臂控制指令生成器。请把用户自然语言转换为可执行的 JSON action。只输出 JSON，不要输出解释。")))

    report_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "dataset_file": str(dataset_file),
            "base_model_path": base_model_path,
            "finetuned_model_path": finetuned_model_path,
            "finetune_method": finetune_cfg.get("finetune_method", "lora"),
            "num_eval_samples": num_samples,
            "backend": backend,
            "quantization": quantization,
        },
    }

    if dry_run:
        report["dry_run"] = True
        print("\n[Dry Run] Benchmark configuration:")
        print(safe_json_dumps(report, ensure_ascii=False, indent=2))
        return report

    # Step 1: Pre-finetune evaluation
    if not skip_base_eval and eval_only in (None, "base"):
        if base_model_path:
            logger.info("=== Step 1: Evaluating base model: %s ===", base_model_path)
            pre_report = _eval_model(
                dataset_file=dataset_file,
                model_path=base_model_path,
                report_file=report_dir / "pre_finetune_accuracy.json",
                num_samples=num_samples, seed=seed, backend=backend,
                quantization=quantization, max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )
            report["pre_finetune"] = {
                "model_path": base_model_path,
                "parse_ok_rate": pre_report.get("parse_ok_rate", 0.0),
                "exact_match_rate": pre_report.get("exact_match_rate", 0.0),
                "action_match_rate": pre_report.get("action_match_rate", 0.0),
                "avg_latency_sec": pre_report.get("avg_latency_sec", 0.0),
                "avg_peak_vram_mb": pre_report.get("avg_peak_vram_mb", 0.0),
                "max_peak_vram_mb": pre_report.get("max_peak_vram_mb", 0.0),
            }
            logger.info(
                "Base model: accuracy=%.2f%% action_match=%.2f%% vram=%.0fMB",
                pre_report["exact_match_rate"] * 100,
                pre_report["action_match_rate"] * 100,
                pre_report.get("max_peak_vram_mb", 0),
            )
        else:
            logger.warning("No base_model_path configured, skipping pre-finetune eval")

    # Step 2: Fine-tuning
    if eval_only is None and not skip_train:
        logger.info("=== Step 2: Fine-tuning with method=%s ===", finetune_cfg.get("finetune_method", "lora"))
        try:
            train_result = _run_training(merged_config)
            report["training"] = {
                "method": train_result.get("method", ""),
                "executed": train_result.get("executed", False),
                "command": train_result.get("command_shell", ""),
                "gpus": train_result.get("gpus"),
            }
            if "training_metrics" in train_result:
                tm = train_result["training_metrics"]
                report["training"].update({
                    "total_time_sec": tm.get("total_time_sec", 0),
                    "total_steps": tm.get("total_steps", 0),
                    "total_epochs": tm.get("total_epochs", 0),
                    "final_loss": tm.get("final_loss", 0),
                    "min_loss": tm.get("min_loss", 0),
                    "min_loss_step": tm.get("min_loss_step", 0),
                    "peak_vram_mb": tm.get("peak_vram_mb", 0),
                    "avg_vram_mb": tm.get("avg_vram_mb", 0),
                    "peak_delta_vram_mb": tm.get("peak_delta_vram_mb", 0),
                    "avg_delta_vram_mb": tm.get("avg_delta_vram_mb", 0),
                    "loss_curve": tm.get("loss_curve", {}),
                    "vram_detail": tm.get("vram_detail", {}),
                })
                logger.info(
                    "Training done: %d steps, final_loss=%.4f, peak_vram=%.0fMB, time=%.0fs",
                    tm.get("total_steps", 0), tm.get("final_loss", 0),
                    tm.get("peak_vram_mb", 0), tm.get("total_time_sec", 0),
                )
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            report["training"] = {"error": str(exc)}

    # Step 3: Post-finetune evaluation
    if eval_only in (None, "finetuned"):
        finetuned_abs = Path(finetuned_model_path)
        if not finetuned_abs.is_absolute():
            finetuned_abs = REPO_ROOT / finetuned_abs
        if not finetuned_abs.exists():
            logger.warning(
                "Finetuned model path does not exist: %s — skipping post-finetune eval",
                finetuned_abs,
            )
        elif finetuned_model_path:
            logger.info("=== Step 3: Evaluating finetuned model: %s ===", finetuned_model_path)
            post_report = _eval_model(
                dataset_file=dataset_file,
                model_path=finetuned_model_path,
                report_file=report_dir / "post_finetune_accuracy.json",
                num_samples=num_samples, seed=seed, backend=backend,
                quantization=quantization, max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )
            report["post_finetune"] = {
                "model_path": finetuned_model_path,
                "parse_ok_rate": post_report.get("parse_ok_rate", 0.0),
                "exact_match_rate": post_report.get("exact_match_rate", 0.0),
                "action_match_rate": post_report.get("action_match_rate", 0.0),
                "avg_latency_sec": post_report.get("avg_latency_sec", 0.0),
                "avg_peak_vram_mb": post_report.get("avg_peak_vram_mb", 0.0),
                "max_peak_vram_mb": post_report.get("max_peak_vram_mb", 0.0),
            }
            logger.info(
                "Finetuned model: accuracy=%.2f%% action_match=%.2f%% vram=%.0fMB",
                post_report["exact_match_rate"] * 100,
                post_report["action_match_rate"] * 100,
                post_report.get("max_peak_vram_mb", 0),
            )
        else:
            logger.warning("No finetuned_model_path configured, skipping post-finetune eval")

    # Step 4: Comparison
    if "pre_finetune" in report and "post_finetune" in report:
        pre = report["pre_finetune"]
        post = report["post_finetune"]
        report["comparison"] = {
            "accuracy_delta": post["exact_match_rate"] - pre["exact_match_rate"],
            "action_match_delta": post["action_match_rate"] - pre["action_match_rate"],
            "parse_ok_delta": post["parse_ok_rate"] - pre["parse_ok_rate"],
            "latency_delta_sec": post["avg_latency_sec"] - pre["avg_latency_sec"],
        }

    # Write report
    benchmark_report_path = report_dir / "benchmark_report.json"
    benchmark_report_path.write_text(safe_json_dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Benchmark report: %s", benchmark_report_path)

    # Print summary
    _print_summary(report)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("  Fine-tuning Benchmark Report")
    print("=" * 70)

    cfg = report.get("config", {})
    print(f"  Method     : {cfg.get('finetune_method', 'N/A')}")
    print(f"  Dataset    : {cfg.get('dataset_file', 'N/A')}")
    print(f"  Eval samples: {cfg.get('num_eval_samples', 'N/A')}")

    if "training" in report and "error" not in report["training"]:
        t = report["training"]
        print(f"\n  --- Training ---")
        print(f"  Total Steps  : {t.get('total_steps', 'N/A')}")
        print(f"  Total Epochs : {t.get('total_epochs', 'N/A')}")
        print(f"  Final Loss   : {t.get('final_loss', 'N/A'):.4f}" if t.get('final_loss') else "  Final Loss   : N/A")
        print(f"  Min Loss     : {t.get('min_loss', 'N/A'):.4f} (step {t.get('min_loss_step', '?')})" if t.get('min_loss') else "  Min Loss     : N/A")
        print(f"  Peak VRAM    : {t.get('peak_vram_mb', 0):.0f} MB")
        print(f"  Training Time: {t.get('total_time_sec', 0):.0f} sec")

    header = f"\n  {'Metric':<25} {'Base Model':>12} {'Finetuned':>12} {'Delta':>10}"
    print(header)
    print("  " + "-" * 60)

    pre = report.get("pre_finetune", {})
    post = report.get("post_finetune", {})
    comp = report.get("comparison", {})

    def _row(name: str, key: str, fmt: str = ".2%", delta_key: str | None = None) -> None:
        pre_val = pre.get(key, "")
        post_val = post.get(key, "")
        delta = comp.get(delta_key, "") if delta_key else ""
        pre_s = f"{pre_val:{fmt}}" if isinstance(pre_val, (int, float)) else str(pre_val) or "—"
        post_s = f"{post_val:{fmt}}" if isinstance(post_val, (int, float)) else str(post_val) or "—"
        delta_s = f"{delta:+{fmt}}" if isinstance(delta, (int, float)) else "—"
        print(f"  {name:<25} {pre_s:>12} {post_s:>12} {delta_s:>10}")

    _row("Accuracy (exact_match)", "exact_match_rate", ".2%", "accuracy_delta")
    _row("Action Match Rate", "action_match_rate", ".2%", "action_match_delta")
    _row("Parse OK Rate", "parse_ok_rate", ".2%", "parse_ok_delta")
    _row("Avg Latency (sec)", "avg_latency_sec", ".3f", "latency_delta_sec")
    _row("Peak VRAM (MB)", "max_peak_vram_mb", ".0f")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning benchmark: pre/post accuracy comparison.")
    parser.add_argument("--config", type=Path, default=Path("experiments/02_finetune_exp/configs/train.yaml"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--eval-only", choices=["base", "finetuned"], default=None,
                        help="Only run evaluation for the specified model.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the fine-tuning step.")
    parser.add_argument("--skip-base-eval", action="store_true",
                        help="Skip base model evaluation (train + eval finetuned only).")
    parser.add_argument("--dry-run", action="store_true", help="Don't run anything, just show config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )
    run_benchmark(
        merged,
        eval_only=args.eval_only,
        skip_train=args.skip_train,
        skip_base_eval=args.skip_base_eval,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
