#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_core.calibration import calibrate_from_merged_config
from src.data_core.generate import run_generate_from_merged_config
from src.eval_core.accuracy import run_accuracy_from_merged_config
from src.finetune_core.train import SUPPORTED_FINETUNE_METHODS, run_finetune_from_merged_config
from src.sim_core.runtime import SimRuntimeConfig, run_instruction_to_action
from src.utils.config import load_merged_config


def _load_cfg(base_config: Path, override_config: Path | None) -> dict:
    return load_merged_config(
        base_config_path=base_config,
        override_config_path=override_config,
    )


def _run_data_generate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    outputs = run_generate_from_merged_config(cfg)
    print(f"[ok] alpaca  : {outputs['alpaca_path']}")
    print(f"[ok] sharegpt: {outputs['sharegpt_path']}")
    print(f"[ok] stats   : {outputs['stats_path']}")


def _run_data_calibrate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    report = calibrate_from_merged_config(cfg)
    print(f"[ok] dataset     : {report['dataset_file']}")
    print(f"[ok] total rows  : {report['total_rows']}")
    print(f"[ok] valid rows  : {report['valid_rows']}")
    print(f"[ok] invalid rows: {report['invalid_rows']}")
    print(f"[ok] valid ratio : {report['valid_ratio']:.2%}")


def _run_finetune_start(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    if args.finetune_method:
        cfg.setdefault("finetune", {}).setdefault("train", {})["finetune_method"] = args.finetune_method
    result = run_finetune_from_merged_config(
        cfg,
        dry_run_override=args.dry_run,
        extra_args=tuple(args.extra_args),
    )
    print(f"[finetune] working_dir: {result['working_dir']}")
    print(f"[finetune] method     : {result['method']}")
    print(f"[finetune] command    : {result['command_shell']}")
    if result.get("gpus") is not None:
        print(f"[finetune] GPUs       : {result['gpus']}")
    print(f"[finetune] executed   : {result['executed']}")
    if "training_metrics" in result:
        tm = result["training_metrics"]
        print(f"[finetune] time (sec) : {tm.get('total_time_sec', 0):.0f}")
        print(f"[finetune] final loss  : {tm.get('final_loss', 0):.4f}")
        print(f"[finetune] min loss    : {tm.get('min_loss', 0):.4f} (step {tm.get('min_loss_step', 0)})")
        print(f"[finetune] peak VRAM   : {tm.get('peak_vram_mb', 0):.0f} MB")
        if "peak_delta_vram_mb" in tm:
            print(f"[finetune] peak ΔVRAM  : {tm.get('peak_delta_vram_mb', 0):.0f} MB")


def _run_eval_accuracy(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    report = run_accuracy_from_merged_config(cfg)
    print(f"[ok] evaluated samples : {report['num_samples_evaluated']}")
    print(f"[ok] parse ok          : {report['parse_ok']} ({report['parse_ok_rate']:.4f})")
    print(f"[ok] exact match       : {report['exact_match']} ({report['exact_match_rate']:.4f})")
    print(f"[ok] action match      : {report['action_match']} ({report['action_match_rate']:.4f})")
    if report.get("mode") == "local":
        print(f"[ok] avg latency (sec) : {report.get('avg_latency_sec', 0):.3f}")
        print(f"[ok] avg throughput    : {report.get('avg_throughput_tps', 0):.1f} tokens/s")
        print(f"[ok] peak VRAM (MB)    : {report.get('max_peak_vram_mb', 0):.0f}")


def _run_finetune_benchmark(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_benchmark",
        Path(__file__).resolve().parent / "experiments" / "02_finetune_exp" / "run_benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_benchmark(
        cfg,
        eval_only=args.eval_only,
        skip_train=args.skip_train,
        skip_base_eval=args.skip_base_eval,
        dry_run=args.dry_run,
    )


def _run_app_instruction(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    runtime = SimRuntimeConfig(
        config_path=None,
        instruction=args.instruction,
        print_raw=args.print_raw,
        disable_sim_state=args.disable_sim_state,
    )
    result = run_instruction_to_action(runtime, merged_config=cfg)
    if args.print_raw:
        print("[model_raw]")
        print(result["raw"])
    if result.get("scene_state") is not None:
        print("[scene_state]")
        print(json.dumps(result["scene_state"], ensure_ascii=False, indent=2))
    print("[action_json]")
    print(json.dumps(result["payload"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for data, finetune, evaluation and simulation workflows."
    )
    root_subparsers = parser.add_subparsers(dest="domain", required=True)

    data_parser = root_subparsers.add_parser("data", help="Data preparation commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    data_generate_parser = data_subparsers.add_parser("generate", help="Generate tool-call dataset.")
    data_generate_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    data_generate_parser.add_argument("--config", type=Path, default=None)
    data_generate_parser.set_defaults(handler=_run_data_generate)

    data_calibrate_parser = data_subparsers.add_parser("calibrate", help="Validate dataset quality.")
    data_calibrate_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    data_calibrate_parser.add_argument("--config", type=Path, default=None)
    data_calibrate_parser.set_defaults(handler=_run_data_calibrate)

    finetune_parser = root_subparsers.add_parser("finetune", help="Model fine-tuning commands.")
    finetune_subparsers = finetune_parser.add_subparsers(dest="finetune_command", required=True)

    finetune_start_parser = finetune_subparsers.add_parser("start", help="Start fine-tuning.")
    finetune_start_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    finetune_start_parser.add_argument("--config", type=Path, default=None)
    finetune_start_parser.add_argument("--dry-run", action="store_true", help="Do not execute command.")
    finetune_start_parser.add_argument(
        "--finetune-method",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_FINETUNE_METHODS),
        help="Reserved compatibility flag; prefer config override.",
    )
    finetune_start_parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    finetune_start_parser.set_defaults(handler=_run_finetune_start)

    finetune_benchmark_parser = finetune_subparsers.add_parser(
        "benchmark", help="Run pre/post fine-tuning accuracy benchmark.")
    finetune_benchmark_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    finetune_benchmark_parser.add_argument("--config", type=Path, default=None)
    finetune_benchmark_parser.add_argument("--dry-run", action="store_true")
    finetune_benchmark_parser.add_argument("--skip-train", action="store_true",
                                           help="Skip training, only evaluate.")
    finetune_benchmark_parser.add_argument("--skip-base-eval", action="store_true",
                                           help="Skip base model eval (train + eval finetuned only).")
    finetune_benchmark_parser.add_argument("--eval-only", choices=["base", "finetuned"], default=None,
                                           help="Only evaluate one model.")
    finetune_benchmark_parser.set_defaults(handler=_run_finetune_benchmark)

    eval_parser = root_subparsers.add_parser("eval", help="Evaluation commands.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_accuracy_parser = eval_subparsers.add_parser("accuracy", help="Run tool-call accuracy evaluation.")
    eval_accuracy_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    eval_accuracy_parser.add_argument("--config", type=Path, default=None)
    eval_accuracy_parser.set_defaults(handler=_run_eval_accuracy)

    app_parser = root_subparsers.add_parser("app", help="Simulation runtime commands.")
    app_subparsers = app_parser.add_subparsers(dest="app_command", required=True)

    app_instruction_parser = app_subparsers.add_parser("run-instruction", help="Run instruction -> action.")
    app_instruction_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    app_instruction_parser.add_argument("--config", type=Path, default=None)
    app_instruction_parser.add_argument("--instruction", type=str, required=True)
    app_instruction_parser.add_argument("--print-raw", action="store_true")
    app_instruction_parser.add_argument("--disable-sim-state", action="store_true")
    app_instruction_parser.set_defaults(handler=_run_app_instruction)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        raise SystemExit(2)
    handler(args)


if __name__ == "__main__":
    main()
