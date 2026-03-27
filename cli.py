#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_core.calibration import calibrate_from_merged_config
from src.data_core.generate import run_generate_from_merged_config
from src.data_core.split_dataset import run_split_from_merged_config
from src.eval_core.accuracy import run_accuracy_from_merged_config
from src.eval_core.inference_benchmark import InferenceBenchmarkConfig, run_inference_benchmark
from src.finetune_core.train import SUPPORTED_FINETUNE_METHODS, run_finetune_from_merged_config
from src.sim_core.runtime import SimRuntimeConfig, run_instruction_to_action, run_model_interactive_session
from src.utils.config import load_merged_config


def _load_cfg(base_config: Path, override_config: Path | None) -> dict:
    return load_merged_config(
        base_config_path=base_config,
        override_config_path=override_config,
    )


def _print_generate_progress(update: dict[str, object]) -> None:
    event = str(update.get("event", "batch_completed"))
    if event == "batch_started":
        print(
            "[progress] "
            f"{update.get('batch_label', '?')} started "
            f"difficulty={update['difficulty']} "
            f"request={int(update['requested_batch_size'])} "
            f"timeout={int(update['timeout'])}s",
            flush=True,
        )
        return
    if event == "batch_retry":
        print(
            "[retry] "
            f"{update.get('batch_label', '?')} "
            f"attempt {int(update['attempt'])}/{int(update['max_retries'])} "
            f"difficulty={update['difficulty']} "
            f"request={int(update['requested_batch_size'])} "
            f"timeout={int(update['timeout'])}s "
            f"error={update['error']}",
            flush=True,
        )
        return
    if event == "batch_failed":
        print(
            "[failed] "
            f"{update.get('batch_label', '?')} "
            f"difficulty={update['difficulty']} "
            f"request={int(update['requested_batch_size'])} "
            f"error={update['error']}",
            flush=True,
        )
        return

    unique_samples = int(update["unique_samples"])
    target_samples = int(update["target_samples"])
    percent = 100.0 * unique_samples / max(1, target_samples)
    print(
        "[progress] "
        f"{update.get('batch_label', '?')} "
        f"round {int(update['round_idx'])}/{int(update['max_rounds'])} "
        f"batch {int(update['batch_idx'])}/{int(update['batch_total'])} "
        f"difficulty={update['difficulty']} "
        f"accepted={int(update['accepted_count'])} "
        f"dup={int(update['duplicate_count'])} "
        f"invalid={int(update['invalid_count'])} "
        f"total={unique_samples}/{target_samples} "
        f"({percent:.1f}%)",
        flush=True,
    )


def _run_data_generate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    outputs = run_generate_from_merged_config(
        cfg,
        progress_callback=_print_generate_progress,
    )
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


def _run_data_split(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    metadata = run_split_from_merged_config(
        cfg,
        input_file=args.input_file,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_name=args.train_name,
        val_name=args.val_name,
        test_name=args.test_name,
        metadata_name=args.metadata_name,
        preserve_existing_splits=args.preserve_existing_splits,
    )
    print(f"[split] train: {metadata['splits']['train']['num_samples']} -> {metadata['splits']['train']['path']}")
    print(f"[split] val  : {metadata['splits']['val']['num_samples']} -> {metadata['splits']['val']['path']}")
    print(f"[split] test : {metadata['splits']['test']['num_samples']} -> {metadata['splits']['test']['path']}")
    print(f"[split] meta : {metadata.get('metadata_file', '')}")


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


def _format_optional_metric(value: object, *, digits: int) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "N/A"


def _run_eval_benchmark(args: argparse.Namespace) -> None:
    cfg = InferenceBenchmarkConfig(
        backend=args.backend,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        quantization=args.quantization,
        require_gpu=bool(args.require_gpu),
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        prompt=args.prompt,
        prompts_file=str(args.prompts_file) if args.prompts_file else None,
        use_chat=args.use_chat,
        warmup_batches=args.warmup_batches,
        shuffle_prompts=not args.no_shuffle_prompts,
        random_seed=args.random_seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=not args.no_trust_remote_code,
        use_flash_attention=args.use_flash_attention,
        output_json=str(args.output_json),
        output_csv=str(args.output_csv) if args.output_csv else None,
    )
    report = run_inference_benchmark(cfg)
    print(f"[ok] backend          : {report['backend']}")
    print(f"[ok] quantization     : {report['quantization']}")
    print(f"[ok] batch_size       : {report['batch_size']}")
    print(f"[ok] num_samples      : {report['num_samples']}")
    print(f"[ok] avg_latency (s)  : {report['avg_latency']:.4f}")
    print(f"[ok] p50_latency (s)  : {report['p50_latency']:.4f}")
    print(f"[ok] p95_latency (s)  : {report['p95_latency']:.4f}")
    print(f"[ok] throughput       : {report['throughput']:.4f} samples/s")
    print(f"[ok] token_count      : {report.get('token_count_method', 'unknown')}")
    print(f"[ok] prompt_sampling  : {report.get('prompt_sampling_strategy', 'unknown')}")
    print(f"[ok] avg_ttft (s)     : {_format_optional_metric(report.get('avg_ttft_sec'), digits=4)}")
    print(f"[ok] avg_tpot (s)     : {_format_optional_metric(report.get('avg_tpot_sec'), digits=6)}")
    print(
        f"[ok] avg sec/output tok: "
        f"{_format_optional_metric(report.get('avg_e2e_time_per_output_token_sec'), digits=6)}"
    )
    print(f"[ok] peak_memory (MB) : {report['peak_memory']:.2f}")
    print(f"[ok] errors           : {report['errors']}")
    print(f"[ok] json report      : {args.output_json}")
    if args.output_csv:
        print(f"[ok] csv report       : {args.output_csv}")


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


def _run_app_interactive(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.base_config, args.config)
    run_model_interactive_session(merged_config=cfg, disable_sim_state=args.disable_sim_state)


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

    data_split_parser = data_subparsers.add_parser("split", help="Split dataset into train/val/test.")
    data_split_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    data_split_parser.add_argument("--config", type=Path, default=None)
    data_split_parser.add_argument("--input-file", type=Path, default=None)
    data_split_parser.add_argument("--out-dir", type=Path, default=None)
    data_split_parser.add_argument("--train-ratio", type=float, default=None)
    data_split_parser.add_argument("--val-ratio", type=float, default=None)
    data_split_parser.add_argument("--test-ratio", type=float, default=None)
    data_split_parser.add_argument("--seed", type=int, default=None)
    data_split_parser.add_argument("--train-name", type=str, default=None)
    data_split_parser.add_argument("--val-name", type=str, default=None)
    data_split_parser.add_argument("--test-name", type=str, default=None)
    data_split_parser.add_argument("--metadata-name", type=str, default=None)
    data_split_parser.add_argument(
        "--preserve-existing-splits",
        action="store_true",
        help="仅对新增样本做切分并追加到现有 split，保留旧 train/val/test。",
    )
    data_split_parser.set_defaults(handler=_run_data_split)

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

    eval_benchmark_parser = eval_subparsers.add_parser(
        "benchmark", help="Run local inference benchmark (HF/vLLM)."
    )
    eval_benchmark_parser.add_argument("--backend", required=True, choices=["transformers", "vllm", "llama.cpp", "exllamav2"])
    eval_benchmark_parser.add_argument("--model-path", required=True)
    eval_benchmark_parser.add_argument("--tokenizer-path", default=None)
    eval_benchmark_parser.add_argument("--quantization", default=None)
    eval_benchmark_parser.add_argument("--require-gpu", action="store_true")
    eval_benchmark_parser.add_argument("--batch-size", type=int, default=1)
    eval_benchmark_parser.add_argument("--num-samples", type=int, default=32)
    eval_benchmark_parser.add_argument(
        "--prompt",
        type=str,
        default="Generate one short JSON action for robot arm control.",
    )
    eval_benchmark_parser.add_argument("--prompts-file", type=Path, default=None)
    eval_benchmark_parser.add_argument("--use-chat", action="store_true")
    eval_benchmark_parser.add_argument("--warmup-batches", type=int, default=1)
    eval_benchmark_parser.add_argument("--no-shuffle-prompts", action="store_true")
    eval_benchmark_parser.add_argument("--random-seed", type=int, default=42)
    eval_benchmark_parser.add_argument("--max-new-tokens", type=int, default=128)
    eval_benchmark_parser.add_argument("--temperature", type=float, default=0.0)
    eval_benchmark_parser.add_argument("--max-model-len", type=int, default=4096)
    eval_benchmark_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    eval_benchmark_parser.add_argument("--use-flash-attention", action="store_true")
    eval_benchmark_parser.add_argument("--no-trust-remote-code", action="store_true")
    eval_benchmark_parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/03_eval_exp/reports/inference_benchmark.json"),
    )
    eval_benchmark_parser.add_argument("--output-csv", type=Path, default=None)
    eval_benchmark_parser.set_defaults(handler=_run_eval_benchmark)

    app_parser = root_subparsers.add_parser("app", help="Simulation runtime commands.")
    app_subparsers = app_parser.add_subparsers(dest="app_command", required=True)

    app_instruction_parser = app_subparsers.add_parser("run-instruction", help="Run instruction -> action.")
    app_instruction_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    app_instruction_parser.add_argument("--config", type=Path, default=None)
    app_instruction_parser.add_argument("--instruction", type=str, required=True)
    app_instruction_parser.add_argument("--print-raw", action="store_true")
    app_instruction_parser.add_argument("--disable-sim-state", action="store_true")
    app_instruction_parser.set_defaults(handler=_run_app_instruction)

    app_interactive_parser = app_subparsers.add_parser(
        "interactive",
        help="Keep the simulation running and repeatedly execute model-generated actions.",
    )
    app_interactive_parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    app_interactive_parser.add_argument("--config", type=Path, default=None)
    app_interactive_parser.add_argument("--disable-sim-state", action="store_true")
    app_interactive_parser.set_defaults(handler=_run_app_interactive)

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
