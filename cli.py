#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.app.run_instruction_to_action import (
    DEFAULT_CONFIG_PATH as APP_DEFAULT_CONFIG_PATH,
    run_instruction_to_action,
)
from src.data.run_augment_dataset import (
    DEFAULT_CONFIG_PATH as DATA_AUGMENT_DEFAULT_CONFIG_PATH,
    run_augment,
)
from src.data.run_generate_dataset import (
    DEFAULT_CONFIG_PATH as DATA_GENERATE_DEFAULT_CONFIG_PATH,
    run_generate,
)
from src.eval.run_accuracy_eval import (
    DEFAULT_CONFIG_PATH as EVAL_DEFAULT_CONFIG_PATH,
    run_accuracy_eval,
)
from src.finetune.run_finetune import (
    DEFAULT_PIPELINE_CONFIG,
    SUPPORTED_FINETUNE_METHODS,
    run_finetune,
)


def _run_data_generate(args: argparse.Namespace) -> None:
    run_generate(
        config_path=args.config,
        passthrough=args.extra_args,
    )


def _run_data_augment(args: argparse.Namespace) -> None:
    run_augment(
        config_path=args.config,
        passthrough=args.extra_args,
    )


def _run_finetune_start(args: argparse.Namespace) -> None:
    run_finetune(
        pipeline_config=args.pipeline_config,
        llamafactory_dir=args.llamafactory_dir,
        config=args.config,
        gpus=args.gpus,
        dry_run=args.dry_run,
        finetune_method=args.finetune_method,
        extra_args=args.extra_args,
    )


def _run_eval_accuracy(args: argparse.Namespace) -> None:
    run_accuracy_eval(
        config_path=args.config,
        passthrough=args.extra_args,
    )


def _run_app_instruction(args: argparse.Namespace) -> None:
    run_instruction_to_action(
        config_path=args.config,
        instruction=args.instruction,
        print_raw=args.print_raw,
        disable_sim_state=args.disable_sim_state,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for data, finetune, evaluation and app workflows."
    )
    root_subparsers = parser.add_subparsers(dest="domain", required=True)

    data_parser = root_subparsers.add_parser("data", help="Data preparation commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    data_generate_parser = data_subparsers.add_parser(
        "generate",
        help="Generate tool-call dataset.",
    )
    data_generate_parser.add_argument(
        "--config",
        type=Path,
        default=DATA_GENERATE_DEFAULT_CONFIG_PATH,
        help="Config path used to load data.generate defaults.",
    )
    data_generate_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to src.data.generate_genesis_franka_dataset.",
    )
    data_generate_parser.set_defaults(handler=_run_data_generate)

    data_augment_parser = data_subparsers.add_parser(
        "augment",
        help="Augment generated dataset through API calls.",
    )
    data_augment_parser.add_argument(
        "--config",
        type=Path,
        default=DATA_AUGMENT_DEFAULT_CONFIG_PATH,
        help="Config path used to load data.augment defaults.",
    )
    data_augment_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to src.data.augment_genesis_franka_dataset_with_api.",
    )
    data_augment_parser.set_defaults(handler=_run_data_augment)

    finetune_parser = root_subparsers.add_parser("finetune", help="Model fine-tuning commands.")
    finetune_subparsers = finetune_parser.add_subparsers(dest="finetune_command", required=True)

    finetune_start_parser = finetune_subparsers.add_parser(
        "start",
        help="Start fine-tuning with LLaMA-Factory.",
    )
    finetune_start_parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
        help="Unified config path for finetune.train section.",
    )
    finetune_start_parser.add_argument(
        "--llamafactory-dir",
        type=Path,
        default=None,
        help="Path to LLaMA-Factory project.",
    )
    finetune_start_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML train config path.",
    )
    finetune_start_parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1.",
    )
    finetune_start_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only.",
    )
    finetune_start_parser.add_argument(
        "--finetune-method",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_FINETUNE_METHODS),
        help="Fine-tuning method override.",
    )
    finetune_start_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to llamafactory-cli train.",
    )
    finetune_start_parser.set_defaults(handler=_run_finetune_start)

    eval_parser = root_subparsers.add_parser("eval", help="Evaluation commands.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_accuracy_parser = eval_subparsers.add_parser(
        "accuracy",
        help="Run tool-call accuracy evaluation.",
    )
    eval_accuracy_parser.add_argument(
        "--config",
        type=Path,
        default=EVAL_DEFAULT_CONFIG_PATH,
        help="Config path used to load test.accuracy_eval defaults.",
    )
    eval_accuracy_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to src.eval.evaluate_toolcall_accuracy.",
    )
    eval_accuracy_parser.set_defaults(handler=_run_eval_accuracy)

    app_parser = root_subparsers.add_parser("app", help="Application runtime commands.")
    app_subparsers = app_parser.add_subparsers(dest="app_command", required=True)

    app_instruction_parser = app_subparsers.add_parser(
        "run-instruction",
        help="Run instruction -> action conversion.",
    )
    app_instruction_parser.add_argument(
        "--config",
        type=Path,
        default=APP_DEFAULT_CONFIG_PATH,
        help="Unified config YAML path.",
    )
    app_instruction_parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="One-shot instruction. Empty means interactive mode.",
    )
    app_instruction_parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print raw model response.",
    )
    app_instruction_parser.add_argument(
        "--disable-sim-state",
        action="store_true",
        help="Disable simulation state injection.",
    )
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
