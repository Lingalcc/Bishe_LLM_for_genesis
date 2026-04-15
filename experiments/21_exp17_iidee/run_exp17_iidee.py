#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "21_exp17_iidee" / "configs" / "iidee.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.accuracy import run_accuracy_from_merged_config
from src.utils.config import load_merged_config
from src.utils.run_meta import record_run_meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验 21 / Exp17：重要性感知动态早退（IIDEE）最小可运行版。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="实验覆盖配置。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG, help="基础配置。")
    parser.add_argument("--model-path", type=str, default=None, help="覆盖本地模型路径。")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="覆盖 tokenizer 路径。")
    parser.add_argument("--test-file", type=Path, default=None, help="覆盖测试集文件。")
    parser.add_argument("--report-file", type=Path, default=None, help="覆盖报告输出路径。")
    parser.add_argument("--num-samples", type=int, default=None, help="评测样本数。")
    parser.add_argument("--seed", type=int, default=None, help="抽样随机种子。")
    parser.add_argument("--backend", type=str, default=None, help="推理后端，当前早退仅支持 transformers。")
    parser.add_argument("--quantization", type=str, default=None, help="量化方式。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数。")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度。")
    parser.add_argument("--early-exit-enabled", action="store_true", help="显式启用早退。")
    parser.add_argument("--disable-early-exit", action="store_true", help="显式关闭早退，便于做 full-depth 对照。")
    parser.add_argument("--exit-layers", type=str, default=None, help="候选退出层，例如 10,14,18,22,28。")
    parser.add_argument("--tau-importance", type=float, default=None, help="累计重要性阈值 tau_I。")
    parser.add_argument("--tau-confidence", type=float, default=None, help="置信度阈值 tau_C。")
    parser.add_argument("--importance-file", type=Path, default=None, help="层重要性文件路径。")
    parser.add_argument("--early-exit-warmup-tokens", type=int, default=None, help="前多少个生成 token 保持当前深度，不执行降层提交。")
    parser.add_argument("--early-exit-min-streak", type=int, default=None, help="连续多少个 token 命中退出条件后，才对下一 token 提交降层。")
    parser.add_argument("--early-exit-fallback-on-invalid-json", action="store_true", help="若早退输出不是可解析 JSON，则自动用 full-depth 重跑。")
    parser.add_argument("--disable-early-exit-fallback", action="store_true", help="关闭 invalid JSON 自动 full-depth 回退。")
    parser.add_argument("--early-exit-protect-open-string", action="store_true", help="若当前仍处于引号未闭合状态，则禁止提交退出。")
    parser.add_argument("--early-exit-draft-only-layers", type=str, default=None, help="这些层只做草拟候选，不允许成为最终退出层，例如 22。")
    return parser.parse_args(argv)


def _ensure_accuracy_section(merged_config: dict[str, Any]) -> dict[str, Any]:
    test_cfg = merged_config.setdefault("test", {})
    if not isinstance(test_cfg, dict):
        raise ValueError("merged_config.test 必须是 dict。")
    accuracy_cfg = test_cfg.setdefault("accuracy_eval", {})
    if not isinstance(accuracy_cfg, dict):
        raise ValueError("merged_config.test.accuracy_eval 必须是 dict。")
    return accuracy_cfg


def _apply_override(section: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        section[key] = value


def _json_inline(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _print_report(report: dict[str, Any]) -> None:
    print(f"[ok] evaluated samples : {report['num_samples_evaluated']}")
    print(f"[ok] avg latency sec   : {report['avg_latency_sec']:.4f}")
    print(f"[ok] avg tokens/s      : {report['avg_throughput_tps']:.4f}")
    print(f"[ok] parse ok          : {report['parse_ok']} ({report['parse_ok_rate']:.4f})")
    print(f"[ok] exact match       : {report['exact_match']} ({report['exact_match_rate']:.4f})")
    print(f"[ok] action match      : {report['action_match']} ({report['action_match_rate']:.4f})")

    early_exit = report.get("early_exit", {})
    if isinstance(early_exit, dict) and early_exit:
        print(f"[ok] avg exit/sample   : {float(early_exit.get('avg_exit_layer_per_sample', 0.0)):.4f}")
        print(f"[ok] avg exit/token    : {float(early_exit.get('avg_exit_layer_per_token', 0.0)):.4f}")
        print(f"[ok] fallback used     : {int(early_exit.get('fallback_used_count', 0) or 0)}")
        print(f"[ok] string guard blk : {int(early_exit.get('string_guard_blocked_tokens', 0) or 0)}")
        print(f"[ok] draft candidates  : {int(early_exit.get('draft_only_candidate_tokens', 0) or 0)}")
        print(f"[ok] draft matches     : {int(early_exit.get('draft_verified_matches', 0) or 0)}")
        print(f"[ok] draft mismatches  : {int(early_exit.get('draft_verified_mismatches', 0) or 0)}")
        print(f"[ok] token exits       : {_json_inline(early_exit.get('token_exit_layer_histogram', {}))}")
        candidate_probe_summary = early_exit.get("candidate_probe_summary", {})
        if isinstance(candidate_probe_summary, dict):
            for layer_key, layer_stats in sorted(candidate_probe_summary.items(), key=lambda item: int(item[0])):
                if not isinstance(layer_stats, dict):
                    continue
                print(
                    "[ok] candidate probe  : "
                    f"layer={layer_key} "
                    f"avg_max_prob={float(layer_stats.get('avg_max_prob', 0.0)):.4f} "
                    f"meet_I={float(layer_stats.get('meets_importance_rate', 0.0)):.4f} "
                    f"meet_C={float(layer_stats.get('meets_confidence_rate', 0.0)):.4f} "
                    f"exit_rate={float(layer_stats.get('exit_rate', 0.0)):.4f}"
                )
        for item in early_exit.get("sample_exit_layer_stats", []):
            if not isinstance(item, dict):
                continue
            print(
                "[ok] sample exit layer : "
                f"dataset_index={item.get('dataset_index')} "
                f"avg_exit_layer={float(item.get('avg_exit_layer', 0.0)):.4f} "
                f"tokens={int(item.get('tokens_generated', 0) or 0)}"
            )

    parse_fail = report.get("parse_failure_diagnostics", {})
    if isinstance(parse_fail, dict) and parse_fail:
        print(f"[ok] parse fail samples : {int(parse_fail.get('failure_count', 0) or 0)}")
        print(
            "[ok] first error kinds : "
            f"{_json_inline(parse_fail.get('first_error_kind_histogram', {}))}"
        )
        print(
            "[ok] first error toks  : "
            f"{_json_inline(parse_fail.get('first_error_token_category_histogram', {}))}"
        )
        print(
            "[ok] first error exits : "
            f"{_json_inline(parse_fail.get('first_error_exit_layer_histogram', {}))}"
        )
        for item in parse_fail.get("samples", []):
            if not isinstance(item, dict):
                continue
            print(
                "[ok] parse fail sample : "
                f"dataset_index={item.get('dataset_index')} "
                f"kind={item.get('first_error_kind')} "
                f"token_category={item.get('token_category')} "
                f"exit_layer={item.get('exit_layer')} "
                f"token={item.get('token_text')} "
                f"pos={item.get('char_position')}"
            )

    early_exit_breaks = report.get("early_exit_parse_break_diagnostics", {})
    if isinstance(early_exit_breaks, dict) and early_exit_breaks:
        print(f"[ok] early-exit breaks : {int(early_exit_breaks.get('failure_count', 0) or 0)}")
        print(
            "[ok] break error kinds : "
            f"{_json_inline(early_exit_breaks.get('first_error_kind_histogram', {}))}"
        )
        print(
            "[ok] break tok cats   : "
            f"{_json_inline(early_exit_breaks.get('first_error_token_category_histogram', {}))}"
        )
        print(
            "[ok] break exit layers: "
            f"{_json_inline(early_exit_breaks.get('first_error_exit_layer_histogram', {}))}"
        )
        for item in early_exit_breaks.get("samples", []):
            if not isinstance(item, dict):
                continue
            print(
                "[ok] early-exit break : "
                f"dataset_index={item.get('dataset_index')} "
                f"kind={item.get('first_error_kind')} "
                f"token_category={item.get('token_category')} "
                f"exit_layer={item.get('exit_layer')} "
                f"token={item.get('token_text')} "
                f"pos={item.get('char_position')}"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )
    accuracy_cfg = _ensure_accuracy_section(merged_config)

    _apply_override(accuracy_cfg, "mode", "local")
    _apply_override(accuracy_cfg, "model_path", args.model_path)
    _apply_override(accuracy_cfg, "tokenizer_path", args.tokenizer_path)
    _apply_override(accuracy_cfg, "test_file", str(args.test_file) if args.test_file else None)
    _apply_override(accuracy_cfg, "report_file", str(args.report_file) if args.report_file else None)
    _apply_override(accuracy_cfg, "num_samples", args.num_samples)
    _apply_override(accuracy_cfg, "seed", args.seed)
    _apply_override(accuracy_cfg, "backend", args.backend)
    _apply_override(accuracy_cfg, "quantization", args.quantization)
    _apply_override(accuracy_cfg, "max_new_tokens", args.max_new_tokens)
    _apply_override(accuracy_cfg, "temperature", args.temperature)
    _apply_override(accuracy_cfg, "exit_layers", args.exit_layers)
    _apply_override(accuracy_cfg, "tau_importance", args.tau_importance)
    _apply_override(accuracy_cfg, "tau_confidence", args.tau_confidence)
    _apply_override(accuracy_cfg, "importance_file", str(args.importance_file) if args.importance_file else None)
    _apply_override(accuracy_cfg, "early_exit_warmup_tokens", args.early_exit_warmup_tokens)
    _apply_override(accuracy_cfg, "early_exit_min_streak", args.early_exit_min_streak)
    _apply_override(accuracy_cfg, "early_exit_draft_only_layers", args.early_exit_draft_only_layers)
    if args.early_exit_fallback_on_invalid_json:
        accuracy_cfg["early_exit_fallback_on_invalid_json"] = True
    if args.disable_early_exit_fallback:
        accuracy_cfg["early_exit_fallback_on_invalid_json"] = False
    if args.early_exit_protect_open_string:
        accuracy_cfg["early_exit_protect_open_string"] = True

    if args.early_exit_enabled:
        accuracy_cfg["early_exit_enabled"] = True
    if args.disable_early_exit:
        accuracy_cfg["early_exit_enabled"] = False

    report_file = Path(
        accuracy_cfg.get("report_file", "experiments/21_exp17_iidee/reports/exp17_iidee_report.json")
    )
    meta_path = record_run_meta(
        report_file.parent,
        merged_config=merged_config,
        cli_args=vars(args),
        argv=sys.argv,
        seed=int(accuracy_cfg.get("seed", 42)),
        data_paths=[
            accuracy_cfg.get("test_file", ""),
            accuracy_cfg.get("dataset_file", ""),
            accuracy_cfg.get("importance_file", ""),
        ],
        extra_meta={
            "entry": "experiments/21_exp17_iidee/run_exp17_iidee.py",
            "stage": "exp17_iidee",
        },
    )
    print(f"[ok] run meta          : {meta_path}")

    report = run_accuracy_from_merged_config(merged_config)
    _print_report(report)


if __name__ == "__main__":
    main()
