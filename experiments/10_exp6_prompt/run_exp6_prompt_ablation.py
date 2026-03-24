#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_core.accuracy import run_accuracy_from_merged_config
from src.utils.config import load_merged_config
from src.utils.run_meta import record_run_meta
from src.utils.secrets import safe_json_dumps

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "compare.yaml"
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验10 Exp6：Prompt 对照消融实验。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="实验配置文件。")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG_PATH, help="基础配置文件。")
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="可选，指定要运行的 Prompt 版本，多个名称用逗号分隔，如 baseline,optimized。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验配置与 Prompt，不真正执行评测。")
    return parser.parse_args()


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path.resolve())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_variants(config: dict[str, Any], requested: set[str]) -> list[dict[str, Any]]:
    section = config.get("prompt_experiment", {})
    if not isinstance(section, dict):
        raise ValueError("配置缺少 prompt_experiment 段。")

    raw_variants = section.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("prompt_experiment.variants 必须是非空列表。")

    variants: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in raw_variants:
        if not isinstance(item, dict):
            raise TypeError("prompt_experiment.variants 的每一项都必须是对象。")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("存在未命名的 Prompt 变体。")
        if name in seen_names:
            raise ValueError(f"Prompt 变体名称重复: {name}")
        seen_names.add(name)
        if requested and name not in requested:
            continue

        prompt_file = _resolve_repo_path(str(item.get("prompt_file", "")).strip())
        report_file = _resolve_repo_path(str(item.get("report_file", "")).strip())
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_file}")

        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Prompt 文件为空: {prompt_file}")

        variants.append(
            {
                "name": name,
                "label": str(item.get("label", name)).strip() or name,
                "prompt_file": prompt_file,
                "prompt_text": prompt_text,
                "prompt_sha256": _sha256_text(prompt_text),
                "report_file": report_file,
            }
        )

    if requested and not variants:
        raise ValueError(f"未匹配到任何指定 variants: {sorted(requested)}")
    if not variants:
        raise ValueError("没有可执行的 Prompt 变体。")
    return variants


def _prepare_variant_config(merged_config: dict[str, Any], *, system_prompt: str, report_file: Path) -> dict[str, Any]:
    variant_cfg = copy.deepcopy(merged_config)
    test_section = variant_cfg.setdefault("test", {})
    if not isinstance(test_section, dict):
        raise TypeError("配置中的 test 段必须是对象。")
    acc_section = test_section.setdefault("accuracy_eval", {})
    if not isinstance(acc_section, dict):
        raise TypeError("配置中的 test.accuracy_eval 段必须是对象。")
    acc_section["system_prompt"] = system_prompt
    acc_section["report_file"] = _repo_relative(report_file)
    return variant_cfg


def _build_summary_md(
    *,
    dataset_file: Path,
    mode: str,
    model_desc: str,
    results: list[dict[str, Any]],
    delta: dict[str, Any] | None,
) -> str:
    lines = [
        "# 实验10 Exp6：Prompt 对照消融结果",
        "",
        f"- 测试集：`{_repo_relative(dataset_file)}`",
        f"- 评测模式：`{mode}`",
        f"- 模型：`{model_desc}`",
        "",
        "| Variant | Parse OK | Exact Match | Action Match | Samples | Report |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            "| {label} | {parse:.4f} | {exact:.4f} | {action:.4f} | {samples} | `{report}` |".format(
                label=item["label"],
                parse=item["parse_ok_rate"],
                exact=item["exact_match_rate"],
                action=item["action_match_rate"],
                samples=item["num_samples_evaluated"],
                report=item["report_file"],
            )
        )

    if delta is not None:
        lines.extend(
            [
                "",
                "## Optimized 相对 Baseline 的变化",
                "",
                f"- `exact_match_rate`: {delta['exact_match_rate_delta']:+.4f}",
                f"- `action_match_rate`: {delta['action_match_rate_delta']:+.4f}",
                f"- `parse_ok_rate`: {delta['parse_ok_rate_delta']:+.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    requested = {name.strip() for name in args.variants.split(",") if name.strip()}
    merged_config = load_merged_config(
        base_config_path=args.base_config,
        override_config_path=args.config if args.config.exists() else None,
    )

    variants = _load_variants(merged_config, requested)
    exp_section = merged_config.get("prompt_experiment", {})
    if not isinstance(exp_section, dict):
        raise ValueError("配置缺少 prompt_experiment 段。")

    summary_json_path = _resolve_repo_path(exp_section.get("summary_json", "experiments/10_exp6_prompt/reports/prompt_ablation_summary.json"))
    summary_md_path = _resolve_repo_path(exp_section.get("summary_md", "experiments/10_exp6_prompt/reports/prompt_ablation_summary.md"))
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)

    acc_section = merged_config.get("test", {}).get("accuracy_eval", {})
    if not isinstance(acc_section, dict):
        raise ValueError("配置缺少 test.accuracy_eval 段。")
    dataset_file = _resolve_repo_path(acc_section.get("test_file") or acc_section.get("dataset_file"))
    mode = str(acc_section.get("mode", "api"))
    model_desc = (
        str(acc_section.get("model_path", "")).strip()
        if mode == "local"
        else str(acc_section.get("model", "")).strip()
    ) or ("local_model" if mode == "local" else "api_model")

    meta_path = record_run_meta(
        summary_json_path.parent,
        merged_config=merged_config,
        cli_args=vars(args),
        argv=sys.argv,
        seed=(int(acc_section["seed"]) if acc_section.get("seed") is not None else None),
        data_paths=[dataset_file, args.config, args.base_config, *[item["prompt_file"] for item in variants]],
        extra_meta={
            "entry": "experiments/10_exp6_prompt/run_exp6_prompt_ablation.py",
            "stage": "prompt_ablation",
            "variants": [item["name"] for item in variants],
        },
    )
    print(f"[ok] run meta          : {meta_path}")

    if args.dry_run:
        print(f"[ok] dataset           : {_repo_relative(dataset_file)}")
        for item in variants:
            print(
                f"[ok] variant           : {item['name']}  "
                f"prompt={_repo_relative(item['prompt_file'])}  sha256={item['prompt_sha256'][:12]}"
            )
        print("[ok] dry-run completed : 配置、Prompt 与输出路径校验通过")
        return

    result_rows: list[dict[str, Any]] = []
    for item in variants:
        print(f"[info] running variant : {item['name']}")
        variant_cfg = _prepare_variant_config(
            merged_config,
            system_prompt=item["prompt_text"],
            report_file=item["report_file"],
        )
        report = run_accuracy_from_merged_config(variant_cfg)
        row = {
            "name": item["name"],
            "label": item["label"],
            "prompt_file": _repo_relative(item["prompt_file"]),
            "prompt_sha256": item["prompt_sha256"],
            "report_file": _repo_relative(item["report_file"]),
            "num_samples_evaluated": int(report["num_samples_evaluated"]),
            "parse_ok": int(report["parse_ok"]),
            "parse_ok_rate": float(report["parse_ok_rate"]),
            "exact_match": int(report["exact_match"]),
            "exact_match_rate": float(report["exact_match_rate"]),
            "action_match": int(report["action_match"]),
            "action_match_rate": float(report["action_match_rate"]),
        }
        result_rows.append(row)
        print(
            f"[ok] {item['name']:<17} parse_ok={row['parse_ok_rate']:.4f}  "
            f"exact_match={row['exact_match_rate']:.4f}  action_match={row['action_match_rate']:.4f}"
        )

    best_variant = max(result_rows, key=lambda x: (x["exact_match_rate"], x["action_match_rate"], x["parse_ok_rate"]))
    rows_by_name = {item["name"]: item for item in result_rows}
    delta: dict[str, Any] | None = None
    if "baseline" in rows_by_name and "optimized" in rows_by_name:
        baseline = rows_by_name["baseline"]
        optimized = rows_by_name["optimized"]
        delta = {
            "baseline": baseline["label"],
            "optimized": optimized["label"],
            "exact_match_rate_delta": optimized["exact_match_rate"] - baseline["exact_match_rate"],
            "action_match_rate_delta": optimized["action_match_rate"] - baseline["action_match_rate"],
            "parse_ok_rate_delta": optimized["parse_ok_rate"] - baseline["parse_ok_rate"],
        }

    summary = {
        "experiment": "exp6_prompt_ablation",
        "dataset_file": _repo_relative(dataset_file),
        "mode": mode,
        "model": model_desc,
        "variants": result_rows,
        "best_variant": {
            "name": best_variant["name"],
            "label": best_variant["label"],
            "exact_match_rate": best_variant["exact_match_rate"],
        },
        "delta_optimized_vs_baseline": delta,
    }
    summary_json_path.write_text(safe_json_dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        _build_summary_md(
            dataset_file=dataset_file,
            mode=mode,
            model_desc=model_desc,
            results=result_rows,
            delta=delta,
        ),
        encoding="utf-8",
    )

    print(f"[ok] summary json      : {summary_json_path}")
    print(f"[ok] summary md        : {summary_md_path}")
    print(f"[ok] best variant      : {best_variant['label']} ({best_variant['exact_match_rate']:.4f})")


if __name__ == "__main__":
    main()
