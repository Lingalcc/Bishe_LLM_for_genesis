#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_compare_module():
    script_path = Path(__file__).with_name("run_exp2_peft_compare.py")
    spec = importlib.util.spec_from_file_location("exp3_compare", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载实验脚本: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    compare = load_compare_module()
    compare.ensure_runtime_dirs()

    result = compare.run_single_method("galore")

    csv_path = compare.RESULTS_CSV_PATH
    if csv_path.exists():
        df = compare.pd.read_csv(csv_path)
        if "Method" in df.columns and (df["Method"] == "galore").any():
            df.loc[df["Method"] == "galore", list(result.keys())] = list(result.values())
        else:
            df = compare.pd.concat([df, compare.pd.DataFrame([result])], ignore_index=True)
    else:
        df = compare.pd.DataFrame([result])

    method_order = {name: index for index, name in enumerate(compare.METHODS)}
    if "Method" in df.columns:
        df = df.sort_values(
            by="Method",
            key=lambda column: column.map(lambda value: method_order.get(str(value), len(method_order))),
        ).reset_index(drop=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[result] 已回写结果到: {csv_path}", flush=True)
    print(df.to_string(index=False), flush=True)

    compare.plot_all_figures(csv_path)


if __name__ == "__main__":
    main()
