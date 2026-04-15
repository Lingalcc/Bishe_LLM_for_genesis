from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_merged_config


def _load_exp16_module():
    module_path = REPO_ROOT / "experiments" / "20_exp16_genesis_show" / "run_exp16_genesis_show.py"
    spec = importlib.util.spec_from_file_location("exp16_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_exp16_module()
    merged = load_merged_config(
        base_config_path=REPO_ROOT / "configs" / "base.yaml",
        override_config_path=REPO_ROOT / "experiments" / "17_exp13_sim_success" / "configs" / "sim_success.yaml",
    )

    output_root = REPO_ROOT / "tmp" / "exp13_case_gallery"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "case12_stats_ok_exec_fail",
            "首先将夹爪完全打开，然后移动末端到x=0.3m，y=0.4m，z=0.2m的位置，接着将关节2和4以-0.2rad/s的速度移动",
        ),
        (
            "case5_stats_low_exec_ok",
            "将机械臂移动到工作台中央上方40厘米处，保持竖直向下姿态",
        ),
        (
            "case14_stats_low_exec_ok",
            "将机械臂移动到工作台中心上方30厘米，然后向下移动到距离工作台10厘米处",
        ),
        (
            "case1_stats_low_exec_ok",
            "将机械臂关节1和3分别设置为-0.3和0.5弧度，等待2秒，然后获取当前状态",
        ),
    ]

    for name, instruction in cases:
        session_dir = output_root / name
        snap_dir = session_dir / "snaps"
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[case] {name}", flush=True)
        report_path = module.run_single_demo(
            merged_config=merged,
            instruction=instruction,
            print_raw=False,
            disable_sim_state=False,
            session_dir=session_dir,
            snap_dir=snap_dir,
        )
        print(f"[report] {report_path}", flush=True)


if __name__ == "__main__":
    main()
