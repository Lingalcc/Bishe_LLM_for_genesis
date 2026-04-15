from __future__ import annotations

import copy
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.app_common import build_interactive_env, collect_scene_state, preload_local_engine, predict_actions_from_instruction
from src.utils.config import load_merged_config
from src.utils.secrets import safe_json_dumps


def _load_exp16_module():
    module_path = Path("experiments/20_exp16_genesis_show/run_exp16_genesis_show.py").resolve()
    spec = importlib.util.spec_from_file_location("exp16_debug_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_exp16_module()
    instructions = [
        "打开夹爪",
        "关闭夹爪",
        "移动到方块上方10厘米处",
        "移动到方块上方10厘米处，夹爪朝下",
        "先张开夹爪，再移动到工作台中央上方30厘米处",
        "读取当前机械臂状态",
        "移动到方块上方，下降到方块高度后闭合夹爪，再抬起到更高位置",
        "移动到方块上方，张开夹爪，下降后夹住，再抬起",
    ]

    merged_config = load_merged_config(
        base_config_path=module.DEFAULT_BASE_CONFIG,
        override_config_path=module.DEFAULT_CONFIG,
    )
    merged_config.setdefault("app", {}).setdefault("interactive", {})["show_viewer"] = False

    args = module.parse_args([])
    skip = module._resolve_skip_vllm_compat_check(merged_config, args)
    module._apply_vllm_compat_policy(skip)

    out_root = Path("experiments/20_exp16_genesis_show/reports") / f"debug_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snaps_dir = out_root / "snaps"
    out_root.mkdir(parents=True, exist_ok=True)
    snaps_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT {out_root}", flush=True)

    print("Preloading engine...", flush=True)
    preload_local_engine(merged_config, warmup=True)
    print("Engine ready.", flush=True)

    summary: list[dict[str, object]] = []
    for idx, instruction in enumerate(instructions, start=1):
        print(f"RUN {idx}: {instruction}", flush=True)
        manager = None
        try:
            manager, robot = build_interactive_env(
                show_viewer=False,
                cfg=copy.deepcopy(merged_config),
                debug_camera_name=module.DEFAULT_SNAP_CAMERA_NAME,
                debug_camera_options=module.DEFAULT_SNAP_CAMERA_OPTIONS,
            )
            scene_before = collect_scene_state(robot.manager)
            model_raw, payload = predict_actions_from_instruction(
                instruction,
                merged_config,
                scene_state=scene_before,
            )
            results = robot.execute_json(payload)
            scene_after = collect_scene_state(robot.manager)
            snap_path = manager.save_camera_rgb(
                module.DEFAULT_SNAP_CAMERA_NAME,
                snaps_dir / f"turn_{idx:03d}.png",
            )
            report = module._build_turn_report(
                turn_index=idx,
                instruction=instruction,
                scene_state=scene_before,
                scene_state_after=scene_after,
                model_raw=model_raw,
                payload=payload,
                results=results,
                elapsed_sec=0.0,
                snap_path=str(snap_path),
            )
            report_path = module._save_turn_report(out_root, report)
            entities_before = {item["name"]: item["state"] for item in scene_before.get("entities", [])}
            entities_after = {item["name"]: item["state"] for item in scene_after.get("entities", [])}
            item = {
                "turn_index": idx,
                "instruction": instruction,
                "report_path": str(report_path),
                "snap_path": str(snap_path),
                "execution_success": report["execution_success"],
                "actions": [cmd.get("action") for cmd in payload.get("commands", [])],
                "cube_before": entities_before.get("cube", {}).get("pos"),
                "cube_after": entities_after.get("cube", {}).get("pos"),
            }
            summary.append(item)
            print(safe_json_dumps(item, ensure_ascii=False), flush=True)
        except Exception as exc:
            error_report = module._build_turn_report(
                turn_index=idx,
                instruction=instruction,
                scene_state=None,
                scene_state_after=None,
                model_raw="",
                payload={},
                results=[],
                elapsed_sec=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
            report_path = module._save_turn_report(out_root, error_report)
            item = {
                "turn_index": idx,
                "instruction": instruction,
                "report_path": str(report_path),
                "snap_path": None,
                "execution_success": False,
                "actions": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary.append(item)
            print(safe_json_dumps(item, ensure_ascii=False), flush=True)
        finally:
            if manager is not None:
                manager.release(destroy_runtime=True)

    summary_path = out_root / "summary.json"
    summary_path.write_text(safe_json_dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY {summary_path}", flush=True)


if __name__ == "__main__":
    main()
