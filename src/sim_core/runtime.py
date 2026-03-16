from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.app.app_common import (
    DEFAULT_APP_EXAMPLE_JSON,
    build_interactive_env,
    collect_scene_state,
    preload_local_engine,
    predict_actions_from_instruction,
    print_state,
)
from src.utils.config import get_section, load_config


@dataclass(frozen=True)
class SimRuntimeConfig:
    config_path: Path | None = None
    instruction: str = ""
    action: str = ""
    action_file: Path | None = None
    print_raw: bool = False
    disable_sim_state: bool = False


def _resolve_runtime_config(cfg: SimRuntimeConfig, merged_config: dict[str, Any] | None) -> dict[str, Any]:
    if merged_config is not None:
        return merged_config
    if cfg.config_path is None:
        raise ValueError("config_path is required when merged_config is not provided")
    return load_config(cfg.config_path)


def run_instruction_to_action(
    cfg: SimRuntimeConfig,
    *,
    merged_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_cfg = _resolve_runtime_config(cfg, merged_config)
    app_cfg = get_section(merged_cfg, "app", "interactive")
    show_viewer = bool(app_cfg.get("show_viewer", True))

    state_cfg = get_section(merged_cfg, "app", "state_injection")
    use_sim_state = bool(state_cfg.get("enable_instruction_to_action", True))
    if cfg.disable_sim_state:
        use_sim_state = False

    manager = None
    try:
        if use_sim_state:
            manager, _ = build_interactive_env(show_viewer=show_viewer, cfg=merged_cfg)

        scene_state = collect_scene_state(manager) if manager is not None else None
        raw, payload = predict_actions_from_instruction(
            cfg.instruction.strip(),
            merged_cfg,
            scene_state=scene_state,
        )
        return {
            "raw": raw,
            "payload": payload,
            "scene_state": scene_state,
            "example": DEFAULT_APP_EXAMPLE_JSON,
        }
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


def run_action_to_motion(
    cfg: SimRuntimeConfig,
    *,
    merged_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_cfg = _resolve_runtime_config(cfg, merged_config)
    app_cfg = get_section(merged_cfg, "app", "interactive")
    enabled = bool(app_cfg.get("enabled", True))
    show_viewer = bool(app_cfg.get("show_viewer", True))
    if not enabled:
        return {"disabled": True}

    raw_action = cfg.action.strip()
    if cfg.action_file is not None:
        raw_action = cfg.action_file.read_text(encoding="utf-8").strip()
    if not raw_action:
        raise ValueError("action text is empty")

    manager = None
    try:
        manager, robot = build_interactive_env(show_viewer=show_viewer, cfg=merged_cfg)
        results = robot.execute_json(raw_action)
        return {"disabled": False, "results": results}
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


def run_instruction_to_motion(
    cfg: SimRuntimeConfig,
    *,
    merged_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_cfg = _resolve_runtime_config(cfg, merged_config)
    app_cfg = get_section(merged_cfg, "app", "interactive")
    enabled = bool(app_cfg.get("enabled", True))
    show_viewer = bool(app_cfg.get("show_viewer", True))
    if not enabled:
        return {"disabled": True}

    state_cfg = get_section(merged_cfg, "app", "state_injection")
    use_sim_state = bool(state_cfg.get("enable_instruction_to_motion", True))
    if cfg.disable_sim_state:
        use_sim_state = False

    manager = None
    try:
        manager, robot = build_interactive_env(show_viewer=show_viewer, cfg=merged_cfg)
        scene_state = collect_scene_state(robot.manager) if use_sim_state else None
        raw, payload = predict_actions_from_instruction(
            cfg.instruction.strip(),
            merged_cfg,
            scene_state=scene_state,
        )
        results = robot.execute_json(payload)
        return {
            "disabled": False,
            "raw": raw,
            "payload": payload,
            "scene_state": scene_state,
            "results": results,
        }
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


def run_interactive_session(config_path: Path) -> None:
    merged_cfg = load_config(config_path)
    app_cfg = get_section(merged_cfg, "app", "interactive")
    enabled = bool(app_cfg.get("enabled", True))
    show_viewer = bool(app_cfg.get("show_viewer", True))
    if not enabled:
        print("[app] disabled by config. Set app.interactive.enabled=true to run.")
        return

    print("Starting Genesis interactive control...")
    manager = None
    try:
        manager, robot = build_interactive_env(show_viewer=show_viewer, cfg=merged_cfg)
        print("/help /state /scene /example /quit")
        print(DEFAULT_APP_EXAMPLE_JSON)

        while True:
            raw = input("\n>>> ").strip()
            if not raw:
                continue
            if raw == "/quit":
                break
            if raw == "/help":
                print("/help /state /scene /example /quit")
                continue
            if raw == "/example":
                print(DEFAULT_APP_EXAMPLE_JSON)
                continue
            if raw == "/state":
                print_state(robot)
                continue
            if raw == "/scene":
                entities = robot.manager.get_entities()
                print("[scene_entities]", sorted(entities.keys()))
                for name in sorted(entities.keys()):
                    try:
                        st = robot.manager.get_entity_state(name)
                    except Exception:
                        st = {}
                    print(f"- {name}: {json.dumps(st, ensure_ascii=False)}")
                continue
            try:
                results = robot.execute_json(raw)
                print("[ok] Executed commands:")
                for item in results:
                    print(
                        f"  - idx={item.get('index')} action={item.get('action')} status={item.get('status')}"
                    )
            except Exception as exc:
                print(f"[error] {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)


def run_model_interactive_session(
    config_path: Path | None = None,
    *,
    merged_config: dict[str, Any] | None = None,
    disable_sim_state: bool = False,
) -> None:
    if merged_config is not None:
        merged_cfg = merged_config
    else:
        if config_path is None:
            raise ValueError("config_path is required when merged_config is not provided")
        merged_cfg = load_config(config_path)
    app_cfg = get_section(merged_cfg, "app", "interactive")
    enabled = bool(app_cfg.get("enabled", True))
    show_viewer = bool(app_cfg.get("show_viewer", True))
    if not enabled:
        print("[app] disabled by config. Set app.interactive.enabled=true to run.")
        return

    state_cfg = get_section(merged_cfg, "app", "state_injection")
    use_sim_state = bool(state_cfg.get("enable_instruction_to_motion", True))
    if disable_sim_state:
        use_sim_state = False
    inference_cfg = get_section(merged_cfg, "app", "inference")
    inference_mode = str(inference_cfg.get("mode", "api")).strip().lower()

    print("Starting Genesis model-driven interactive control...")
    manager = None
    try:
        if inference_mode == "local":
            print("[app] preloading local inference engine...")
            try:
                preload_local_engine(merged_cfg, warmup=True)
                print("[app] local inference engine ready.")
            except Exception:
                # Preload failure should not prevent interactive use; the actual
                # inference path will raise a more specific error if needed.
                pass

        manager, robot = build_interactive_env(show_viewer=show_viewer, cfg=merged_cfg)
        print("/help /state /scene /example /raw on|off /quit")
        print("Enter natural-language instructions and the model will generate + execute actions.")
        print(DEFAULT_APP_EXAMPLE_JSON)

        print_raw = False
        while True:
            raw = input("\n>>> ").strip()
            if not raw:
                continue
            if raw == "/quit":
                break
            if raw == "/help":
                print("/help /state /scene /example /raw on|off /quit")
                continue
            if raw == "/example":
                print(DEFAULT_APP_EXAMPLE_JSON)
                continue
            if raw == "/state":
                print_state(robot)
                continue
            if raw == "/scene":
                entities = robot.manager.get_entities()
                print("[scene_entities]", sorted(entities.keys()))
                for name in sorted(entities.keys()):
                    try:
                        st = robot.manager.get_entity_state(name)
                    except Exception:
                        st = {}
                    print(f"- {name}: {json.dumps(st, ensure_ascii=False)}")
                continue
            if raw.startswith("/raw"):
                parts = raw.split(maxsplit=1)
                if len(parts) == 2 and parts[1] in {"on", "off"}:
                    print_raw = parts[1] == "on"
                    print(f"[app] raw model output {'enabled' if print_raw else 'disabled'}.")
                else:
                    print("[app] usage: /raw on|off")
                continue
            try:
                scene_state = collect_scene_state(robot.manager) if use_sim_state else None
                model_raw, payload = predict_actions_from_instruction(
                    raw,
                    merged_cfg,
                    scene_state=scene_state,
                )
                if print_raw:
                    print("[model_raw]")
                    print(model_raw)
                if scene_state is not None:
                    print("[scene_state]")
                    print(json.dumps(scene_state, ensure_ascii=False, indent=2))
                print("[action_json]")
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                results = robot.execute_json(payload)
                print("[ok] Executed commands:")
                for item in results:
                    print(
                        f"  - idx={item.get('index')} action={item.get('action')} status={item.get('status')}"
                    )
            except Exception as exc:
                print(f"[error] {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)
