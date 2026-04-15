#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from src.app.local_llm_engine import LocalLLMEngine
from src.genesis.sim_runtime import DEFAULT_FRANKA_MJCF, preflight_sim_environment
from src.protocols.toolcall import extract_first_json, normalize_payload, validate_payload
from src.utils.config import get_section
from src.utils.secrets import MissingSecretError, redact_text, resolve_api_key_from_env


DEFAULT_APP_SYSTEM_PROMPT = (
    "你是 Franka 机械臂控制指令生成器。"
    "请把用户自然语言转换为可执行的 JSON action。"
    "如果输入中包含[STATE_CONTEXT]...[/STATE_CONTEXT]，你必须利用其中的物体名字、状态、坐标和姿态进行决策。"
    "只输出 JSON，不要输出解释。"
)

DEFAULT_APP_EXAMPLE_JSON = """{
  "commands": [
    {"action": "open_gripper", "position": 0.04},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.18], "quat": [0, 1, 0, 0]},
    {"action": "close_gripper", "position": 0.0},
    {"action": "wait"},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.30], "quat": [0, 1, 0, 0]}
  ]
}"""

_LOCAL_ENGINE_CACHE: dict[str, LocalLLMEngine] = {}
DEFAULT_DOWNWARD_GRASP_QUAT = [0.0, 1.0, 0.0, 0.0]


def extract_first_json_from_text(text: str) -> Any:
    return extract_first_json(text)


def normalize_action_payload(payload: Any) -> dict[str, Any]:
    normalized = normalize_payload(payload)
    validate_payload(normalized, policy="execution")
    return normalized


def _instruction_requires_downward_grasp_pose(instruction: str) -> bool:
    text = instruction.strip()
    if not text:
        return False
    if "朝下" in text or "夹爪朝下" in text:
        return True
    return "方块" in text and "上方" in text


def _apply_demo_pose_overrides(instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not _instruction_requires_downward_grasp_pose(instruction):
        return payload

    commands = payload.get("commands")
    if not isinstance(commands, list):
        return payload

    adjusted = False
    patched_commands: list[dict[str, Any]] = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            patched_commands.append(cmd)
            continue
        patched = dict(cmd)
        if patched.get("action") == "move_ee":
            patched["quat"] = list(DEFAULT_DOWNWARD_GRASP_QUAT)
            adjusted = True
        patched_commands.append(patched)

    if not adjusted:
        return payload
    return {"commands": patched_commands}


def collect_scene_state(manager: Any) -> dict[str, Any]:
    entities = manager.get_entities()
    snapshots: list[dict[str, Any]] = []
    for name in sorted(entities.keys()):
        try:
            params = manager.get_entity_params(name, include_state=False)
        except Exception:
            params = {"name": name}
        try:
            runtime_state = manager.get_entity_state(name)
        except Exception:
            runtime_state = {}
        snapshots.append(
            {
                "name": name,
                "category": params.get("category", ""),
                "entity_class": params.get("entity_class", ""),
                "state": runtime_state,
            }
        )
    return {"entities": snapshots}


def build_state_context_text(scene_state: dict[str, Any]) -> str:
    state_text = json.dumps(scene_state, ensure_ascii=False, separators=(",", ":"))
    return f"[STATE_CONTEXT]{state_text}[/STATE_CONTEXT]"


def inject_state_into_instruction(instruction: str, scene_state: dict[str, Any] | None) -> str:
    if not scene_state:
        return instruction
    return f"{build_state_context_text(scene_state)}\n用户指令: {instruction}"


def call_chat_completions(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    timeout: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if frequency_penalty is not None:
        payload["frequency_penalty"] = float(frequency_penalty)
    if presence_penalty is not None:
        payload["presence_penalty"] = float(presence_penalty)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    choices = obj.get("choices", [])
    if not choices:
        raise ValueError("empty choices from API")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content from API")
    return content


def _resolve_inference_config(cfg: dict[str, Any]) -> dict[str, Any]:
    inf_cfg = get_section(cfg, "app", "inference")
    if inf_cfg:
        return inf_cfg

    # Backward compatibility: map legacy `app.model` config into `app.inference`.
    legacy_model_cfg = get_section(cfg, "app", "model")
    return {
        "mode": "api",
        "system_prompt": str(legacy_model_cfg.get("system_prompt", DEFAULT_APP_SYSTEM_PROMPT)),
        "max_retries": int(legacy_model_cfg.get("max_retries", 3)),
        "sleep_seconds": float(legacy_model_cfg.get("sleep_seconds", 0.0)),
        "api": {
            "api_base": str(legacy_model_cfg.get("api_base", "https://api.openai.com/v1")),
            "model": str(legacy_model_cfg.get("model", "gpt-5")),
            "api_key": str(legacy_model_cfg.get("api_key", "")),
            "api_key_env": "OPENAI_API_KEY",
            "timeout": int(legacy_model_cfg.get("timeout", 120)),
            "generation": {
                "temperature": float(legacy_model_cfg.get("temperature", 0.0)),
                "max_tokens": int(legacy_model_cfg.get("max_tokens", 1200)),
            },
        },
        "local": {},
    }


def _resolve_api_key(api_cfg: dict[str, Any]) -> str:
    try:
        return resolve_api_key_from_env(
            api_key=str(api_cfg.get("api_key", "")),
            api_key_env=str(api_cfg.get("api_key_env", "OPENAI_API_KEY")),
            default_env="OPENAI_API_KEY",
            source_name="Interactive app API",
        )
    except MissingSecretError as exc:
        raise RuntimeError(str(exc)) from exc


def _build_local_prompt_prefix(system_prompt: str, scene_state: dict[str, Any] | None = None) -> str:
    prefix = f"{system_prompt}\n\n请严格输出 JSON，不要附带解释。\n"
    if scene_state:
        return f"{prefix}{build_state_context_text(scene_state)}\n用户指令: "
    return f"{prefix}用户输入：\n"


def _build_local_prompt(
    system_prompt: str,
    user_prompt: str,
    scene_state: dict[str, Any] | None = None,
) -> str:
    return f"{_build_local_prompt_prefix(system_prompt, scene_state)}{user_prompt}\n"


def _get_local_engine(local_cfg: dict[str, Any]) -> LocalLLMEngine:
    key = json.dumps(local_cfg, ensure_ascii=False, sort_keys=True)
    cached = _LOCAL_ENGINE_CACHE.get(key)
    if cached is not None:
        return cached

    gen_cfg = local_cfg.get("generation", {})
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}

    model_path = str(local_cfg.get("model_path", "")).strip()
    if not model_path:
        raise RuntimeError("app.inference.local.model_path is required in local mode.")

    quantization_value = local_cfg.get("quantization")
    quantization = None
    if quantization_value is not None:
        text = str(quantization_value).strip()
        quantization = text if text else None

    engine = LocalLLMEngine(
        model_path=model_path,
        backend=str(local_cfg.get("backend", "auto")).strip().lower() or "auto",
        quantization=quantization,  # type: ignore[arg-type]
        max_model_len=int(local_cfg.get("max_model_len", 4096)),
        gpu_memory_utilization=float(local_cfg.get("gpu_memory_utilization", 0.9)),
        trust_remote_code=bool(local_cfg.get("trust_remote_code", True)),
        default_temperature=float(gen_cfg.get("temperature", 0.0)),
        default_top_p=float(gen_cfg.get("top_p", 1.0)),
        default_max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
        enable_prefix_caching=bool(local_cfg.get("enable_prefix_caching", False)),
        prefix_cache_max_entries=int(local_cfg.get("prefix_cache_max_entries", 4)),
    )
    _LOCAL_ENGINE_CACHE[key] = engine
    return engine


def preload_local_engine(cfg: dict[str, Any], *, warmup: bool = False) -> LocalLLMEngine | None:
    inf_cfg = _resolve_inference_config(cfg)
    mode = str(inf_cfg.get("mode", "api")).strip().lower()
    if mode != "local":
        return None

    local_cfg = inf_cfg.get("local", {})
    if not isinstance(local_cfg, dict):
        raise TypeError("app.inference.local must be a mapping object")

    engine = _get_local_engine(local_cfg)
    if warmup:
        gen_cfg = local_cfg.get("generation", {})
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}
        system_prompt = str(inf_cfg.get("system_prompt", DEFAULT_APP_SYSTEM_PROMPT))
        prefix_prompt = _build_local_prompt_prefix(system_prompt)
        engine.warm_prefix(prefix_prompt, cache_key="app::default_prompt_prefix")
        engine.generate_with_prefix(
            prefix_prompt,
            "输出一个最短 JSON：{\"commands\":[{\"action\":\"wait\"}]}\n",
            temperature=float(gen_cfg.get("temperature", 0.0)),
            top_p=float(gen_cfg.get("top_p", 1.0)),
            max_new_tokens=min(64, int(gen_cfg.get("max_new_tokens", 128))),
            cache_key="app::default_prompt_prefix",
        )
    return engine


def predict_actions_from_instruction(
    instruction: str,
    cfg: dict[str, Any],
    *,
    scene_state: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    inf_cfg = _resolve_inference_config(cfg)
    mode = str(inf_cfg.get("mode", "api")).strip().lower()
    max_retries = int(inf_cfg.get("max_retries", 3))
    sleep_seconds = float(inf_cfg.get("sleep_seconds", 0.0))
    system_prompt = str(inf_cfg.get("system_prompt", DEFAULT_APP_SYSTEM_PROMPT))
    if max_retries <= 0:
        raise ValueError("app.inference.max_retries must be > 0")

    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            if mode == "api":
                prompt_text = inject_state_into_instruction(instruction, scene_state)
                api_cfg = inf_cfg.get("api", {})
                if not isinstance(api_cfg, dict):
                    raise TypeError("app.inference.api must be a mapping object")

                gen_cfg = api_cfg.get("generation", {})
                if not isinstance(gen_cfg, dict):
                    gen_cfg = {}

                api_key = _resolve_api_key(api_cfg)

                raw = call_chat_completions(
                    api_base=str(api_cfg.get("api_base", "https://api.openai.com/v1")),
                    api_key=api_key,
                    model=str(api_cfg.get("model", "gpt-5")),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=float(gen_cfg.get("temperature", 0.0)),
                    max_tokens=int(gen_cfg.get("max_tokens", 1200)),
                    top_p=(
                        float(gen_cfg["top_p"]) if gen_cfg.get("top_p") is not None else None
                    ),
                    frequency_penalty=(
                        float(gen_cfg["frequency_penalty"])
                        if gen_cfg.get("frequency_penalty") is not None
                        else None
                    ),
                    presence_penalty=(
                        float(gen_cfg["presence_penalty"])
                        if gen_cfg.get("presence_penalty") is not None
                        else None
                    ),
                    timeout=int(api_cfg.get("timeout", 120)),
                )
            elif mode == "local":
                local_cfg = inf_cfg.get("local", {})
                if not isinstance(local_cfg, dict):
                    raise TypeError("app.inference.local must be a mapping object")
                gen_cfg = local_cfg.get("generation", {})
                if not isinstance(gen_cfg, dict):
                    gen_cfg = {}

                engine = _get_local_engine(local_cfg)
                prefix_prompt = _build_local_prompt_prefix(system_prompt, scene_state)
                raw = engine.generate_with_prefix(
                    prefix_prompt,
                    f"{instruction}\n",
                    temperature=(
                        float(gen_cfg["temperature"]) if gen_cfg.get("temperature") is not None else None
                    ),
                    top_p=float(gen_cfg["top_p"]) if gen_cfg.get("top_p") is not None else None,
                    max_new_tokens=(
                        int(gen_cfg["max_new_tokens"])
                        if gen_cfg.get("max_new_tokens") is not None
                        else None
                    ),
                )
            else:
                raise ValueError("app.inference.mode must be 'api' or 'local'")

            payload = normalize_action_payload(extract_first_json_from_text(raw))
            payload = _apply_demo_pose_overrides(instruction, payload)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return raw, payload
        except (ValueError, json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError) as err:
            last_err = err
            time.sleep(min(8.0, 0.8 * (2**i)))

    raise RuntimeError(f"model prediction failed: {redact_text(str(last_err))}")


def _resolve_sim_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    sim_cfg = get_section(cfg, "app", "sim")
    return sim_cfg if isinstance(sim_cfg, dict) else {}


def build_interactive_env(
    show_viewer: bool = True,
    *,
    cfg: dict[str, Any] | None = None,
    debug_camera_name: str | None = None,
    debug_camera_options: dict[str, Any] | None = None,
):
    sim_cfg = _resolve_sim_config(cfg)
    backend = str(sim_cfg.get("backend", "gpu")).strip() or "gpu"
    robot_file = str(sim_cfg.get("robot_file", DEFAULT_FRANKA_MJCF)).strip() or DEFAULT_FRANKA_MJCF
    robot_type = str(sim_cfg.get("robot_type", "mjcf")).strip().lower() or "mjcf"
    genesis_repo = sim_cfg.get("genesis_repo")
    asset_root = sim_cfg.get("asset_root")
    gravity_cfg = sim_cfg.get("gravity", (0.0, 0.0, -9.81))
    gravity = (0.0, 0.0, -9.81)
    if isinstance(gravity_cfg, (list, tuple)) and len(gravity_cfg) == 3:
        gravity = (float(gravity_cfg[0]), float(gravity_cfg[1]), float(gravity_cfg[2]))

    preflight = preflight_sim_environment(
        robot_file=robot_file,
        robot_type=robot_type,
        genesis_repo=genesis_repo,
        asset_root=asset_root,
    )
    import genesis as gs
    from src.genesis.genesis_tools import GenesisManager, GenesisRobot

    manager = GenesisManager(
        backend=backend,
        init_kwargs={"precision": "32", "logging_level": "warning"},
        scene_kwargs={
            "show_viewer": show_viewer,
            "sim_options": gs.options.SimOptions(dt=0.01, gravity=gravity),
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(2.5, -1.6, 1.4),
                camera_lookat=(0.5, 0.0, 0.3),
                camera_fov=35,
                res=(1280, 720),
                max_FPS=60,
            ),
        },
        auto_init=True,
        auto_create_scene=True,
        force_reinit=True,
    )

    manager.add_plane("ground")
    manager.add_robot_from_file(
        name="franka",
        file=str(preflight.resolved_robot_file),
        robot_type=robot_type,
    )
    manager.add_object(
        name="cube",
        morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
    )
    if debug_camera_name:
        camera_options = dict(debug_camera_options or {})
        manager.add_camera(debug_camera_name, **camera_options)
    manager.build_scene()
    manager.step(1)

    robot = GenesisRobot(
        manager=manager,
        robot_name="franka",
        ee_link_name="hand",
        arm_dofs_idx_local=list(range(7)),
        gripper_dofs_idx_local=[7, 8],
        default_interp_steps=120,
        default_command_steps=120,
        default_init_qpos=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
        default_init_steps=150,
        default_hold_kp=[4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
        default_hold_kv=[450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
        default_hold_force_lower=[-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0],
        default_hold_force_upper=[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0],
        default_hold_steps=90,
    )
    robot.move_to_default_pose()
    return manager, robot


def print_state(robot: Any) -> None:
    state = robot.manager.get_entity_state(robot.robot_name)
    keys = list(state.keys())
    print(f"[state] keys: {keys}")
    if "qpos" in state:
        qpos = state["qpos"]
        if isinstance(qpos, list) and qpos:
            if isinstance(qpos[0], list):
                print(f"[state] qpos[0][:7] = {qpos[0][:7]}")
            else:
                print(f"[state] qpos[:7] = {qpos[:7]}")
