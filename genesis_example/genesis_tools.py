from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class ManagedEntity:
    name: str
    entity: Any
    category: str
    meta: dict[str, Any] = field(default_factory=dict)


class GenesisManager:
    """
    A lightweight lifecycle manager for Genesis scenes.

    Core features:
    1) Initialize/reinitialize Genesis runtime.
    2) Create and build one managed scene.
    3) Add robots and objects with stable names.
    4) Query scene entities and their key parameters.
    5) Release scene/runtime resources safely.
    """

    def __init__(
        self,
        backend: str | int | None = "gpu",
        *,
        init_kwargs: dict[str, Any] | None = None,
        scene_kwargs: dict[str, Any] | None = None,
        auto_init: bool = True,
        auto_create_scene: bool = True,
        force_reinit: bool = False,
    ) -> None:
        self.backend = self._resolve_backend(backend)
        self.init_kwargs = dict(init_kwargs or {})
        self.scene_kwargs = dict(scene_kwargs or {})

        self.scene: gs.Scene | None = None
        self._entities: dict[str, ManagedEntity] = {}

        if auto_init:
            self.init_genesis(force_reinit=force_reinit)
        if auto_create_scene:
            self.create_scene(**self.scene_kwargs)

    # -------------------------------------------------------------------------
    # Initialization and lifecycle
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(backend: str | int | None) -> int:
        if backend is None:
            return gs.gpu
        if isinstance(backend, str):
            name = backend.lower()
            if not hasattr(gs, name):
                raise ValueError(
                    f"Unknown backend '{backend}'. Expected one of: cpu/gpu/cuda/metal/vulkan."
                )
            return getattr(gs, name)
        return backend

    def init_genesis(self, *, force_reinit: bool = False, **override_init_kwargs: Any) -> None:
        """
        Initialize Genesis runtime if needed.
        """
        init_kwargs = dict(self.init_kwargs)
        init_kwargs.update(override_init_kwargs)

        backend = self._resolve_backend(init_kwargs.pop("backend", self.backend))
        already_initialized = bool(getattr(gs, "_initialized", False))

        if already_initialized and force_reinit:
            gs.destroy()
            already_initialized = False

        if not already_initialized:
            gs.init(backend=backend, **init_kwargs)

        self.backend = backend

    def create_scene(self, **scene_kwargs: Any) -> gs.Scene:
        """
        Create a managed scene. If one already exists, it will be released first.
        """
        if self.scene is not None:
            self.release_scene()

        self.scene = gs.Scene(**scene_kwargs)
        self._entities.clear()
        return self.scene

    def build_scene(self, **build_kwargs: Any) -> None:
        scene = self._require_scene()
        if not scene.is_built:
            scene.build(**build_kwargs)

    def reset_scene(self, *args: Any, **kwargs: Any) -> None:
        self._require_scene().reset(*args, **kwargs)

    def step(self, n_steps: int = 1, **step_kwargs: Any) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        scene = self._require_scene()
        for _ in range(n_steps):
            scene.step(**step_kwargs)

    def release_scene(self) -> None:
        if self.scene is not None:
            self.scene.destroy()
            self.scene = None
        self._entities.clear()

    def release(self, *, destroy_runtime: bool = False) -> None:
        """
        Release scene resources and optionally destroy Genesis runtime.
        """
        self.release_scene()
        if destroy_runtime and bool(getattr(gs, "_initialized", False)):
            gs.destroy()

    # -------------------------------------------------------------------------
    # Entity management
    # -------------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        *,
        morph: Any,
        material: Any = None,
        surface: Any = None,
        category: str = "object",
        visualize_contact: bool = False,
        vis_mode: str | None = None,
    ) -> Any:
        scene = self._require_scene()
        if scene.is_built:
            raise RuntimeError("Cannot add entity after scene.build().")
        if name in self._entities:
            raise ValueError(f"Entity name '{name}' already exists.")

        entity = scene.add_entity(
            morph=morph,
            material=material,
            surface=surface,
            visualize_contact=visualize_contact,
            vis_mode=vis_mode,
        )

        self._entities[name] = ManagedEntity(
            name=name,
            entity=entity,
            category=category,
            meta={
                "visualize_contact": visualize_contact,
                "vis_mode": vis_mode,
            },
        )
        return entity

    def add_robot(
        self,
        name: str,
        *,
        morph: Any,
        material: Any = None,
        surface: Any = None,
        visualize_contact: bool = False,
        vis_mode: str | None = None,
    ) -> Any:
        robot_morph_types = (gs.morphs.URDF, gs.morphs.MJCF, gs.morphs.Drone)
        if not isinstance(morph, robot_morph_types):
            raise TypeError("add_robot expects morph in {gs.morphs.URDF, gs.morphs.MJCF, gs.morphs.Drone}.")
        return self.add_entity(
            name,
            morph=morph,
            material=material,
            surface=surface,
            category="robot",
            visualize_contact=visualize_contact,
            vis_mode=vis_mode,
        )

    def add_robot_from_file(
        self,
        name: str,
        *,
        file: str,
        robot_type: str | None = None,
        morph_kwargs: dict[str, Any] | None = None,
        material: Any = None,
        surface: Any = None,
        visualize_contact: bool = False,
        vis_mode: str | None = None,
    ) -> Any:
        morph_kwargs = dict(morph_kwargs or {})

        if robot_type is None:
            suffix = Path(file).suffix.lower()
            if suffix == ".urdf":
                robot_type = "urdf"
            elif suffix == ".xml":
                robot_type = "mjcf"
            else:
                raise ValueError("Cannot infer robot_type. Please set robot_type explicitly: urdf/mjcf/drone.")

        robot_type = robot_type.lower()
        builders = {
            "urdf": gs.morphs.URDF,
            "mjcf": gs.morphs.MJCF,
            "drone": gs.morphs.Drone,
        }
        if robot_type not in builders:
            raise ValueError(f"Unsupported robot_type '{robot_type}'. Use urdf/mjcf/drone.")

        morph = builders[robot_type](file=file, **morph_kwargs)
        return self.add_robot(
            name=name,
            morph=morph,
            material=material,
            surface=surface,
            visualize_contact=visualize_contact,
            vis_mode=vis_mode,
        )

    def add_object(
        self,
        name: str,
        *,
        morph: Any,
        material: Any = None,
        surface: Any = None,
        visualize_contact: bool = False,
        vis_mode: str | None = None,
    ) -> Any:
        return self.add_entity(
            name,
            morph=morph,
            material=material,
            surface=surface,
            category="object",
            visualize_contact=visualize_contact,
            vis_mode=vis_mode,
        )

    def add_plane(self, name: str = "plane", **plane_kwargs: Any) -> Any:
        return self.add_object(name=name, morph=gs.morphs.Plane(**plane_kwargs))

    def get_entity(self, name: str) -> Any:
        if name not in self._entities:
            raise KeyError(f"Entity '{name}' not found.")
        return self._entities[name].entity

    def get_entities(self, category: str | None = None) -> dict[str, Any]:
        if category is None:
            return {name: obj.entity for name, obj in self._entities.items()}
        return {name: obj.entity for name, obj in self._entities.items() if obj.category == category}

    # -------------------------------------------------------------------------
    # Introspection helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: GenesisManager._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [GenesisManager._to_serializable(v) for v in value]
        return str(value)

    def _option_to_dict(self, option: Any) -> Any:
        if option is None:
            return None
        if hasattr(option, "model_dump"):
            try:
                return self._to_serializable(option.model_dump())
            except Exception:
                pass
        if hasattr(option, "__dict__"):
            raw = {}
            for key, val in vars(option).items():
                if key.startswith("_"):
                    continue
                raw[key] = self._to_serializable(val)
            if raw:
                return raw
        return str(option)

    def get_entity_state(self, name: str) -> dict[str, Any]:
        scene = self._require_scene()
        if not scene.is_built:
            raise RuntimeError("Scene must be built before querying runtime state.")

        entity = self.get_entity(name)
        state: dict[str, Any] = {}

        fn_map = {
            "pos": "get_pos",
            "quat": "get_quat",
            "vel": "get_vel",
            "ang": "get_ang",
            "qpos": "get_qpos",
            "dofs_position": "get_dofs_position",
            "dofs_velocity": "get_dofs_velocity",
            "dofs_force": "get_dofs_force",
        }
        for out_key, fn_name in fn_map.items():
            if hasattr(entity, fn_name):
                fn = getattr(entity, fn_name)
                try:
                    state[out_key] = self._to_serializable(fn())
                except Exception:
                    continue

        if hasattr(entity, "get_mass"):
            try:
                state["mass"] = float(entity.get_mass())
            except Exception:
                pass

        return state

    def get_entity_params(self, name: str, *, include_state: bool = False) -> dict[str, Any]:
        managed = self._entities.get(name)
        if managed is None:
            raise KeyError(f"Entity '{name}' not found.")

        entity = managed.entity
        params: dict[str, Any] = {
            "name": name,
            "category": managed.category,
            "entity_class": entity.__class__.__name__,
            "uid": self._to_serializable(getattr(entity, "uid", None)),
            "idx": self._to_serializable(getattr(entity, "idx", None)),
            "is_built": bool(getattr(entity, "is_built", False)),
            "morph_type": entity.morph.__class__.__name__ if hasattr(entity, "morph") else None,
            "material_type": entity.material.__class__.__name__ if hasattr(entity, "material") else None,
            "surface_type": entity.surface.__class__.__name__ if hasattr(entity, "surface") else None,
            "morph": self._option_to_dict(getattr(entity, "morph", None)),
            "material": self._option_to_dict(getattr(entity, "material", None)),
            "surface": self._option_to_dict(getattr(entity, "surface", None)),
            "meta": self._to_serializable(managed.meta),
        }

        for field_name in (
            "n_qs",
            "n_dofs",
            "n_links",
            "n_joints",
            "q_start",
            "q_end",
            "dof_start",
            "dof_end",
            "link_start",
            "link_end",
        ):
            if hasattr(entity, field_name):
                try:
                    params[field_name] = self._to_serializable(getattr(entity, field_name))
                except Exception:
                    continue

        if hasattr(entity, "joints"):
            try:
                params["joint_names"] = [joint.name for joint in entity.joints]
            except Exception:
                pass
        if hasattr(entity, "links"):
            try:
                params["link_names"] = [link.name for link in entity.links]
            except Exception:
                pass

        if include_state:
            params["state"] = self.get_entity_state(name)

        return params

    def get_entity_with_params(self, name: str, *, include_state: bool = False) -> tuple[Any, dict[str, Any]]:
        entity = self.get_entity(name)
        params = self.get_entity_params(name, include_state=include_state)
        return entity, params

    def get_all_entity_params(self, *, include_state: bool = False) -> dict[str, dict[str, Any]]:
        return {
            name: self.get_entity_params(name, include_state=include_state)
            for name in self._entities.keys()
        }

    def get_scene_info(self, *, include_entities: bool = False, include_state: bool = False) -> dict[str, Any]:
        scene = self._require_scene()
        info = {
            "initialized": bool(getattr(gs, "_initialized", False)),
            "backend": self._to_serializable(getattr(gs, "backend", None)),
            "scene_uid": self._to_serializable(getattr(scene, "uid", None)),
            "scene_is_built": bool(scene.is_built),
            "dt": self._to_serializable(scene.dt),
            "t": self._to_serializable(scene.t),
            "substeps": self._to_serializable(scene.substeps),
            "n_envs": self._to_serializable(getattr(scene, "n_envs", 0)),
            "entity_count": len(self._entities),
            "entity_names": list(self._entities.keys()),
        }

        if include_entities:
            info["entities"] = self.get_all_entity_params(include_state=include_state)
        return info

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _require_scene(self) -> gs.Scene:
        if self.scene is None:
            raise RuntimeError("Scene is not created. Call create_scene() first.")
        return self.scene

    # Context manager API
    def __enter__(self) -> "GenesisManager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release(destroy_runtime=True)


class GenesisRobot:
    """
    Robot wrapper that executes LLM-generated JSON commands on a managed Genesis robot entity.

    Expected command format:
    - Single command:
      {"action": "open_gripper", "steps": 80}
    - Sequence:
      {"commands": [ ... ]}
    - Or a JSON list: [{...}, {...}]
    """

    def __init__(
        self,
        manager: GenesisManager,
        robot_name: str,
        *,
        ee_link_name: str = "hand",
        arm_dofs_idx_local: list[int] | None = None,
        gripper_dofs_idx_local: list[int] | None = None,
        default_interp_steps: int = 120,
        default_command_steps: int | None = None,
        default_init_qpos: list[float] | None = None,
        default_init_steps: int | None = None,
        default_gripper_open: float = 0.04,
        default_gripper_close: float = 0.0,
    ) -> None:
        self.manager = manager
        self.robot_name = robot_name
        self.ee_link_name = ee_link_name
        self.default_interp_steps = default_interp_steps
        # Fallback steps for generic control commands.
        self.default_command_steps = int(default_command_steps or default_interp_steps)
        self.default_init_qpos = None if default_init_qpos is None else np.asarray(default_init_qpos, dtype=float)
        self.default_init_steps = int(default_init_steps or default_interp_steps)
        self.default_gripper_open = float(default_gripper_open)
        self.default_gripper_close = float(default_gripper_close)

        self._arm_dofs_idx_local = arm_dofs_idx_local
        self._gripper_dofs_idx_local = gripper_dofs_idx_local

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------
    @property
    def entity(self) -> Any:
        return self.manager.get_entity(self.robot_name)

    @property
    def scene(self) -> gs.Scene:
        return self.manager._require_scene()

    def _require_built_scene(self) -> None:
        if not self.scene.is_built:
            raise RuntimeError("Scene must be built before robot command execution.")

    def _infer_default_dof_groups(self) -> tuple[list[int], list[int]]:
        n_dofs = int(getattr(self.entity, "n_dofs", 0))
        if n_dofs <= 0:
            return [], []

        if self._arm_dofs_idx_local is None and self._gripper_dofs_idx_local is None:
            if n_dofs >= 2:
                arm = list(range(0, n_dofs - 2))
                gripper = [n_dofs - 2, n_dofs - 1]
            else:
                arm = list(range(0, n_dofs))
                gripper = []
            return arm, gripper

        arm = self._arm_dofs_idx_local or []
        gripper = self._gripper_dofs_idx_local or []
        return arm, gripper

    @staticmethod
    def _to_numpy_1d(values: Any) -> np.ndarray:
        """
        Convert list/ndarray/tensor-like values to a 1D numpy array on host.
        """
        if torch is not None and isinstance(values, torch.Tensor):
            arr = values.detach().cpu().numpy()
        else:
            arr = np.asarray(values)
        return np.asarray(arr, dtype=float).reshape(-1)

    def move_to_default_pose(self, *, steps: int | None = None) -> bool:
        """
        Move robot to a configured fixed initial pose.

        Returns False if no default pose is configured.
        """
        self._require_built_scene()
        if self.default_init_qpos is None:
            return False

        qpos = self.default_init_qpos
        n_qs = int(getattr(self.entity, "n_qs", 0))
        if n_qs > 0 and int(qpos.shape[-1]) != n_qs:
            raise ValueError(
                f"default_init_qpos length mismatch: got {int(qpos.shape[-1])}, expected {n_qs}."
            )

        self.entity.set_qpos(qpos)
        settle_steps = int(steps if steps is not None else self.default_init_steps)
        self.manager.step(max(1, settle_steps))
        return True

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_first_json_from_text(text: str) -> Any:
        payload = text.strip()
        if not payload:
            raise ValueError("Instruction text is empty.")

        # 1) full-text JSON
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass

        # 2) fenced code block
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", payload, flags=re.IGNORECASE)
        if code_match:
            inner = code_match.group(1).strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

        # 3) raw decoder from the first JSON token
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(payload):
            if ch not in "{[":
                continue
            try:
                obj, _end = decoder.raw_decode(payload[idx:])
                return obj
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON payload found in instruction text.")

    @classmethod
    def parse_instructions(cls, instruction: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(instruction, str):
            obj = cls._extract_first_json_from_text(instruction)
        else:
            obj = instruction

        if isinstance(obj, list):
            commands = obj
        elif isinstance(obj, dict) and "commands" in obj:
            commands = obj["commands"]
        elif isinstance(obj, dict):
            commands = [obj]
        else:
            raise TypeError("Instruction JSON must be an object, a command list, or {'commands': [...]} format.")

        if not isinstance(commands, list) or not commands:
            raise ValueError("Instruction commands must be a non-empty list.")

        for i, cmd in enumerate(commands):
            if not isinstance(cmd, dict):
                raise TypeError(f"Command at index {i} must be a JSON object.")
            if "action" not in cmd:
                raise ValueError(f"Command at index {i} missing required key: 'action'.")

        return commands

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    def execute_json(self, instruction: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._require_built_scene()
        commands = self.parse_instructions(instruction)
        results: list[dict[str, Any]] = []
        for i, cmd in enumerate(commands):
            result = self.execute_command(cmd)
            result["index"] = i
            results.append(result)
        return results

    def execute_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        action = str(cmd["action"]).strip().lower()

        if action in {"step", "wait"}:
            steps = int(cmd.get("steps", 1))
            self.manager.step(max(1, steps))
            return {"action": action, "status": "ok", "steps": steps}

        if action == "reset_scene":
            self.manager.reset_scene()
            return {"action": action, "status": "ok"}

        if action == "set_qpos":
            qpos = np.asarray(cmd["qpos"], dtype=float)
            self.entity.set_qpos(qpos)
            steps = int(cmd.get("steps", self.default_command_steps))
            self.manager.step(max(1, steps))
            return {"action": action, "status": "ok", "qpos_dim": int(qpos.shape[-1]), "steps": steps}

        if action in {"set_dofs_position", "control_dofs_position", "control_dofs_velocity", "control_dofs_force"}:
            values = np.asarray(cmd["values"], dtype=float)
            dofs = cmd.get("dofs_idx_local", None)
            steps = int(cmd.get("steps", self.default_command_steps))

            fn = getattr(self.entity, action)
            if dofs is None:
                fn(values)
            else:
                fn(values, dofs)
            self.manager.step(max(1, steps))
            return {"action": action, "status": "ok", "steps": steps}

        if action in {"open_gripper", "close_gripper"}:
            _arm, gripper = self._infer_default_dof_groups()
            if not gripper:
                raise ValueError("No gripper dofs configured/inferred for this robot.")
            tgt = self.default_gripper_open if action == "open_gripper" else self.default_gripper_close
            value = float(cmd.get("position", tgt))
            steps = int(cmd.get("steps", 60))
            self.entity.control_dofs_position(np.full((len(gripper),), value, dtype=float), gripper)
            self.manager.step(max(1, steps))
            return {"action": action, "status": "ok", "steps": steps, "target": value}

        if action == "move_ee":
            pos = np.asarray(cmd["pos"], dtype=float)
            quat = cmd.get("quat", None)
            quat_arr = None if quat is None else np.asarray(quat, dtype=float)
            link_name = str(cmd.get("link_name", self.ee_link_name))
            steps = int(cmd.get("steps", self.default_interp_steps))
            arm_dofs, _gripper = self._infer_default_dof_groups()
            if not arm_dofs:
                raise ValueError("No arm dofs configured/inferred for IK control.")

            ee_link = self.entity.get_link(link_name)
            qpos = self.entity.inverse_kinematics(link=ee_link, pos=pos, quat=quat_arr)
            arm_target = self._to_numpy_1d(qpos)[: len(arm_dofs)]
            self.entity.control_dofs_position(arm_target, arm_dofs)
            self.manager.step(max(1, steps))
            return {"action": action, "status": "ok", "steps": steps, "link_name": link_name}

        if action == "get_state":
            state = self.manager.get_entity_state(self.robot_name)
            return {"action": action, "status": "ok", "state": state}

        raise ValueError(
            f"Unsupported action '{action}'. "
            "Supported actions: step/wait/reset_scene/set_qpos/set_dofs_position/control_dofs_position/"
            "control_dofs_velocity/control_dofs_force/open_gripper/close_gripper/move_ee/get_state."
        )
