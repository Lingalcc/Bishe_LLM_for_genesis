from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.protocols.toolcall import validate_payload
from src.genesis.sim_runtime import resolve_robot_asset_path


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _call_first(obj: Any, names: list[str], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"none of methods exist: {names}")


@dataclass
class _EntityRecord:
    name: str
    category: str
    entity_class: str
    params: dict[str, Any]
    entity: Any | None = None


class GenesisManager:
    def __init__(
        self,
        *,
        backend: str = "cpu",
        init_kwargs: dict[str, Any] | None = None,
        scene_kwargs: dict[str, Any] | None = None,
        auto_init: bool = True,
        auto_create_scene: bool = True,
        force_reinit: bool = False,
    ) -> None:
        self.backend = backend
        self.init_kwargs = dict(init_kwargs or {})
        self.scene_kwargs = dict(scene_kwargs or {})
        self.auto_init = auto_init
        self.auto_create_scene = auto_create_scene
        self.force_reinit = force_reinit

        self.gs: Any | None = None
        self.scene: Any | None = None
        self._records: dict[str, _EntityRecord] = {}
        self._built = False

        if self.auto_init:
            self.init_runtime(force_reinit=self.force_reinit)
        if self.auto_create_scene:
            self.create_scene()

    def init_runtime(self, *, force_reinit: bool = False) -> None:
        if self.gs is None:
            import genesis as gs  # lazy import to avoid import-time side effects

            self.gs = gs
        gs = self.gs
        if gs is None:
            raise RuntimeError("genesis module is unavailable")

        initialized = bool(getattr(gs, "_initialized", False))
        if initialized and force_reinit:
            try:
                gs.destroy()
            except Exception:
                pass
            initialized = bool(getattr(gs, "_initialized", False))

        if not initialized:
            init_kwargs = dict(self.init_kwargs)
            backend_value = self.backend
            if isinstance(self.backend, str):
                backend_attr = getattr(gs, self.backend, None)
                if backend_attr is not None:
                    backend_value = backend_attr
            gs.init(backend=backend_value, **init_kwargs)

    def create_scene(self) -> Any:
        if self.gs is None:
            self.init_runtime(force_reinit=False)
        gs = self.gs
        if gs is None:
            raise RuntimeError("genesis module is unavailable")
        self.scene = gs.Scene(**dict(self.scene_kwargs))
        self._built = False
        return self.scene

    def _require_scene(self) -> Any:
        if self.scene is None:
            self.create_scene()
        if self.scene is None:
            raise RuntimeError("scene is not created")
        return self.scene

    def add_plane(self, name: str = "ground", **morph_kwargs: Any) -> Any:
        if self.gs is None:
            self.init_runtime(force_reinit=False)
        gs = self.gs
        if gs is None:
            raise RuntimeError("genesis module is unavailable")
        scene = self._require_scene()

        morph = gs.morphs.Plane(**morph_kwargs) if morph_kwargs else gs.morphs.Plane()
        entity = scene.add_entity(morph)
        self._records[name] = _EntityRecord(
            name=name,
            category="plane",
            entity_class=type(entity).__name__,
            params={"morph_kwargs": dict(morph_kwargs)},
            entity=entity,
        )
        return entity

    def add_robot_from_file(
        self,
        *,
        name: str,
        file: str,
        robot_type: str = "mjcf",
        morph_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if self.gs is None:
            self.init_runtime(force_reinit=False)
        gs = self.gs
        if gs is None:
            raise RuntimeError("genesis module is unavailable")
        scene = self._require_scene()

        kwargs = dict(morph_kwargs or {})
        rt = robot_type.strip().lower()
        resolved_file = resolve_robot_asset_path(file, robot_type=rt)
        if rt == "mjcf":
            morph = gs.morphs.MJCF(file=str(resolved_file), **kwargs)
        elif rt == "urdf":
            morph = gs.morphs.URDF(file=str(resolved_file), **kwargs)
        else:
            raise ValueError(f"unsupported robot_type: {robot_type}")
        entity = scene.add_entity(morph)
        self._records[name] = _EntityRecord(
            name=name,
            category="robot",
            entity_class=type(entity).__name__,
            params={
                "file": file,
                "resolved_file": str(resolved_file),
                "robot_type": rt,
                "morph_kwargs": kwargs,
            },
            entity=entity,
        )
        return entity

    def add_object(self, *, name: str, morph: Any, **params: Any) -> Any:
        scene = self._require_scene()
        entity = scene.add_entity(morph)
        record_params = dict(params)
        if "morph_kwargs" not in record_params:
            mk = getattr(morph, "__dict__", None)
            if isinstance(mk, dict):
                record_params["morph_kwargs"] = dict(mk)
        self._records[name] = _EntityRecord(
            name=name,
            category="object",
            entity_class=type(entity).__name__,
            params=record_params,
            entity=entity,
        )
        return entity

    def build_scene(self) -> None:
        scene = self._require_scene()
        scene.build()
        self._built = True

    def step(self, n_steps: int = 1) -> None:
        scene = self._require_scene()
        for _ in range(max(1, int(n_steps))):
            scene.step()

    def get_entities(self) -> dict[str, Any]:
        return {name: rec.entity for name, rec in self._records.items() if rec.entity is not None}

    def get_entity(self, name: str) -> Any:
        rec = self._records.get(name)
        if rec is None or rec.entity is None:
            raise KeyError(f"entity not found: {name}")
        return rec.entity

    def get_entity_state(self, name: str) -> dict[str, Any]:
        entity = self.get_entity(name)
        state: dict[str, Any] = {}

        for key, names in {
            "pos": ["get_pos", "get_position"],
            "quat": ["get_quat", "get_orientation", "get_quaternion"],
            "qpos": ["get_qpos", "get_dofs_position"],
            "qvel": ["get_qvel", "get_dofs_velocity"],
        }.items():
            try:
                state[key] = _to_builtin(_call_first(entity, names))
            except Exception:
                pass
        return state

    def get_entity_params(self, name: str, *, include_state: bool = False) -> dict[str, Any]:
        rec = self._records.get(name)
        if rec is None:
            raise KeyError(f"entity not found: {name}")
        out = {
            "name": rec.name,
            "category": rec.category,
            "entity_class": rec.entity_class,
            **dict(rec.params),
        }
        if include_state:
            out["state"] = self.get_entity_state(name)
        return out

    def get_entity_with_params(self, name: str, *, include_state: bool = False) -> tuple[Any, dict[str, Any]]:
        return self.get_entity(name), self.get_entity_params(name, include_state=include_state)

    def get_all_entity_params(self, *, include_state: bool = False) -> dict[str, dict[str, Any]]:
        return {
            name: self.get_entity_params(name, include_state=include_state)
            for name in sorted(self._records.keys())
        }

    def get_scene_info(self, *, include_entities: bool = True, include_state: bool = False) -> dict[str, Any]:
        names = sorted(self._records.keys())
        info: dict[str, Any] = {
            "built": self._built,
            "entity_count": len(names),
            "entity_names": names,
        }
        if include_entities:
            info["entities"] = self.get_all_entity_params(include_state=include_state)
        return info

    def release(self, *, destroy_runtime: bool = True) -> None:
        self.scene = None
        self._records.clear()
        self._built = False
        if destroy_runtime and self.gs is not None:
            try:
                self.gs.destroy()
            except Exception:
                pass


class GenesisRobot:
    def __init__(
        self,
        manager: GenesisManager,
        *,
        robot_name: str,
        ee_link_name: str,
        arm_dofs_idx_local: list[int],
        gripper_dofs_idx_local: list[int],
        default_interp_steps: int = 120,
        default_command_steps: int = 120,
        default_init_qpos: list[float] | None = None,
        default_init_steps: int = 120,
    ) -> None:
        self.manager = manager
        self.robot_name = robot_name
        self.ee_link_name = ee_link_name
        self.arm_dofs_idx_local = list(arm_dofs_idx_local)
        self.gripper_dofs_idx_local = list(gripper_dofs_idx_local)
        self.default_interp_steps = int(default_interp_steps)
        self.default_command_steps = int(default_command_steps)
        self.default_init_qpos = list(default_init_qpos) if default_init_qpos is not None else None
        self.default_init_steps = int(default_init_steps)

        self.robot = self.manager.get_entity(robot_name)
        self._ee_link = self._resolve_ee_link()

    def _resolve_ee_link(self) -> Any | None:
        robot = self.robot
        for name in ("get_link", "find_link"):
            fn = getattr(robot, name, None)
            if callable(fn):
                try:
                    return fn(self.ee_link_name)
                except Exception:
                    continue
        return None

    def move_to_default_pose(self) -> None:
        if self.default_init_qpos is None:
            return
        self._set_qpos(self.default_init_qpos, steps=self.default_init_steps)

    def execute_json(self, payload_like: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        commands = self._normalize_commands(payload_like)
        results: list[dict[str, Any]] = []
        for idx, cmd in enumerate(commands):
            action = str(cmd.get("action", "")).strip()
            item: dict[str, Any] = {"index": idx, "action": action}
            try:
                ret = self._execute_one(cmd)
                item["status"] = "ok"
                if ret is not None:
                    item["result"] = ret
            except Exception as exc:
                item["status"] = "error"
                item["error"] = f"{type(exc).__name__}: {exc}"
            results.append(item)
        return results

    def _normalize_commands(self, payload_like: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return validate_payload(payload_like, policy="execution")

    def _execute_one(self, cmd: dict[str, Any]) -> Any:
        action = str(cmd.get("action", "")).strip()
        steps = max(1, int(self.default_command_steps))

        if action in {"wait", "step"}:
            self.manager.step(steps)
            return {"steps": steps}

        if action == "get_state":
            return self.manager.get_entity_state(self.robot_name)

        if action == "open_gripper":
            pos = float(cmd.get("position", 0.04))
            self._set_gripper(pos, steps=steps)
            return {"position": pos}

        if action == "close_gripper":
            pos = float(cmd.get("position", 0.0))
            self._set_gripper(pos, steps=steps)
            return {"position": pos}

        if action == "set_qpos":
            qpos = cmd.get("qpos")
            if not isinstance(qpos, list) or not qpos:
                raise ValueError("set_qpos requires non-empty 'qpos' list")
            self._set_qpos([float(v) for v in qpos], steps=steps)
            return {"len": len(qpos)}

        if action in {"set_dofs_position", "control_dofs_position"}:
            values = self._ensure_number_list(cmd.get("values"), "values")
            dofs = self._ensure_int_list_or_none(cmd.get("dofs_idx_local"))
            self._control_dofs_position(values, dofs_idx_local=dofs, steps=steps)
            return {"len": len(values)}

        if action == "control_dofs_velocity":
            values = self._ensure_number_list(cmd.get("values"), "values")
            dofs = self._ensure_int_list_or_none(cmd.get("dofs_idx_local"))
            _call_first(self.robot, ["control_dofs_velocity"], values=values, dofs_idx_local=dofs)
            self.manager.step(steps)
            return {"len": len(values)}

        if action == "control_dofs_force":
            values = self._ensure_number_list(cmd.get("values"), "values")
            dofs = self._ensure_int_list_or_none(cmd.get("dofs_idx_local"))
            _call_first(self.robot, ["control_dofs_force"], values=values, dofs_idx_local=dofs)
            self.manager.step(steps)
            return {"len": len(values)}

        if action == "move_ee":
            pos = self._ensure_number_list(cmd.get("pos"), "pos", min_len=3)
            quat = self._ensure_number_list(cmd.get("quat"), "quat", min_len=4)
            self._move_ee(pos=pos[:3], quat=quat[:4], steps=steps)
            return {"pos": pos[:3], "quat": quat[:4]}

        if action == "reset_scene":
            self.manager.step(1)
            return {"status": "noop_reset"}

        raise ValueError(f"unsupported action: {action}")

    @staticmethod
    def _ensure_number_list(value: Any, key: str, *, min_len: int = 1) -> list[float]:
        if not isinstance(value, list) or len(value) < min_len:
            raise ValueError(f"'{key}' must be list with len >= {min_len}")
        out: list[float] = []
        for item in value:
            out.append(float(item))
        return out

    @staticmethod
    def _ensure_int_list_or_none(value: Any) -> list[int] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("'dofs_idx_local' must be int list")
        return [int(v) for v in value]

    def _set_qpos(self, qpos: list[float], *, steps: int) -> None:
        qpos_try = False
        for name in ("set_qpos",):
            fn = getattr(self.robot, name, None)
            if not callable(fn):
                continue
            qpos_try = True
            try:
                fn(qpos)
                self.manager.step(max(1, steps))
                return
            except TypeError:
                try:
                    fn(qpos=qpos)
                    self.manager.step(max(1, steps))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        arm_values = qpos[: len(self.arm_dofs_idx_local)]
        if arm_values:
            self._control_dofs_position(arm_values, dofs_idx_local=self.arm_dofs_idx_local, steps=max(1, steps))
        if len(qpos) > len(self.arm_dofs_idx_local) and self.gripper_dofs_idx_local:
            gripper_values = qpos[len(self.arm_dofs_idx_local) : len(self.arm_dofs_idx_local) + len(self.gripper_dofs_idx_local)]
            if gripper_values:
                self._control_dofs_position(
                    gripper_values,
                    dofs_idx_local=self.gripper_dofs_idx_local,
                    steps=max(1, steps),
                )
        if qpos_try:
            return
        self.manager.step(max(1, steps))

    def _set_gripper(self, position: float, *, steps: int) -> None:
        values = [float(position) for _ in self.gripper_dofs_idx_local]
        if not values:
            return
        self._control_dofs_position(values, dofs_idx_local=self.gripper_dofs_idx_local, steps=steps)

    def _control_dofs_position(self, values: list[float], *, dofs_idx_local: list[int] | None, steps: int) -> None:
        for call in (
            lambda: _call_first(self.robot, ["control_dofs_position"], values=values, dofs_idx_local=dofs_idx_local),
            lambda: _call_first(self.robot, ["set_dofs_position"], values=values, dofs_idx_local=dofs_idx_local),
            lambda: _call_first(self.robot, ["control_dofs_position"], values, dofs_idx_local),
            lambda: _call_first(self.robot, ["set_dofs_position"], values, dofs_idx_local),
        ):
            try:
                call()
                self.manager.step(max(1, steps))
                return
            except Exception:
                continue
        raise RuntimeError("robot does not support dof position control interface")

    def _move_ee(self, *, pos: list[float], quat: list[float], steps: int) -> None:
        target_qpos = None
        if self._ee_link is not None:
            for name in ("inverse_kinematics", "ik"):
                fn = getattr(self.robot, name, None)
                if callable(fn):
                    try:
                        target_qpos = fn(link=self._ee_link, pos=pos, quat=quat)
                        break
                    except TypeError:
                        try:
                            target_qpos = fn(self._ee_link, pos, quat)
                            break
                        except Exception:
                            pass
                    except Exception:
                        pass
        if target_qpos is not None:
            qpos_list = _to_builtin(target_qpos)
            if isinstance(qpos_list, list):
                arm = [float(v) for v in qpos_list[: len(self.arm_dofs_idx_local)]]
                if arm:
                    self._control_dofs_position(
                        arm,
                        dofs_idx_local=self.arm_dofs_idx_local,
                        steps=max(1, steps),
                    )
                    return

        # If IK is unavailable, keep behavior deterministic and still progress simulation.
        self.manager.step(max(1, steps))
