from __future__ import annotations

import json
from typing import Any

from src.genesis.genesis_tools import GenesisManager, GenesisRobot
from src.genesis.sim_runtime import DEFAULT_FRANKA_MJCF, preflight_sim_environment


class GenesisInteractiveTestEnv:
    """Interactive Genesis test env with Franka and one cube."""

    def __init__(self, *, show_viewer: bool = True) -> None:
        self.show_viewer = show_viewer
        self.manager: GenesisManager | None = None
        self.robot: GenesisRobot | None = None
        self._cube_name = "cube"
        self._cube_init_pos = [0.65, 0.0, 0.02]
        self.robot_file = DEFAULT_FRANKA_MJCF
        self.robot_type = "mjcf"

    def setup(self) -> None:
        preflight = preflight_sim_environment(
            robot_file=self.robot_file,
            robot_type=self.robot_type,
        )
        import genesis as gs

        self.manager = GenesisManager(
            backend="cpu",
            init_kwargs={"precision": "32", "logging_level": "warning"},
            scene_kwargs={
                "show_viewer": self.show_viewer,
                "sim_options": gs.options.SimOptions(dt=0.01),
            },
            auto_init=True,
            auto_create_scene=True,
            force_reinit=True,
        )
        self.manager.add_plane("ground")
        self.manager.add_robot_from_file(
            name="franka",
            file=str(preflight.resolved_robot_file),
            robot_type=self.robot_type,
        )
        self.manager.add_object(
            name=self._cube_name,
            morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=tuple(self._cube_init_pos)),
        )
        self.manager.build_scene()
        self.robot = GenesisRobot(
            self.manager,
            robot_name="franka",
            ee_link_name="hand",
            arm_dofs_idx_local=list(range(7)),
            gripper_dofs_idx_local=[7, 8],
            default_command_steps=120,
        )
        self.move_robot_to_init_pose_with_ik()

    def move_robot_to_init_pose_with_ik(self) -> None:
        if self.robot is None:
            raise RuntimeError("Robot controller has not been initialized.")
        init_actions = {
            "commands": [
                {"action": "open_gripper", "position": 0.04},
                {"action": "move_ee", "pos": [0.55, 0.00, 0.35], "quat": [0, 1, 0, 0]},
                {"action": "wait"},
            ]
        }
        self.execute_action_json(init_actions)

    def get_cube_positions(self) -> dict[str, dict[str, Any]]:
        if self.manager is None:
            raise RuntimeError("Environment has not been setup.")
        state = self._safe_get_entity_state(self._cube_name)
        params = self._safe_get_entity_params(self._cube_name)
        pos = self._extract_position(state, params, fallback_pos=self._cube_init_pos)
        return {self._cube_name: {"position": pos}}

    def execute_action_json(self, action_payload: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.robot is None:
            raise RuntimeError("Robot controller has not been initialized.")
        raw = action_payload if isinstance(action_payload, str) else json.dumps(action_payload, ensure_ascii=False)
        return self.robot.execute_json(raw)

    def close(self) -> None:
        if self.manager is not None:
            self.manager.release(destroy_runtime=True)
        self.manager = None
        self.robot = None

    def _safe_get_entity_state(self, name: str) -> dict[str, Any]:
        if self.manager is None:
            return {}
        try:
            state = self.manager.get_entity_state(name)
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def _safe_get_entity_params(self, name: str) -> dict[str, Any]:
        if self.manager is None:
            return {}
        try:
            params = self.manager.get_entity_params(name, include_state=False)
            return params if isinstance(params, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_position(
        state: dict[str, Any],
        params: dict[str, Any],
        *,
        fallback_pos: list[float],
    ) -> list[float]:
        for key in ("pos", "position"):
            v = state.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                return [float(v[0]), float(v[1]), float(v[2])]
        morph_kwargs = params.get("morph_kwargs")
        if isinstance(morph_kwargs, dict):
            v = morph_kwargs.get("pos")
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                return [float(v[0]), float(v[1]), float(v[2])]
        return [float(fallback_pos[0]), float(fallback_pos[1]), float(fallback_pos[2])]
