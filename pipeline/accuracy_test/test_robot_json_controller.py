import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GENESIS_PATH = REPO_ROOT / "Genesis"
if str(LOCAL_GENESIS_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_GENESIS_PATH))

import genesis as gs

from genesis_example.genesis_tools import GenesisManager, GenesisRobot


def build_demo_manager() -> GenesisManager:
    manager = GenesisManager(
        backend="cpu",
        init_kwargs={"precision": "32", "logging_level": "warning"},
        scene_kwargs={
            "show_viewer": True,
            "sim_options": gs.options.SimOptions(dt=0.01),
        },
        auto_init=True,
        auto_create_scene=True,
        force_reinit=True,
    )
    manager.add_plane("ground")
    manager.add_robot_from_file(
        name="franka",
        file="xml/franka_emika_panda/panda.xml",
        robot_type="mjcf",
    )
    manager.add_object(
        name="cube",
        morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
    )
    manager.build_scene()
    return manager


def test_llm_json_control() -> None:
    manager = build_demo_manager()
    robot = GenesisRobot(
        manager,
        robot_name="franka",
        ee_link_name="hand",
        arm_dofs_idx_local=list(range(7)),
        gripper_dofs_idx_local=[7, 8],
    )

    llm_output_text = """
请执行以下机械臂动作：
```json
{
  "commands": [
    {"action": "open_gripper", "position": 0.04, "steps": 40},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.18], "quat": [0, 1, 0, 0], "steps": 120},
    {"action": "close_gripper", "position": 0.0, "steps": 40},
    {"action": "wait", "steps": 20},
    {"action": "get_state"}
  ]
}
```
"""

    results = robot.execute_json(llm_output_text)
    print("Execution results:")
    for item in results:
        print(item["index"], item["action"], item["status"])

    state = manager.get_entity_state("franka")
    print("Robot state keys:", list(state.keys()))
    assert "qpos" in state, "Expected qpos in robot state."

    manager.release(destroy_runtime=True)
    print("test_llm_json_control passed.")


if __name__ == "__main__":
    test_llm_json_control()
