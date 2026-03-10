import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GENESIS_PATH = REPO_ROOT / "Genesis"
if str(LOCAL_GENESIS_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_GENESIS_PATH))

import genesis as gs

from genesis_example.genesis_tools import GenesisManager, GenesisRobot


HELP_TEXT = """
Interactive Robot Control
-------------------------
Input format:
1) Single command JSON:
   {"action": "open_gripper"}

2) Multi-command JSON:
   {"commands": [{"action":"move_ee","pos":[0.65,0,0.18],"quat":[0,1,0,0]}]}

3) LLM text with JSON code block (also supported):
   ```json
   {"commands":[...]}
   ```

Special commands:
  /help      Show this help
  /state     Print current robot state keys and short summary
  /example   Print a ready-to-use JSON example
  /quit      Exit program
"""


EXAMPLE_JSON = """{
  "commands": [
    {"action": "open_gripper", "position": 0.04},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.18], "quat": [0, 1, 0, 0]},
    {"action": "close_gripper", "position": 0.0},
    {"action": "wait", "steps": 20},
    {"action": "move_ee", "pos": [0.65, 0.0, 0.30], "quat": [0, 1, 0, 0]}
  ]
}"""


def build_interactive_env(show_viewer: bool = True) -> tuple[GenesisManager, GenesisRobot]:
    manager = GenesisManager(
        backend="gpu",
        init_kwargs={"precision": "32", "logging_level": "warning"},
        scene_kwargs={
            "show_viewer": show_viewer,
            "sim_options": gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, 0)
            ),
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
        file="xml/franka_emika_panda/panda.xml",
        robot_type="mjcf",
    )
    manager.add_object(
        name="cube",
        morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
    )

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
    )

    robot.move_to_default_pose()

    return manager, robot


def print_state(robot: GenesisRobot) -> None:
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


def main() -> None:
    print("Starting Genesis interactive control...")
    manager = None
    robot = None
    try:
        manager, robot = build_interactive_env(show_viewer=True)

        print(HELP_TEXT.strip())
        print("\nExample JSON:")
        print(EXAMPLE_JSON)
        print("\nRobot initialized to default pose.")
        print("Ready. Input instruction:")

        while True:
            raw = input("\n>>> ").strip()
            if not raw:
                continue

            if raw == "/quit":
                print("Exiting.")
                break
            if raw == "/help":
                print(HELP_TEXT.strip())
                continue
            if raw == "/example":
                print(EXAMPLE_JSON)
                continue
            if raw == "/state":
                print_state(robot)
                continue

            try:
                results = robot.execute_json(raw)
                print("[ok] Executed commands:")
                for item in results:
                    print(f"  - idx={item.get('index')} action={item.get('action')} status={item.get('status')}")
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}")
                print("Tip: use /example to copy a valid JSON command format.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        if manager is not None:
            manager.release(destroy_runtime=True)
