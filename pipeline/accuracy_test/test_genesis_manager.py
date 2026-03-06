import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GENESIS_PATH = REPO_ROOT / "Genesis"
if str(LOCAL_GENESIS_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_GENESIS_PATH))

import genesis as gs

from genesis_example.genesis_tools import GenesisManager


def run_manager_smoke_test() -> None:
    manager = GenesisManager(
        backend="cpu",
        init_kwargs={
            "precision": "32",
            "logging_level": "warning",
        },
        scene_kwargs={
            "show_viewer": False,
            "sim_options": gs.options.SimOptions(dt=0.01),
        },
        auto_init=True,
        auto_create_scene=True,
        force_reinit=True,
    )

    try:
        print("[1/6] Add entities ...")
        manager.add_plane(name="ground")
        manager.add_robot_from_file(
            name="robot",
            file="urdf/panda_bullet/panda.urdf",
            robot_type="urdf",
            morph_kwargs={"fixed": True},
        )
        manager.add_object(
            name="cube",
            morph=gs.morphs.Box(
                size=(0.05, 0.05, 0.05),
                pos=(0.5, 0.0, 0.025),
            ),
        )

        print("[2/6] Query scene info before build ...")
        pre_build_info = manager.get_scene_info(include_entities=True, include_state=False)
        assert pre_build_info["entity_count"] == 3, "Entity count should be 3 before build."
        assert "robot" in pre_build_info["entity_names"], "Robot should exist in managed names."

        print("[3/6] Build and step ...")
        manager.build_scene()
        manager.step(n_steps=5)

        print("[4/6] Query entity object + params ...")
        robot_obj, robot_params = manager.get_entity_with_params("robot", include_state=True)
        assert robot_obj is manager.get_entity("robot"), "Returned object should match manager.get_entity()."
        assert robot_params["category"] == "robot", "Category should be robot."
        assert "state" in robot_params, "State should exist when include_state=True."

        print("[5/6] Query all entity params ...")
        all_params = manager.get_all_entity_params(include_state=True)
        assert "cube" in all_params, "Cube should be included."
        assert "ground" in all_params, "Ground should be included."

        print("[6/6] Release resources ...")
        manager.release(destroy_runtime=True)
        assert manager.scene is None, "Scene should be None after release."
        assert not bool(getattr(gs, "_initialized", False)), "Genesis runtime should be destroyed."

        print("GenesisManager smoke test passed.")

    except Exception:
        print("GenesisManager smoke test failed:")
        traceback.print_exc()
        # Ensure resources are cleaned even if test fails
        manager.release(destroy_runtime=True)
        raise


if __name__ == "__main__":
    run_manager_smoke_test()
