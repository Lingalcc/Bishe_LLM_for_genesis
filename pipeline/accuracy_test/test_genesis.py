import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_GENESIS_PATH = REPO_ROOT / "Genesis"
if str(LOCAL_GENESIS_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_GENESIS_PATH))

import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

scene.build()

for i in range(1000):
    scene.step()
