"""Show the arm, the white paper and the yellow marker. Nothing moves."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

XML = Path(__file__).resolve().parent / "Simulation" / "SO101" / "scene.xml"


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the scene and check the marker is lying on the paper."""
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    paper = data.geom("paper").xpos
    marker = data.geom("marker").xpos
    assert marker[2] > paper[2], "el marcador quedo debajo del papel"
    assert abs(marker[0] - paper[0]) < 0.105 and abs(marker[1] - paper[1]) < 0.075
    return model, data


def main() -> None:
    """Open the scene and leave it still until the window closes."""
    model, data = load()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
