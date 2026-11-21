"""Save one sim episode of the marker being placed on the paper."""

import shutil
from pathlib import Path

import mujoco
import numpy as np

import p12_agarrar as kin
import p14_poner as place

ROOT = Path(__file__).resolve().parent / "data" / "sim_place"
TASK = "Pick the yellow marker outside the white paper and put it completely inside the paper."
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HEIGHT, WIDTH = 240, 320


def _joints(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the six joint positions."""
    return np.array([data.qpos[model.joint(name).qposadr].item() for name in JOINTS], dtype=np.float32)


def record() -> Path:
    """Run one place and write it under data/sim_place. A previous folder is replaced."""
    from lerobot.datasets import LeRobotDataset

    if ROOT.exists():
        shutil.rmtree(ROOT)
    dataset = LeRobotDataset.create(
        repo_id="local/sim_place",
        fps=30,
        root=ROOT,
        robot_type="so101_follower",
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
            "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
            "observation.images.overhead": {
                "dtype": "image",
                "shape": (3, HEIGHT, WIDTH),
                "names": ["channel", "height", "width"],
            },
        },
    )
    model, data = kin.load()
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    try:

        def draw() -> None:
            renderer.update_scene(data, camera="overhead")
            pose = _joints(model, data)
            dataset.add_frame(
                {
                    "observation.state": pose,
                    "action": pose,
                    "observation.images.overhead": renderer.render(),
                    "task": TASK,
                }
            )

        place.place(model, data, draw)
    finally:
        renderer.close()
    dataset.save_episode()
    dataset.finalize()
    return ROOT


def main() -> None:
    """Record the episode and print where it was written."""
    root = record()
    print(root)


if __name__ == "__main__":
    main()
