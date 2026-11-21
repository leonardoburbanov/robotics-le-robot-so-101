"""Save one sim episode of the mapped marker carried to the other side."""

import shutil
from pathlib import Path

import mujoco
import numpy as np

import p8_mirar
import p12_agarrar as kin
import p13_pellizcar as pinch
import p16_mapear as mapped
import p17_alinear as wrist
import p18_llevar as carry

ROOT = Path(__file__).resolve().parent / "data" / "sim_carry"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HEIGHT, WIDTH = 240, 320


def record(frame_bgr: np.ndarray) -> Path:
    """Run one carry and write it under data/sim_carry. A previous folder is replaced."""
    from lerobot.datasets import LeRobotDataset

    nx, ny, yaw, _ = mapped.find_layout(frame_bgr)
    model, data = kin.load()
    mapped.put(model, data, nx, ny, yaw)
    started_inside = carry._covered(model, data)
    task = p8_mirar.PUT_OUTSIDE if started_inside else p8_mirar.PUT_INSIDE
    wrist.align(model, data)
    if ROOT.exists():
        shutil.rmtree(ROOT)
    dataset = LeRobotDataset.create(
        repo_id="local/sim_carry",
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
                    "task": task,
                }
            )

        pinch.pinch(model, data, draw)
        carry._drop(model, data, carry._target(model, data, started_inside), draw)
    finally:
        renderer.close()
    if started_inside:
        assert not carry._covered(model, data), "el marcador siguio sobre el papel"
    else:
        assert carry._covered(model, data), "el marcador no quedo sobre el papel"
    dataset.save_episode()
    dataset.finalize()
    return ROOT


def _joints(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the six joint positions."""
    return np.array([data.qpos[model.joint(name).qposadr].item() for name in JOINTS], dtype=np.float32)


def main() -> None:
    """Grab one overhead frame, record the carry, and print the folder."""
    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        frame = p8_mirar.latest_frame(cam)
    finally:
        cam.disconnect()
    print(record(frame))


if __name__ == "__main__":
    main()
