"""Pick the marker from outside the paper and set it down on the sheet."""

import time

import mujoco
import mujoco.viewer
import numpy as np

import p12_agarrar as kin
import p13_pellizcar as pinch

OUTSIDE = np.array([0.20, 0.08, 0.010])


def _on_paper(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True when the whole marker is lying on the sheet."""
    paper = data.geom("paper").xpos
    marker = data.geom("marker").xpos
    paper_half = model.geom("paper").size[:2]
    marker_half = model.geom("marker").size[:2]
    inside = abs(marker[0] - paper[0]) + marker_half[0] < paper_half[0]
    inside = inside and abs(marker[1] - paper[1]) + marker_half[1] < paper_half[1]
    return bool(inside and 0.005 < marker[2] < 0.03)


def place(model: mujoco.MjModel, data: mujoco.MjData, draw=None) -> None:
    """Start outside the sheet, pinch, carry to the middle, and open."""
    if draw is None:
        draw = lambda: None
    free = model.joint("marker_free").qposadr.item()
    data.qpos[free : free + 3] = OUTSIDE
    mujoco.mj_forward(model, data)
    assert not _on_paper(model, data), "el marcador ya empezaba en el papel"
    pinch.pinch(model, data, draw)
    closed = float(model.jnt_range[model.joint("gripper").id][0])
    opened = float(model.jnt_range[model.joint("gripper").id][1])
    target = data.geom("paper").xpos.copy()
    target[2] = data.site("gripperframe").xpos[2]
    for _ in range(250):
        error = target - data.site("gripperframe").xpos
        error[2] = 0.0
        if float(np.linalg.norm(error)) < 0.02:
            break
        pinch._drive(model, data, np.clip(error, -kin.SPEED, kin.SPEED), closed, 1, draw)
    else:
        raise RuntimeError(f"no llego al papel: {data.site('gripperframe').xpos}")
    for _ in range(80):
        if float(data.site("gripperframe").xpos[2]) < 0.04:
            break
        pinch._drive(model, data, np.array([0.0, 0.0, -kin.SPEED]), closed, 1, draw)
    for step in range(160):
        data.ctrl[pinch._ctrl(model, "gripper")] = closed + (opened - closed) * min(1.0, step / 100.0)
        pinch._physics(model, data, draw)
    for _ in range(80):
        pinch._physics(model, data, draw)
    assert _on_paper(model, data), "el marcador no quedo sobre el papel"


def main() -> None:
    """Play the place in the viewer, then hold the last frame."""
    model, data = kin.load()
    with mujoco.viewer.launch_passive(model, data) as viewer:

        def draw() -> None:
            viewer.sync()
            time.sleep(kin.DT)

        place(model, data, draw)
        while viewer.is_running():
            viewer.sync()
            time.sleep(kin.DT)


if __name__ == "__main__":
    main()
