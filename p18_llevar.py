"""Carry the mapped marker to the other side of the sheet and let go."""

import time

import mujoco
import mujoco.viewer
import numpy as np

import p8_mirar
import p12_agarrar as kin
import p13_pellizcar as pinch
import p16_mapear as mapped
import p17_alinear as wrist


def main() -> None:
    """Read the overhead camera and play the carry in the viewer."""
    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        frame = p8_mirar.latest_frame(cam)
    finally:
        cam.disconnect()
    nx, ny, yaw, _ = mapped.find_layout(frame)
    model, data = kin.load()
    mapped.put(model, data, nx, ny, yaw)
    started_inside = _covered(model, data)
    wrist.align(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:

        def draw() -> None:
            viewer.sync()
            time.sleep(kin.DT)

        pinch.pinch(model, data, draw)
        _drop(model, data, _target(model, data, started_inside), draw)
        where = "fuera" if started_inside else "dentro"
        if started_inside:
            assert not _covered(model, data), "el marcador siguio sobre el papel"
        else:
            assert _covered(model, data), "el marcador no quedo sobre el papel"
        print(f"nx={nx:.2f} ny={ny:.2f} queda={where}")
        while viewer.is_running():
            viewer.sync()
            time.sleep(kin.DT)


def _drop(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, draw) -> None:
    """Carry the closed jaw to target xy, set the marker down, and open."""
    closed = float(model.jnt_range[model.joint("gripper").id][0])
    opened = float(model.jnt_range[model.joint("gripper").id][1])
    goal = target.copy()
    goal[2] = data.site("gripperframe").xpos[2]
    for _ in range(250):
        error = goal - data.site("gripperframe").xpos
        error[2] = 0.0
        if float(np.linalg.norm(error)) < 0.02:
            break
        pinch._drive(model, data, np.clip(error, -kin.SPEED, kin.SPEED), closed, 1, draw)
    else:
        raise RuntimeError(f"no llego al destino: {data.site('gripperframe').xpos}")
    for _ in range(80):
        if float(data.site("gripperframe").xpos[2]) < 0.04:
            break
        pinch._drive(model, data, np.array([0.0, 0.0, -kin.SPEED]), closed, 1, draw)
    for step in range(160):
        data.ctrl[pinch._ctrl(model, "gripper")] = closed + (opened - closed) * min(1.0, step / 100.0)
        pinch._physics(model, data, draw)
    for _ in range(80):
        pinch._physics(model, data, draw)


def _target(model: mujoco.MjModel, data: mujoco.MjData, started_inside: bool) -> np.ndarray:
    """Paper center, or a point beside the sheet that clears the marker's long side."""
    paper = data.geom("paper").xpos.copy()
    if not started_inside:
        return paper
    reach = float(np.max(model.geom("marker").size[:2])) + 0.01
    paper[1] += model.geom("paper").size[1] + reach
    return paper


def _covered(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True when the whole marker is lying inside the sheet."""
    paper = data.geom("paper").xpos
    half = model.geom("paper").size[:2]
    center = data.geom("marker").xpos
    if not (0.005 < center[2] < 0.03):
        return False
    rotation = data.geom("marker").xmat.reshape(3, 3)
    hx, hy = model.geom("marker").size[:2]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            corner = center + rotation[:, 0] * hx * sx + rotation[:, 1] * hy * sy
            if abs(corner[0] - paper[0]) >= half[0] or abs(corner[1] - paper[1]) >= half[1]:
                return False
    return True


if __name__ == "__main__":
    main()
