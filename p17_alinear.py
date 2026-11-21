"""Turn the wrist so the pads close across the mapped marker, then lift it."""

import time

import mujoco
import mujoco.viewer
import numpy as np

import p8_mirar
import p12_agarrar as kin
import p13_pellizcar as pinch
import p16_mapear as mapped


def align(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set wrist_roll so the pads squeeze the marker's narrow side.

    ponytail: tries a roll every 15 degrees and drops any roll that puts a pad
    through the floor. A contact-aware grasp planner is the upgrade.
    """
    saved = data.qpos.copy()
    short = data.geom("marker").xmat.reshape(3, 3)[:, 0].copy()
    short[2] = 0.0
    short /= np.linalg.norm(short)
    marker = data.geom("marker").xpos.copy()
    roll = model.joint("wrist_roll").qposadr.item()
    low, high = model.jnt_range[model.joint("wrist_roll").id]
    best_roll = float(saved[roll])
    best = -1.0
    for deg in range(-90, 91, 15):
        data.qpos[:] = saved
        data.qvel[:] = 0.0
        data.qpos[roll] = float(np.clip(np.deg2rad(deg), low, high))
        mujoco.mj_forward(model, data)
        try:
            kin._go(model, data, marker + np.array([0.0, 0.0, -0.01]), lambda: None)
        except RuntimeError:
            continue
        score = _pad_score(model, data, marker, short)
        if score > best:
            best = score
            best_roll = float(data.qpos[roll])
    if best < 0.5:
        raise RuntimeError("la pinza no puede cruzar el marcador")
    data.qpos[:] = saved
    data.qvel[:] = 0.0
    data.qpos[roll] = best_roll
    mujoco.mj_forward(model, data)


def main() -> None:
    """Read the overhead camera, copy the marker, align the wrist, and lift."""
    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        frame = p8_mirar.latest_frame(cam)
    finally:
        cam.disconnect()
    nx, ny, yaw, _ = mapped.find_layout(frame)
    model, data = kin.load()
    mapped.put(model, data, nx, ny, yaw)
    align(model, data)
    roll = model.joint("wrist_roll").qposadr.item()
    print(f"nx={nx:.2f} ny={ny:.2f} roll_deg={np.degrees(data.qpos[roll]):.0f}")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        def draw() -> None:
            viewer.sync()
            time.sleep(kin.DT)

        pinch.pinch(model, data, draw)
        while viewer.is_running():
            viewer.sync()
            time.sleep(kin.DT)


def _pad_score(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    marker: np.ndarray,
    short: np.ndarray,
) -> float:
    """1 when the closed pads meet on the marker and squeeze its narrow side."""
    grip = model.joint("gripper").qposadr.item()
    data.qpos[grip] = model.jnt_range[model.joint("gripper").id][0]
    mujoco.mj_forward(model, data)
    fixed = data.geom("pad_fixed").xpos.copy()
    jaw = data.geom("pad_jaw").xpos.copy()
    if fixed[2] < 0.007 or jaw[2] < 0.007:
        return 0.0
    if float(np.linalg.norm(0.5 * (fixed + jaw) - marker)) > 0.02:
        return 0.0
    span = jaw - fixed
    span[2] = 0.0
    span /= np.linalg.norm(span)
    return abs(float(span @ short))


if __name__ == "__main__":
    main()
