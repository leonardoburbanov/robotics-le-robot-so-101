"""The arm goes to the marker, closes, and lifts it. Kinematic only."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

XML = Path(__file__).resolve().parent / "Simulation" / "SO101" / "scene.xml"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
GRIP = "gripper"
LAM = 0.05
DT = 1.0 / 30.0
MAX_STEP = np.deg2rad(2.0)
SPEED = 0.05


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the scene with the gripper open."""
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    grip = model.joint(GRIP).qposadr.item()
    data.qpos[grip] = model.jnt_range[model.joint(GRIP).id][1]
    mujoco.mj_forward(model, data)
    return model, data


def _arm(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, int]:
    arm_q = np.array([model.joint(name).qposadr.item() for name in ARM])
    arm_dof = np.array([model.joint(name).dofadr.item() for name in ARM])
    return arm_q, arm_dof, model.site("gripperframe").id


def _step(model: mujoco.MjModel, data: mujoco.MjData, velocity: np.ndarray) -> None:
    """One damped-least-squares step, same limits as the keyboard teleop."""
    arm_q, arm_dof, site = _arm(model)
    arm_j = np.array([model.joint(name).id for name in ARM])
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site)
    jacobian = jacp[:, arm_dof]
    dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + (LAM**2) * np.eye(3), velocity)
    q = data.qpos[arm_q] + np.clip(dq * DT, -MAX_STEP, MAX_STEP)
    low, high = model.jnt_range[arm_j].T
    q = np.clip(q, low, high)
    q[3] = np.clip(q[3], -np.deg2rad(85.9), np.deg2rad(85.9))
    data.qpos[arm_q] = q


def _go(model: mujoco.MjModel, data: mujoco.MjData, goal: np.ndarray, draw) -> None:
    _, _, site = _arm(model)
    for _ in range(600):
        error = goal - data.site_xpos[site]
        if float(np.linalg.norm(error)) < 0.012:
            return
        _step(model, data, np.clip(error, -SPEED, SPEED))
        mujoco.mj_forward(model, data)
        draw()
    raise RuntimeError(f"la pinza no llego: {data.site_xpos[site]}")


def pick(model: mujoco.MjModel, data: mujoco.MjData, draw=None) -> None:
    """Hover, go down, close, and lift. The marker is stuck to the tip after the close.

    ponytail: no contact. The free joint is written by hand. Upgrade path is mj_step
    once the jaw mesh can pinch the box.
    """
    if draw is None:
        draw = lambda: None
    _, _, site = _arm(model)
    grip = model.joint(GRIP).qposadr.item()
    grip_low = model.jnt_range[model.joint(GRIP).id][0]
    free = model.joint("marker_free").qposadr.item()
    marker = data.geom("marker").xpos.copy()

    _go(model, data, marker + np.array([0.0, 0.0, 0.08]), draw)
    _go(model, data, marker + np.array([0.0, 0.0, 0.012]), draw)
    for _ in range(20):
        data.qpos[grip] = max(grip_low, data.qpos[grip] - 0.08)
        mujoco.mj_forward(model, data)
        draw()
    offset = data.geom("marker").xpos - data.site_xpos[site]
    for _ in range(90):
        _step(model, data, np.array([0.0, 0.0, SPEED]))
        mujoco.mj_forward(model, data)
        data.qpos[free : free + 3] = data.site_xpos[site] + offset
        mujoco.mj_forward(model, data)
        draw()
    assert data.geom("marker").xpos[2] > 0.08, "el marcador no subio"


def main() -> None:
    """Play the pick in the viewer, then hold the last frame."""
    model, data = load()
    with mujoco.viewer.launch_passive(model, data) as viewer:

        def draw() -> None:
            viewer.sync()
            time.sleep(DT)

        pick(model, data, draw)
        while viewer.is_running():
            viewer.sync()
            time.sleep(DT)


if __name__ == "__main__":
    main()
