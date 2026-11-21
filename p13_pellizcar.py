"""The jaw pinches the marker and lifts it by contact."""

import time

import mujoco
import mujoco.viewer
import numpy as np

import p12_agarrar as kin

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
HELD = ARM + ["wrist_roll"]
SUBSTEPS = 8


def _ctrl(model: mujoco.MjModel, name: str) -> int:
    return model.actuator(name).id


def _hold(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name in HELD:
        data.ctrl[_ctrl(model, name)] = float(data.qpos[model.joint(name).qposadr].item())


def _physics(model: mujoco.MjModel, data: mujoco.MjData, draw) -> None:
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)
    draw()


def _drive(model: mujoco.MjModel, data: mujoco.MjData, velocity: np.ndarray, grip: float, steps: int, draw) -> None:
    """Move the tip. Joint targets go to the actuators, not straight into qpos."""
    arm_q = np.array([model.joint(name).qposadr.item() for name in ARM])
    arm_dof = np.array([model.joint(name).dofadr.item() for name in ARM])
    arm_j = np.array([model.joint(name).id for name in ARM])
    site = model.site("gripperframe").id
    for _ in range(steps):
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site)
        jacobian = jacp[:, arm_dof]
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + (kin.LAM**2) * np.eye(3), velocity)
        q = data.qpos[arm_q] + np.clip(dq * kin.DT, -kin.MAX_STEP, kin.MAX_STEP)
        low, high = model.jnt_range[arm_j].T
        q = np.clip(q, low, high)
        q[3] = np.clip(q[3], -np.deg2rad(85.9), np.deg2rad(85.9))
        for name, value in zip(ARM, q):
            data.ctrl[_ctrl(model, name)] = float(value)
        data.ctrl[_ctrl(model, "wrist_roll")] = float(data.qpos[model.joint("wrist_roll").qposadr].item())
        data.ctrl[_ctrl(model, "gripper")] = grip
        _physics(model, data, draw)


def _lift(model: mujoco.MjModel, data: mujoco.MjData, closed: float, draw) -> None:
    """Raise the tip while the jaw stays closed."""
    _drive(model, data, np.array([0.0, 0.0, kin.SPEED]), closed, 80, draw)


def pinch(model: mujoco.MjModel, data: mujoco.MjData, draw=None) -> None:
    """Approach, close slowly, and lift. The marker stays up only while the pads squeeze it."""
    if draw is None:
        draw = lambda: None
    model.actuator_forcerange[:] = np.array([-40.0, 40.0])
    model.actuator_forcerange[_ctrl(model, "gripper")] = np.array([-1.2, 1.2])
    marker = data.geom("marker").xpos.copy()
    kin._go(model, data, marker + np.array([0.0, 0.0, -0.01]), draw)
    data.qvel[:] = 0
    _hold(model, data)
    grip = _ctrl(model, "gripper")
    low = float(model.jnt_range[model.joint("gripper").id][0])
    high = float(data.qpos[model.joint("gripper").qposadr].item())
    for step in range(400):
        data.ctrl[grip] = high + (low - high) * min(1.0, step / 320.0)
        _physics(model, data, draw)
    _lift(model, data, low, draw)
    height = float(data.geom("marker").xpos[2])
    gap = float(np.linalg.norm(data.geom("marker").xpos - data.site("gripperframe").xpos))
    assert height > 0.06, "el marcador no subio con el pellizco"
    assert gap < 0.05, "el marcador se solto de la pinza"


def main() -> None:
    """Play the pinch in the viewer, then hold the last frame."""
    model, data = kin.load()
    with mujoco.viewer.launch_passive(model, data) as viewer:

        def draw() -> None:
            viewer.sync()
            time.sleep(kin.DT)

        pinch(model, data, draw)
        while viewer.is_running():
            viewer.sync()
            time.sleep(kin.DT)


if __name__ == "__main__":
    main()
