import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

XML = Path(r"C:\Robotics\robotics-le-robot-so-101\Simulation\SO101\scene.xml")
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
LAM = 0.05
DT = 1.0 / 30.0
MAX_STEP = np.deg2rad(2.0)

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
arm_j = np.array([model.joint(n).id for n in ARM])
arm_q = np.array([model.joint(n).qposadr.item() for n in ARM])
arm_dof = np.array([model.joint(n).dofadr.item() for n in ARM])
site = model.site("gripperframe").id


def dls_xyz(v_xyz: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site)
    J = jacp[:, arm_dof]
    dq = J.T @ np.linalg.solve(J @ J.T + (LAM**2) * np.eye(3), v_xyz)
    q = data.qpos[arm_q] + np.clip(dq * DT, -MAX_STEP, MAX_STEP)
    low, high = model.jnt_range[arm_j].T
    return np.clip(q, low, high)


mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.qpos[arm_q] = dls_xyz(np.array([0.0, 0.0, 0.03]))
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(DT)