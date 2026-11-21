import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pygame

XML = Path(r"C:\Robotics\robotics-le-robot-so-101\Simulation\SO101\scene.xml")
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
SPEED = 0.05
LAM = 0.05
DT = 1.0 / 30.0
MAX_STEP = np.deg2rad(2.0)

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
arm_j = np.array([model.joint(n).id for n in ARM])
arm_q = np.array([model.joint(n).qposadr.item() for n in ARM])
arm_dof = np.array([model.joint(n).dofadr.item() for n in ARM])
site = model.site("gripperframe").id
roll_j = model.joint("wrist_roll").id
roll_q = model.joint("wrist_roll").qposadr.item()
ROLL_SPEED = np.deg2rad(45.0)  # rad/s
grip_j = model.joint("gripper").id
grip_q = model.joint("gripper").qposadr.item()
GRIP_SPEED = 0.8  # rad/s

def dls_xyz(v_xyz: np.ndarray) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site)
    J = jacp[:, arm_dof]
    dq = J.T @ np.linalg.solve(J @ J.T + (LAM**2) * np.eye(3), v_xyz)
    q = data.qpos[arm_q] + np.clip(dq * DT, -MAX_STEP, MAX_STEP)
    low, high = model.jnt_range[arm_j].T
    q = np.clip(q, low, high)
    q[3] = np.clip(q[3], -np.deg2rad(85.9), np.deg2rad(85.9))
    return q


pygame.init()
screen = pygame.display.set_mode((420, 80))
pygame.display.set_caption("clic aqui. SPACE + WASD, R/F")
font = pygame.font.SysFont("consolas", 18)

mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.perf_counter()
        pygame.event.pump()
        k = pygame.key.get_pressed()
        v = np.zeros(3)
        if k[pygame.K_SPACE]:
            v[0] = (k[pygame.K_d] - k[pygame.K_a]) * SPEED
            v[1] = (k[pygame.K_w] - k[pygame.K_s]) * SPEED
            v[2] = (k[pygame.K_r] - k[pygame.K_f]) * SPEED
            roll = data.qpos[roll_q] + (k[pygame.K_e] - k[pygame.K_q]) * ROLL_SPEED * DT
            data.qpos[roll_q] = np.clip(roll, *model.jnt_range[roll_j])
            grip = data.qpos[grip_q] + (k[pygame.K_o] - k[pygame.K_l]) * GRIP_SPEED * DT
            data.qpos[grip_q] = np.clip(grip, *model.jnt_range[grip_j])
        data.qpos[arm_q] = dls_xyz(v)
        mujoco.mj_forward(model, data)
        viewer.sync()
        p = data.site_xpos[site]
        screen.fill((0, 0, 0))
        screen.blit(
            font.render(f"pinza {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}", True, (220, 220, 220)),
            (12, 28),
        )
        pygame.display.flip()
        delay = DT - (time.perf_counter() - t0)
        if delay > 0:
            time.sleep(delay)

pygame.quit()