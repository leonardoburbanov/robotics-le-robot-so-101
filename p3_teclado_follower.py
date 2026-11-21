import time
from pathlib import Path

import mujoco
import numpy as np
import pygame

try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

XML = Path(r"C:\Robotics\robotics-le-robot-so-101\Simulation\SO101\scene.xml")
PORT = "COM4"
ROBOT_ID = "follower"
USE_DEGREES = True
GRIPPER_UNIT = "percent"

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
ROLL = "wrist_roll"
GRIP = "gripper"
JOINTS = ARM + [ROLL, GRIP]
SPEED = 0.05
ROLL_SPEED = np.deg2rad(45.0)
FLEX_SPEED = np.deg2rad(45.0)
GRIP_SPEED = 0.8
LAM = 0.05
DT = 1.0 / 30.0
MAX_STEP = np.deg2rad(2.0)

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
arm_j = np.array([model.joint(n).id for n in ARM])
arm_q = np.array([model.joint(n).qposadr.item() for n in ARM])
arm_dof = np.array([model.joint(n).dofadr.item() for n in ARM])
roll_j = model.joint(ROLL).id
roll_q = model.joint(ROLL).qposadr.item()
grip_j = model.joint(GRIP).id
grip_q = model.joint(GRIP).qposadr.item()
grip_low, grip_high = model.jnt_range[grip_j]
qadr = {n: model.joint(n).qposadr.item() for n in JOINTS}
site = model.site("gripperframe").id


def to_rad(name: str, value: float) -> float:
    if USE_DEGREES and name != GRIP:
        return float(np.deg2rad(value))
    low, high = model.jnt_range[model.joint(name).id]
    return float(low + (value + 100.0) / 200.0 * (high - low))


def action_dict() -> dict[str, float]:
    out = {f"{n}.pos": float(np.rad2deg(data.qpos[qadr[n]])) for n in ARM + [ROLL]}
    if GRIPPER_UNIT == "percent":
        t = (float(data.qpos[grip_q]) - grip_low) / (grip_high - grip_low)
        out["gripper.pos"] = float(np.clip(t, 0.0, 1.0) * 100.0)
    else:
        out["gripper.pos"] = float(np.rad2deg(data.qpos[grip_q]))
    return out


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


robot = SO101Follower(
    SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
)
robot.connect()
obs = robot.get_observation()
for name in JOINTS:
    data.qpos[qadr[name]] = to_rad(name, float(obs[f"{name}.pos"]))
mujoco.mj_forward(model, data)
robot.send_action(action_dict())

pygame.init()
screen = pygame.display.set_mode((420, 80))
pygame.display.set_caption("clic aqui. SPACE + WASD, R/F, Q/E, O/L")
font = pygame.font.SysFont("consolas", 18)

try:
    while True:
        t0 = time.perf_counter()
        pygame.event.pump()
        k = pygame.key.get_pressed()
        v = np.zeros(3)
        if k[pygame.K_SPACE]:
            tip = data.site_xpos[site]
            origin = data.body("shoulder").xpos
            forward = tip[:2] - origin[:2]
            length = float(np.linalg.norm(forward))
            if length < 1e-4:
                forward = np.array([1.0, 0.0])
            else:
                forward = forward / length
            lateral = np.array([-forward[1], forward[0]])
            side = (k[pygame.K_d] - k[pygame.K_a]) * SPEED
            reach = (k[pygame.K_r] - k[pygame.K_f]) * SPEED
            v[0] = forward[0] * reach + lateral[0] * side
            v[1] = forward[1] * reach + lateral[1] * side
            v[2] = (k[pygame.K_w] - k[pygame.K_s]) * SPEED
            roll = data.qpos[roll_q] + (k[pygame.K_e] - k[pygame.K_q]) * ROLL_SPEED * DT
            data.qpos[roll_q] = np.clip(roll, *model.jnt_range[roll_j])
            grip = data.qpos[grip_q] + (k[pygame.K_o] - k[pygame.K_l]) * GRIP_SPEED * DT
            data.qpos[grip_q] = np.clip(grip, *model.jnt_range[grip_j])
        data.qpos[arm_q] = dls_xyz(v)
        if k[pygame.K_SPACE]:
            flex = data.qpos[arm_q[3]] + (k[pygame.K_t] - k[pygame.K_g]) * FLEX_SPEED * DT
            data.qpos[arm_q[3]] = np.clip(flex, -np.deg2rad(85.9), np.deg2rad(85.9))
        mujoco.mj_forward(model, data)
        robot.send_action(action_dict())
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
finally:
    robot.disconnect()
    pygame.quit()