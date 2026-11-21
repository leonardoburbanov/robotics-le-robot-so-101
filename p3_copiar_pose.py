from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

XML = Path(r"C:\Robotics\robotics-le-robot-so-101\Simulation\SO101\scene.xml")
PORT = "COM4"
ROBOT_ID = "follower"
USE_DEGREES = True
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)


def to_rad(name: str, value: float) -> float:
    low, high = model.jnt_range[model.joint(name).id]
    if name == "gripper":
        return float(low + (value / 100.0) * (high - low))
    if USE_DEGREES:
        return float(np.deg2rad(value))
    return float(low + (value + 100.0) / 200.0 * (high - low))


robot = SO101Follower(
    SO101FollowerConfig(port=PORT, id=ROBOT_ID, disable_torque_on_disconnect=False)
)
robot.connect()
obs = robot.get_observation()
for name in JOINTS:
    data.qpos[model.joint(name).qposadr.item()] = to_rad(name, float(obs[f"{name}.pos"]))
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()

robot.disconnect()