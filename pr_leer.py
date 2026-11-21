try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM4"
ROBOT_ID = "follower"

robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
robot.connect()
obs = robot.get_observation()
for key in sorted(obs):
    if key.endswith(".pos"):
        print(f"{key:22} {obs[key]}")
robot.disconnect()