"""Pose the leader by hand, then hold that pose with torque on."""

import pygame
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Backends, OpenCVCameraConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

PORT = "COM3"
LEADER_ID = "my_awesome_leader_arm"
INDEX = 1


def open_cam(index: int) -> OpenCVCamera:
    """Open one camera at 640x480."""
    cam = OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=index,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            backend=Cv2Backends.DSHOW,
        )
    )
    cam.connect()
    return cam


def hold(leader: SO101Leader) -> None:
    """Command the pose the arm is in, then enable torque."""
    pose = leader.get_action()
    leader.send_feedback(pose)
    leader.enable_torque()


print("Sujeta el leader. Al conectar, el torque se apaga.")
input("Enter para conectar... ")

leader = SO101Leader(SO101LeaderConfig(port=PORT, id=LEADER_ID))
leader.connect()
cam = open_cam(INDEX)
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("mueve el leader a mano. q lo deja sujeto")
font = pygame.font.SysFont("consolas", 24)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                raise SystemExit
        frame = cam.read()
        screen.blit(pygame.image.frombuffer(frame[:, :, ::-1].tobytes(), (640, 480), "RGB"), (0, 0))
        screen.blit(font.render("q deja el leader sujeto", True, (255, 255, 0)), (8, 8))
        pygame.display.flip()
finally:
    if leader.is_connected:
        hold(leader)
        leader.bus.disconnect(disable_torque=False)
    cam.disconnect()
    pygame.quit()
