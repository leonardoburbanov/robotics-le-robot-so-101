"""Show the leader camera so it can be aimed by hand. Does not connect the arm."""

import pygame

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Backends, OpenCVCameraConfig

# Right window in p4_cameras.py: the camera mounted on the leader.
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


cam = open_cam(INDEX)
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("camara del leader. q cierra")
font = pygame.font.SysFont("consolas", 24)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise SystemExit
        frame = cam.read()
        rgb = frame[:, :, ::-1].tobytes()
        screen.blit(pygame.image.frombuffer(rgb, (640, 480), "RGB"), (0, 0))
        screen.blit(font.render(f"leader cam {INDEX}", True, (255, 255, 0)), (8, 8))
        pygame.display.flip()
finally:
    cam.disconnect()
    pygame.quit()
