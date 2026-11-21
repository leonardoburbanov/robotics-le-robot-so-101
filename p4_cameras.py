import pygame

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Backends, OpenCVCameraConfig

INDICES = [0, 1]


def open_cam(index: int) -> OpenCVCamera:
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


cams = {i: open_cam(i) for i in INDICES}
pygame.init()
screen = pygame.display.set_mode((640 * len(INDICES), 480))
pygame.display.set_caption("q cierra. La ventana de la izquierda es el primer indice")
font = pygame.font.SysFont("consolas", 24)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise SystemExit
        for n, i in enumerate(INDICES):
            frame = cams[i].read()
            rgb = frame[:, :, ::-1].tobytes()
            image = pygame.image.frombuffer(rgb, (640, 480), "RGB")
            screen.blit(image, (n * 640, 0))
            screen.blit(font.render(str(i), True, (255, 255, 0)), (n * 640 + 8, 8))
        pygame.display.flip()
finally:
    for cam in cams.values():
        cam.disconnect()
    pygame.quit()