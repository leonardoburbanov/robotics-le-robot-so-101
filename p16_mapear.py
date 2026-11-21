"""Copy the marker's place on the sheet from one overhead photo into the sim."""

import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import p8_mirar

XML = Path(__file__).resolve().parent / "Simulation" / "SO101" / "scene.xml"
SHOT = Path(__file__).resolve().parent / "p16_mapa.jpg"


def find_layout(frame_bgr: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Find the sheet and the marker in one overhead photo.

    Returns nx, ny of the marker on the sheet (0 is the middle, 1 is the edge),
    the yaw that puts the marker's long side at the same angle, and the frame
    with both boxes drawn.

    ponytail: minAreaRect, no perspective. Holds while the camera looks straight
    down. A 4-point homography is the upgrade if the corners start to drift.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    paper = _paper_rect(hsv)
    marker = _marker_rect(hsv)
    (cx, cy), (width, height), paper_deg = paper
    theta = np.deg2rad(paper_deg)
    axis_x = np.array([np.cos(theta), np.sin(theta)])
    axis_y = np.array([-np.sin(theta), np.cos(theta)])
    delta = np.array(marker[0]) - np.array([cx, cy])
    nx = float(delta @ axis_x / (width / 2.0))
    ny = float(delta @ axis_y / (height / 2.0))
    relative = (_long_deg(marker) - paper_deg + 90.0) % 180.0 - 90.0
    yaw = float(np.deg2rad(relative - 90.0))
    drawn = frame_bgr.copy()
    cv2.drawContours(drawn, [cv2.boxPoints(paper).astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.drawContours(drawn, [cv2.boxPoints(marker).astype(np.int32)], 0, (0, 0, 255), 2)
    return nx, ny, yaw, drawn


def put(model: mujoco.MjModel, data: mujoco.MjData, nx: float, ny: float, yaw: float) -> None:
    """Lay the sim marker on the virtual sheet at the photo's relative spot.

    The sheet stays where the arm can reach. This photo has no arm in it, so
    it cannot say where the sheet sits relative to the base.
    """
    mujoco.mj_forward(model, data)
    paper = data.geom("paper").xpos
    half = model.geom("paper").size
    z = paper[2] + half[2] + model.geom("marker").size[2]
    # Overhead camera: image x grows with sim +X, image y grows down, sim +Y is up.
    yaw = -yaw
    free = model.joint("marker_free").qposadr.item()
    data.qpos[free : free + 7] = [
        paper[0] + nx * half[0],
        paper[1] - ny * half[1],
        z,
        np.cos(yaw / 2.0),
        0.0,
        0.0,
        np.sin(yaw / 2.0),
    ]
    dof = model.joint("marker_free").dofadr.item()
    data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    """Grab one overhead frame, move the sim marker, and show the scene."""
    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        frame = p8_mirar.latest_frame(cam)
    finally:
        cam.disconnect()
    nx, ny, yaw, drawn = find_layout(frame)
    cv2.imwrite(str(SHOT), drawn)
    print(f"nx={nx:.2f} ny={ny:.2f} yaw_deg={np.degrees(yaw):.0f} shot={SHOT.name}")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    put(model, data, nx, ny, yaw)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)


def _paper_rect(hsv: np.ndarray) -> cv2.typing.RotatedRect:
    """Largest cool-white region, plus any piece the marker split off."""
    hue, sat, val = cv2.split(hsv)
    mask = ((hue >= 80) & (hue <= 120) & (sat <= 70) & (val >= 110)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 3000]
    if not contours:
        raise RuntimeError("no veo la hoja blanca")
    contours.sort(key=cv2.contourArea, reverse=True)
    kept = [contours[0]]
    for contour in contours[1:]:
        if _gap(contour, kept) < 80:
            kept.append(contour)
    return cv2.minAreaRect(cv2.convexHull(np.vstack(kept)))


def _marker_rect(hsv: np.ndarray) -> cv2.typing.RotatedRect:
    """Largest yellow body, with the blue caps that touch it."""
    hue, sat, val = cv2.split(hsv)
    yellow = (hue >= 18) & (hue <= 45) & (sat >= 80) & (val >= 150)
    blue = (hue >= 90) & (hue <= 140) & (sat >= 80) & (val >= 80)
    mask = (yellow | blue).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 400]
    if not contours:
        raise RuntimeError("no veo el marcador amarillo")
    return cv2.minAreaRect(max(contours, key=cv2.contourArea))


def _long_deg(rect: cv2.typing.RotatedRect) -> float:
    """Angle of the long side of a minAreaRect, in degrees."""
    (_, _), (width, height), angle = rect
    return angle if width >= height else angle + 90.0


def _gap(contour: np.ndarray, kept: list[np.ndarray]) -> float:
    """Pixel gap between a contour's box and the nearest kept box."""
    x, y, w, h = cv2.boundingRect(contour)
    best = 1e9
    for other in kept:
        ox, oy, ow, oh = cv2.boundingRect(other)
        dx = max(ox - (x + w), x - (ox + ow), 0)
        dy = max(oy - (y + h), y - (oy + oh), 0)
        best = min(best, (dx * dx + dy * dy) ** 0.5)
    return best


if __name__ == "__main__":
    main()
