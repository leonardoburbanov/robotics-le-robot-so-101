"""One photo from the leader camera. A VLM picks one of the two task strings."""

import base64
import json
import os
import urllib.request
from pathlib import Path

import cv2
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Backends, OpenCVCameraConfig

INDEX = 1
MODEL = "anthropic/claude-sonnet-4.5"
PUT_INSIDE = "Pick the yellow marker outside the white paper and put it completely inside the paper."
PUT_OUTSIDE = "Pick the yellow marker inside the white paper and put it completely outside the paper."


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


def latest_frame(cam: OpenCVCamera) -> cv2.typing.MatLike:
    """Return a frame from after the camera has already been running."""
    frame = cam.read()
    for _ in range(8):
        frame = cam.read()
    return frame


def jpeg_b64(frame_bgr) -> str:
    """Encode a BGR frame as a base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("no se pudo codificar la foto")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def ask(prompt: str, photo_b64: str) -> str:
    """Send one overhead photo and a prompt. Return the text reply."""
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"},
                        },
                    ],
                }
            ],
        }
    ).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request) as response:
        reply = json.load(response)
    return reply["choices"][0]["message"]["content"].strip()


def decide(photo_b64: str) -> dict:
    """Ask the VLM whether the marker is inside the paper and which task applies."""
    prompt = (
        "Look at this overhead photo of a table. There is a white paper and one yellow marker.\n"
        "A white robot arm may be in the photo. Ignore the arm and the gripper.\n"
        "Answer with JSON only, no markdown:\n"
        '{"marker_inside": true or false, "task": "<one sentence>"}\n'
        "marker_inside is true when the yellow marker is lying on the white paper.\n"
        "It is false when the marker is off the paper or only touching the edge.\n"
        "If it is outside, task must be exactly:\n"
        f"{PUT_INSIDE}\n"
        "If it is inside, task must be exactly:\n"
        f"{PUT_OUTSIDE}"
    )
    text = ask(prompt, photo_b64)
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    allowed = {PUT_INSIDE, PUT_OUTSIDE}
    if data.get("task") not in allowed or not isinstance(data.get("marker_inside"), bool):
        raise RuntimeError(f"respuesta fuera de las dos frases: {text}")
    return data


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file. Existing variables stay."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_env(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Falta OPENROUTER_API_KEY en .env.")
    cam = open_cam(INDEX)
    try:
        frame = cam.read()
    finally:
        cam.disconnect()
    print(json.dumps(decide(jpeg_b64(frame)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
