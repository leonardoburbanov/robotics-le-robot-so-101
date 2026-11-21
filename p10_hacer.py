"""The camera picks the sentence, you drive the follower, the camera checks."""

import json
import os
from pathlib import Path

import cv2

import p3_teclado_follower
import p8_mirar


def main() -> None:
    p8_mirar.load_env(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Falta OPENROUTER_API_KEY en .env.")

    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        first = p8_mirar.decide(p8_mirar.jpeg_b64(p8_mirar.latest_frame(cam)))
    finally:
        cam.disconnect()
    print(json.dumps(first, ensure_ascii=False, indent=2))
    print("Mismas teclas de siempre. Escape cuando el marcador ya quedó donde pide la frase.")

    p3_teclado_follower.drive(quit_on_escape=True)

    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        after = p8_mirar.latest_frame(cam)
        cv2.imwrite(str(Path(__file__).resolve().parent / "p10_after.jpg"), after)
        second = p8_mirar.decide(p8_mirar.jpeg_b64(after))
    finally:
        cam.disconnect()
    goal_inside = first["task"] == p8_mirar.PUT_INSIDE
    done = second["marker_inside"] == goal_inside
    print(json.dumps({"after": second, "done": done}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
