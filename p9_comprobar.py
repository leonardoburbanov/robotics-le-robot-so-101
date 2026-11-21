"""Look once, wait while you move the marker by hand, then say if the task is done."""

import json
import os
from pathlib import Path

import p8_mirar


def main() -> None:
    p8_mirar.load_env(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Falta OPENROUTER_API_KEY en .env.")
    cam = p8_mirar.open_cam(p8_mirar.INDEX)
    try:
        first = p8_mirar.decide(p8_mirar.jpeg_b64(cam.read()))
        print(json.dumps(first, ensure_ascii=False, indent=2))
        input("Mueve el marcador como dice la frase y pulsa Enter... ")
        second = p8_mirar.decide(p8_mirar.jpeg_b64(cam.read()))
    finally:
        cam.disconnect()
    goal_inside = first["task"] == p8_mirar.PUT_INSIDE
    done = second["marker_inside"] is goal_inside
    print(json.dumps({"after": second, "done": done}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
