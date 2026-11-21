# pip install mediapipe opencv-python
# hand: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
# pose: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
import os
import signal

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# Pose indices. The forearm is the segment between elbow and wrist.
FOREARMS = ((13, 15), (14, 16))
# Hand index 0 is the wrist. 5, 9, 13, 17 are the knuckles.
PALM = (5, 9, 13, 17)
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

stop = False


def request_stop(signum: int, _frame: object) -> None:
    """Stop the loop. Raising here deadlocks MediaPipe's worker threads."""
    global stop
    stop = True


def forearm_points(elbow_x: float, elbow_y: float, wrist_x: float, wrist_y: float) -> list[tuple[float, float]]:
    """Return the elbow, the mid-forearm, and the wrist in normalized coordinates."""
    return [
        (elbow_x, elbow_y),
        ((elbow_x + wrist_x) / 2, (elbow_y + wrist_y) / 2),
        (wrist_x, wrist_y),
    ]


def for_detection(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Raise local contrast so a hand survives a blown-out white ceiling."""
    lightness, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((CLAHE.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


def forearm_from_hand(wrist_x: float, wrist_y: float, palm_x: float, palm_y: float) -> list[tuple[float, float]]:
    """Return the wrist and two points up the forearm, estimated from the palm."""
    dx, dy = wrist_x - palm_x, wrist_y - palm_y
    return [(wrist_x, wrist_y), (wrist_x + dx, wrist_y + dy), (wrist_x + 2 * dx, wrist_y + 2 * dy)]


signal.signal(signal.SIGINT, request_stop)

hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))

# 0 = gripper, 1 = leader, 2 = the spare webcam.
cap, t = cv2.VideoCapture(2, cv2.CAP_DSHOW), 0
# 128 blows the ceiling out to pure white and the wrist disappears into it.
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 80)
try:
    while cap.isOpened() and not stop:
        ok, frame = cap.read()
        if not ok:
            break
        t += 33
        detected = for_detection(frame)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        hands = hand.detect_for_video(img, t)
        bodies = pose.detect_for_video(img, t)
        h, w = frame.shape[:2]
        for marks, world in zip(hands.hand_landmarks, hands.hand_world_landmarks):
            for p in marks[1:]:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)
            wrist = marks[0]
            center = (int(wrist.x * w), int(wrist.y * h))
            cv2.circle(frame, center, 9, (0, 0, 0), 2)
            cv2.circle(frame, center, 6, (255, 80, 0), -1)
            palm_x = sum(marks[i].x for i in PALM) / 4
            palm_y = sum(marks[i].y for i in PALM) / 4
            pts = [(int(x * w), int(y * h)) for x, y in forearm_from_hand(wrist.x, wrist.y, palm_x, palm_y)]
            cv2.line(frame, pts[0], pts[2], (255, 80, 0), 2)
            for pt in pts[1:]:
                cv2.circle(frame, pt, 5, (255, 80, 0), -1)
            # world[i].x/y/z → meters, origin at the center of the hand
            pinch = ((world[4].x - world[8].x) ** 2 + (world[4].y - world[8].y) ** 2 + (world[4].z - world[8].z) ** 2) ** 0.5
            cv2.putText(frame, f"pinch {pinch * 100:.1f} cm", (10, 30), 0, 1, (0, 255, 0), 2)
        for body in bodies.pose_landmarks:
            for elbow_i, wrist_i in FOREARMS:
                elbow, wrist = body[elbow_i], body[wrist_i]
                if (elbow.visibility or 0) < 0.25 or (wrist.visibility or 0) < 0.25:
                    continue
                pts = [(int(x * w), int(y * h)) for x, y in forearm_points(elbow.x, elbow.y, wrist.x, wrist.y)]
                cv2.line(frame, pts[0], pts[2], (0, 140, 255), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 6, (0, 140, 255), -1)
        cv2.imshow("hands", frame)
        if cv2.waitKey(1) == 27:
            break
finally:
    for node in (hand, pose):
        try:
            node.close()
        except Exception:
            pass
    if brightness >= 0:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.release()
    cv2.destroyAllWindows()

# ponytail: MediaPipe worker threads are non-daemon and block shutdown after Ctrl+C.
os._exit(0)
