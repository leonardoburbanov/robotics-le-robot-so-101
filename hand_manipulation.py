# pip install mediapipe opencv-python
# modelo: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

opts = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2)
lm = vision.HandLandmarker.create_from_options(opts)

# 0 = gripper, 1 = leader, 2 = the spare webcam.
cap, t = cv2.VideoCapture(2, cv2.CAP_DSHOW), 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    t += 33
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, t)
    h, w = frame.shape[:2]
    for hand, world in zip(res.hand_landmarks, res.hand_world_landmarks):
        for p in hand:
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0, 255, 0), -1)
        # world[i].x/y/z → metros, origen en el centro de la mano
        pinch = ((world[4].x-world[8].x)**2 + (world[4].y-world[8].y)**2 + (world[4].z-world[8].z)**2)**0.5
        cv2.putText(frame, f"pinch {pinch*100:.1f} cm", (10, 30), 0, 1, (0, 255, 0), 2)
    cv2.imshow("hands", frame)
    if cv2.waitKey(1) == 27: break