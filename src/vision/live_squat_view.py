import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.pose_detector import PoseDetector
from vision.squat_logic import SquatLogic

detector = PoseDetector()
logic = SquatLogic()

cap = cv2.VideoCapture(0)
state = "IDLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = detector.get_landmarks(frame)
    if landmarks:
        depth_ok = logic.check_depth(landmarks)
        lockout_ok = logic.check_lockout(landmarks)

        if depth_ok:
            state = "DEPTH_OK"
        elif lockout_ok:
            state = "LOCKOUT_OK"
        else:
            state = "IDLE"

        cv2.putText(frame, f"State: {state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Squat Judge Prototype", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
