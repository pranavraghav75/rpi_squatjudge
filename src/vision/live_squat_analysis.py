import os, sys, cv2
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.pose_detector import PoseDetector
from vision.squat_logic import SquatLogic

detector = PoseDetector()
logic = SquatLogic()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# --- state vars ---
state = "IDLE"
rep_count = 0
prev_hip_y = None
direction = "NONE"
depth_buf, lockout_buf = deque(maxlen=5), deque(maxlen=5)

def stable_true(buf, min_trues=3):
    return sum(buf) >= min_trues

def get_motion_dir(curr_hip_y, prev_hip_y, tol=0.002):
    if prev_hip_y is None:
        return "NONE"
    delta = curr_hip_y - prev_hip_y
    if abs(delta) < tol:
        return "NONE"
    return "DESCENT" if delta > 0 else "ASCENT"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = detector.process(frame)
    landmarks = results.pose_landmarks.landmark if (results and results.pose_landmarks) else None

    if landmarks:
        detector.draw(frame, results)

        out = logic.check_depth_and_lockout(landmarks)
        hip_y = out["debug"]["hip_y"]
        depth_buf.append(bool(out["depth_ok"]))
        lockout_buf.append(bool(out["lockout_ok"]))

        # Determine current direction of motion
        direction = get_motion_dir(hip_y, prev_hip_y)
        prev_hip_y = hip_y

        # Determine primary state
        if stable_true(depth_buf):
            state = "DEPTH_OK"
        elif stable_true(lockout_buf):
            state = "LOCKOUT_OK"
        else:
            # not in either zone → show motion direction if any
            state = direction

        # Rep counter logic
        # Transition: LOCKOUT_OK → DEPTH_OK → LOCKOUT_OK counts as one rep
        # We'll remember the last confirmed state
        if not hasattr(logic, "last_confirmed"):
            logic.last_confirmed = "LOCKOUT_OK"

        if logic.last_confirmed == "LOCKOUT_OK" and state == "DEPTH_OK":
            logic.last_confirmed = "DEPTH_OK"
        elif logic.last_confirmed == "DEPTH_OK" and state == "LOCKOUT_OK":
            rep_count += 1
            logic.last_confirmed = "LOCKOUT_OK"

        # --- Display overlay ---
        dbg = out["debug"]
        side = out["visible_side"] or "NONE"
        cv2.putText(frame, f"Side:{side}  Reps:{rep_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"State:{state}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"DepthΔ:{dbg.get('depth_delta', 0):.3f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        state = "IDLE"
        direction = "NONE"
        depth_buf.clear()
        lockout_buf.clear()
        cv2.putText(frame, "No landmarks detected (stand sideways, full body in view)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Squat Judge Prototype", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()