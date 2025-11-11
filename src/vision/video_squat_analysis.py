import os, sys
from collections import deque
import os
import sys
import cv2
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.pose_detector import PoseDetector
from vision.squat_logic import SquatLogic

# Usage: python src/vision/test.py [path/to/video.mov]
video_path = sys.argv[1] if len(sys.argv) > 1 else "mov1.mov"
if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

detector = PoseDetector()
logic = SquatLogic()

# State / smoothing buffers
state = "IDLE"
rep_count = 0
prev_hip_y = None
direction = "NONE"
depth_buf, lockout_buf = deque(maxlen=5), deque(maxlen=5)
# Additional smoothing buffers to reduce flicker
hip_y_buf = deque(maxlen=7)            # store recent hip y positions
prev_smoothed_hip_y = None
direction_buf = deque(maxlen=5)        # require direction to be stable for a few frames
# detection presence buffer: helps ignore short landmark dropouts
detect_buf = deque(maxlen=7)
error_reported = False

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

    # track detection presence to avoid reacting to short dropouts
    detect_buf.append(1 if landmarks else 0)
    detection_confident = sum(detect_buf) >= 3

    if landmarks:
        detector.draw(frame, results)

        out = logic.check_depth_and_lockout(landmarks)
        hip_y = out.get("debug", {}).get("hip_y")

        # append hip y and compute smoothed value (median-like via mean after trimming)
        if hip_y is not None:
            hip_y_buf.append(hip_y)
        smoothed_hip_y = None
        if len(hip_y_buf) > 0:
            # simple mean smoothing; using a slightly larger window reduces jitter
            smoothed_hip_y = sum(hip_y_buf) / len(hip_y_buf)

        depth_buf.append(bool(out.get("depth_ok")))
        lockout_buf.append(bool(out.get("lockout_ok")))

        # Determine current direction of motion using smoothed hip positions
        direction_raw = get_motion_dir(smoothed_hip_y, prev_smoothed_hip_y)
        prev_smoothed_hip_y = smoothed_hip_y
        direction_buf.append(direction_raw)
        # require the direction to be stable for at least 3 of the recent frames
        direction_stable = "NONE"
        # count occurrences of each direction in the buffer
        from collections import Counter
        cnt = Counter(direction_buf)
        if cnt:
            most_common, count = cnt.most_common(1)[0]
            if count >= 3 and most_common != "NONE":
                direction_stable = most_common

        # Determine primary state with hysteresis
        if stable_true(depth_buf):
            state = "DEPTH_OK"
        elif stable_true(lockout_buf):
            state = "LOCKOUT_OK"
        else:
            state = direction_stable

        # Rep counter logic (hysteresis-aware)
        if not hasattr(logic, "last_confirmed"):
            logic.last_confirmed = "LOCKOUT_OK"

        if logic.last_confirmed == "LOCKOUT_OK" and state == "DEPTH_OK":
            logic.last_confirmed = "DEPTH_OK"
        elif logic.last_confirmed == "DEPTH_OK" and state == "LOCKOUT_OK":
            rep_count += 1
            logic.last_confirmed = "LOCKOUT_OK"

        # Display overlay
        dbg = out.get("debug", {})
        side = out.get("visible_side") or "NONE"

        # Top primary status: show lockout first if confirmed, otherwise depth, otherwise direction/idle.
        # For ascent/descent show yellow, for none show red. Use bold text (thickness=3).
        if stable_true(lockout_buf):
            top_status = "lockout_ok"
            top_color = (0, 255, 0)  # green
        elif stable_true(depth_buf):
            top_status = "depth_ok"
            top_color = (0, 255, 0)  # green
        else:
            ds = (direction_stable or "idle").upper()
            if ds in ("ASCENT", "DESCENT"):
                top_status = ds.lower()
                top_color = (0, 255, 255)  # yellow
            else:
                top_status = "none"
                top_color = (0, 0, 255)  # red

        # draw bold top status
        cv2.putText(frame, f"State: {top_status}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, top_color, 3)

        cv2.putText(frame, f"Reps: {rep_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # cv2.putText(frame, f"State:{state}", (20, 85),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # cv2.putText(frame, f"DepthΔ:{dbg.get('depth_delta', 0):.3f}", (20, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        # If detection temporarily lost but not sustained, keep previous state to avoid flicker
        if detection_confident:
            # short dropout: don't immediately clear buffers
            cv2.putText(frame, "Brief detection dropout — holding state",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # reset error flag if we had reported earlier
            error_reported = False
        else:
            # sustained no-detect → reset state and buffers
            state = "IDLE"
            direction = "NONE"
            depth_buf.clear()
            lockout_buf.clear()
            hip_y_buf.clear()
            direction_buf.clear()
            prev_smoothed_hip_y = None
            # show error at very top (replace top_status)
            cv2.putText(frame, "no_landmarks_detected", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(frame, "No landmarks detected (stand sideways, full body in view)",
                        (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # print an error message once to stderr
            if not error_reported:
                print("ERROR: No landmarks detected by camera — check camera view and lighting", file=sys.stderr)
                error_reported = True

    cv2.imshow("Video Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()