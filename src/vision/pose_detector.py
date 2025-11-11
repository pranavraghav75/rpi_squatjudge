import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results

    def draw(self, frame_bgr, results):
        if not results or not results.pose_landmarks:
            return
        self.mp_draw.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
        )
