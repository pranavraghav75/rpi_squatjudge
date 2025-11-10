import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame):
        """Run MediaPipe pose detection and return both results and landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            return results, landmarks
        return None, None

    def draw_landmarks(self, frame, results):
        """Draw skeleton overlay."""
        self.mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
        )
