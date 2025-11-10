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

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark

    def draw_landmarks(self, frame, landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            mp.solutions.pose.PoseLandmark,
            mp.solutions.pose.POSE_CONNECTIONS
        )
