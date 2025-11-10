import mediapipe as mp
import numpy as np

class SquatLogic:
    def __init__(self, depth_margin=0.02, lockout_thresh=0.05):
        self.depth_margin = depth_margin
        self.lockout_thresh = lockout_thresh
        self.pose_ids = mp.solutions.pose.PoseLandmark

    def get_coords(self, landmarks, name):
        lm = landmarks[self.pose_ids[name].value]
        return np.array([lm.x, lm.y, lm.z])

    def check_depth(self, landmarks):
        hip = self.get_coords(landmarks, "LEFT_HIP")
        knee = self.get_coords(landmarks, "LEFT_KNEE")
        return hip[1] > knee[1] + self.depth_margin

    def check_lockout(self, landmarks):
        hip = self.get_coords(landmarks, "LEFT_HIP")
        knee = self.get_coords(landmarks, "LEFT_KNEE")
        shoulder = self.get_coords(landmarks, "LEFT_SHOULDER")
        torso_angle = abs(shoulder[1] - hip[1])
        return (hip[1] < knee[1] + self.lockout_thresh) and (torso_angle < 0.1)
