import mediapipe as mp
import numpy as np
from math import atan2

class SquatLogic:
    """
    Side-agnostic logic:
    - Picks LEFT or RIGHT based on higher landmark visibility
    - Depth: hip_y > knee_y + margin  (y is downwards in MediaPipe)
    - Lockout: hip above knee (with margin) AND torso nearly vertical
    """
    def __init__(self,
                 depth_margin=0.02,          # increase margin a bit for stability
                 lockout_margin=0.02,
                 torso_vertical_deg=12,      # <= 12° from vertical
                 min_vis=0.6):               # require solid visibility
        self.P = mp.solutions.pose.PoseLandmark
        self.depth_margin = depth_margin
        self.lockout_margin = lockout_margin
        self.torso_vertical_deg = torso_vertical_deg
        self.min_vis = min_vis

    def _lm(self, landmarks, enum):
        lm = landmarks[self.P[enum].value]
        return lm.x, lm.y, lm.z, lm.visibility

    def _side_vis(self, landmarks, side):  # side = "LEFT" or "RIGHT"
        hip_v = self._lm(landmarks, f"{side}_HIP")[3]
        knee_v = self._lm(landmarks, f"{side}_KNEE")[3]
        sh_v  = self._lm(landmarks, f"{side}_SHOULDER")[3]
        return min(hip_v, knee_v, sh_v)

    def _pick_side(self, landmarks):
        left_vis  = self._side_vis(landmarks, "LEFT")
        right_vis = self._side_vis(landmarks, "RIGHT")
        if left_vis >= right_vis and left_vis >= self.min_vis:
            return "LEFT", left_vis, right_vis
        if right_vis > left_vis and right_vis >= self.min_vis:
            return "RIGHT", left_vis, right_vis
        # neither side confidently visible
        return None, left_vis, right_vis

    def check_depth_and_lockout(self, landmarks):
        side, left_vis, right_vis = self._pick_side(landmarks)
        if side is None:
            return dict(visible_side=None, depth_ok=False, lockout_ok=False,
                        debug={"left_vis": left_vis, "right_vis": right_vis})

        hip_x, hip_y, _, _   = self._lm(landmarks, f"{side}_HIP")
        knee_x, knee_y, _, _ = self._lm(landmarks, f"{side}_KNEE")
        sh_x, sh_y, _, _     = self._lm(landmarks, f"{side}_SHOULDER")

        # ---------- DEPTH CHECK ----------
        # y increases downward. So when squatting, hip_y gets LARGER than knee_y.
        depth_delta = hip_y - knee_y

        # Use a more generous threshold (0.03–0.05 works better for most webcams)
        depth_ok = depth_delta > self.depth_margin

        # ---------- LOCKOUT CHECK ----------
        dx = abs(sh_x - hip_x)
        dy = abs(sh_y - hip_y) + 1e-6
        angle_from_vertical_deg = np.degrees(atan2(dx, dy))

        hip_above_knee = hip_y < (knee_y + self.lockout_margin)
        torso_upright  = angle_from_vertical_deg <= self.torso_vertical_deg
        lockout_ok = hip_above_knee and torso_upright

        return dict(
            visible_side=side,
            depth_ok=depth_ok,
            lockout_ok=lockout_ok,
            debug={
                "left_vis": left_vis,
                "right_vis": right_vis,
                "hip_y": hip_y,
                "knee_y": knee_y,
                "depth_delta": depth_delta,
                "angle_from_vertical_deg": angle_from_vertical_deg
            }
        )

