class SquatJudgeFSM:
    def __init__(self):
        self.state = "IDLE"

    def transition(self, new_state):
        print(f"{self.state} → {new_state}")
        self.state = new_state

    def run(self):
        self.transition("SETUP")
        # Placeholder: later add pose detection, cues, etc.
        self.transition("SQUAT_CMD")
        self.transition("DESCENT")
        self.transition("DEPTH_OK")
        self.transition("ASCENT")
        self.transition("LOCKOUT_OK")
        self.transition("RACK_CMD")
        self.transition("LOG")
        self.transition("IDLE")
