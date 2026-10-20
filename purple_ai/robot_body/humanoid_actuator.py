"""Humanoid robot body interface — replaces desktop automation for a bipedal robot."""
from purple_ai.safe_mode.guardrails import SafeModeGuard


class HumanoidActuator:
    """Controls joints: head (pan/tilt), arms (shoulder/elbow/gripper), legs (hip/knee/ankle)."""

    JOINTS = {
        "head_pan": 0.0, "head_tilt": 0.0,
        "left_arm_shoulder": 0.0, "left_arm_elbow": 0.0, "left_gripper": 0.0,
        "right_arm_shoulder": 0.0, "right_arm_elbow": 0.0, "right_gripper": 0.0,
        "left_hip": 0.0, "left_knee": 0.0, "left_ankle": 0.0,
        "right_hip": 0.0, "right_knee": 0.0, "right_ankle": 0.0,
    }

    def __init__(self, guard: SafeModeGuard = None):
        self.guard = guard or SafeModeGuard(safe=True)
        self.current_pose = self.JOINTS.copy()
        self.fallback_safe_pose = {k: 0.0 for k in self.JOINTS}

    def _allowed(self, action: str) -> bool:
        return self.guard.allow_desktop_action(action)

    def move_head(self, pan: float, tilt: float) -> bool:
        if not self._allowed("move_head"):
            return False
        self.current_pose["head_pan"] = max(-90, min(90, pan))
        self.current_pose["head_tilt"] = max(-45, min(45, tilt))
        return True

    def move_arm(self, side: str, shoulder: float, elbow: float) -> bool:
        if not self._allowed("move_arm"):
            return False
        key_s = f"{side}_arm_shoulder"
        key_e = f"{side}_arm_elbow"
        self.current_pose[key_s] = max(-180, min(180, shoulder))
        self.current_pose[key_e] = max(0, min(140, elbow))
        return True

    def gripper(self, side: str, close: float) -> bool:
        if not self._allowed("gripper"):
            return False
        self.current_pose[f"{side}_gripper"] = max(0.0, min(1.0, close))
        return True

    def move_leg(self, side: str, hip: float, knee: float, ankle: float) -> bool:
        if not self._allowed("move_leg"):
            return False
        self.current_pose[f"{side}_hip"] = max(-30, min(30, hip))
        self.current_pose[f"{side}_knee"] = max(-10, min(90, knee))
        self.current_pose[f"{side}_ankle"] = max(-30, min(30, ankle))
        return True

    def stand(self) -> bool:
        if not self._allowed("stand"):
            return False
        self.current_pose = {
            "head_pan": 0, "head_tilt": 0,
            "left_arm_shoulder": 0, "left_arm_elbow": 0, "left_gripper": 0,
            "right_arm_shoulder": 0, "right_arm_elbow": 0, "right_gripper": 0,
            "left_hip": 0, "left_knee": 0, "left_ankle": 0,
            "right_hip": 0, "right_knee": 0, "right_ankle": 0,
        }
        return True

    def sit_down(self) -> bool:
        if not self._allowed("sit_down"):
            return False
        # Safe seated pose
        self.current_pose["left_hip"] = 45
        self.current_pose["left_knee"] = 90
        self.current_pose["right_hip"] = 45
        self.current_pose["right_knee"] = 90
        return True
