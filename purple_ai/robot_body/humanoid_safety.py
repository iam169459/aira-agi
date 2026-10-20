"""Physical safety systems for the humanoid robot."""
from enum import Enum
import time


class Risk(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class HumanoidSafety:
    """Fall detection, collision proximity, motor overload, and emergency stop."""

    def __init__(self, actuator=None, balance=None):
        self.actuator = actuator
        self.balance = balance
        self.emergency_stopped = False
        self.collision_distance_cm = 500.0
        self.min_safe_distance_cm = 30.0
        self.motor_load = 0.0
        self.last_obstacle_scan = 0.0
        self.safety_log = []

    def log(self, event: str, risk: Risk = Risk.LOW):
        entry = {"time": time.time(), "event": event, "risk": risk.name}
        self.safety_log.append(entry)
        if len(self.safety_log) > 200:
            self.safety_log = self.safety_log[-200:]

    def check_fall(self) -> Risk:
        if self.balance and self.balance.is_fallen():
            self.log("fall detected", Risk.CRITICAL)
            self.emergency_stopped = True
            return Risk.CRITICAL
        return Risk.NONE

    def check_obstacle(self, distance_cm: float) -> Risk:
        self.collision_distance_cm = distance_cm
        if distance_cm < self.min_safe_distance_cm:
            self.log(f"obstacle at {distance_cm:.1f}cm", Risk.CRITICAL)
            self.emergency_stopped = True
            if self.balance:
                self.balance.stop()
            return Risk.CRITICAL
        if distance_cm < self.min_safe_distance_cm * 3:
            self.log(f"near obstacle {distance_cm:.1f}cm", Risk.MEDIUM)
            return Risk.MEDIUM
        if distance_cm < 100:
            self.log(f"obstacle ahead {distance_cm:.1f}cm", Risk.LOW)
            return Risk.LOW
        return Risk.NONE

    def check_motor_load(self, load: float) -> Risk:
        self.motor_load = load
        if load > 0.95:
            self.log(f"motor overload {load:.2f}", Risk.HIGH)
            return Risk.HIGH
        if load > 0.8:
            return Risk.MEDIUM
        return Risk.NONE

    def emergency_stop(self) -> bool:
        self.emergency_stopped = True
        if self.balance:
            self.balance.stop()
        if self.actuator:
            self.actuator.sit_down()
        self.log("emergency stop triggered", Risk.CRITICAL)
        return True

    def reset(self) -> bool:
        if self.balance and self.balance.fallen:
            return False
        self.emergency_stopped = False
        self.log("safety reset", Risk.NONE)
        return True

    def can_act(self) -> bool:
        return not self.emergency_stopped
