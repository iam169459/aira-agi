"""Balance + locomotion controller for bipedal humanoid (IMU-driven)."""
import math
import time


class BalanceController:
    """Keeps the humanoid upright using gyro/accel feedback and gait commands."""

    def __init__(self, actuator=None, loop_hz: int = 100):
        self.actuator = actuator
        self.loop_hz = loop_hz
        self.pitch = 0.0
        self.roll = 0.0
        self.walking = False
        self.fallen = False
        self.step_phase = 0.0
        self._last_loop = time.time()

    def read_imu(self, gyro_x=0.0, gyro_y=0.0, accel_x=0.0, accel_y=0.0, accel_z=9.81):
        """Feed IMU samples. Tilt estimated from accelerometer; gyro used for rate."""
        self.pitch = math.atan2(accel_x, math.sqrt(accel_y ** 2 + accel_z ** 2))
        self.roll = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2))
        return self.pitch, self.roll

    def is_fallen(self, pitch_threshold=0.6, roll_threshold=0.6) -> bool:
        if abs(self.pitch) > pitch_threshold or abs(self.roll) > roll_threshold:
            self.fallen = True
            if self.actuator:
                self.actuator.sit_down()
            return True
        return False

    def balance_step(self) -> bool:
        """One corrective micro-step to counteract tilt. Returns False if fallen."""
        if self.is_fallen():
            return False
        correction = -self.pitch * 20
        if self.actuator:
            self.actuator.move_leg("left", hip=correction, knee=0, ankle=-correction * 0.5)
            self.actuator.move_leg("right", hip=-correction * 0.5, knee=0, ankle=correction * 0.25)
        return True

    def walk(self, duration_s: float = 2.0, speed: float = 1.0) -> bool:
        """Alternate leg swings for a simple gait."""
        if self.fallen or not self.actuator:
            return False
        self.walking = True
        steps = int(duration_s * self.loop_hz / 2)
        for i in range(steps):
            if self.is_fallen():
                self.walking = False
                return False
            phase = (self.step_phase + 0.08 * speed) % 1.0
            self.step_phase = phase
            swing = math.sin(phase * 2 * math.pi) * 20 * speed
            self.actuator.move_leg("left", hip=swing, knee=abs(swing) * 0.5, ankle=-swing * 0.3)
            self.actuator.move_leg("right", hip=-swing, knee=abs(swing) * 0.5, ankle=swing * 0.3)
            time.sleep(1.0 / self.loop_hz)
        self.walking = False
        return True

    def stop(self) -> bool:
        self.walking = False
        if self.actuator:
            self.actuator.move_leg("left", 0, 0, 0)
            self.actuator.move_leg("right", 0, 0, 0)
        return True
