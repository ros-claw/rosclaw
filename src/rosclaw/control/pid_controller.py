"""PID Controller for mobile base motion control.

Simple PID implementation for differential-drive or holonomic mobile bases.
Can operate in open-loop (mock) or closed-loop (with feedback) mode.
"""

from dataclasses import dataclass


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.1


class PIDController:
    """Discrete-time PID controller for position/velocity control.

    Usage:
        pid = PIDController(PIDGains(kp=2.0, ki=0.1, kd=0.5))
        while not_at_target:
            error = target - current_position
            cmd = pid.update(error, dt=0.01)
            robot.set_velocity(cmd)
    """

    def __init__(self, gains: PIDGains | None = None) -> None:
        self.gains = gains or PIDGains()
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._output_limit: tuple[float, float] = (-float("inf"), float("inf"))
        self._integral_limit: float = float("inf")

    def set_output_limit(self, min_val: float, max_val: float) -> None:
        """Clamp controller output to [min_val, max_val]."""
        self._output_limit = (min_val, max_val)

    def set_integral_limit(self, limit: float) -> None:
        """Clamp integral term to prevent windup."""
        self._integral_limit = limit

    def reset(self) -> None:
        """Reset internal state (integral, previous error)."""
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """Compute PID output for given error and timestep.

        Args:
            error: setpoint - measured_value
            dt: time step in seconds (must be > 0)

        Returns:
            Control output (e.g. velocity command).
        """
        if dt <= 0:
            return 0.0

        # Proportional
        p = self.gains.kp * error

        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = max(-self._integral_limit, min(self._integral_limit, self._integral))
        i = self.gains.ki * self._integral

        # Derivative (on error)
        d = self.gains.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p + i + d
        return max(self._output_limit[0], min(self._output_limit[1], output))

    def simulate_step(
        self,
        current: float,
        target: float,
        dt: float,
        plant_gain: float = 1.0,
    ) -> tuple[float, float]:
        """Simulate one control step: compute command and apply to plant.

        Args:
            current: Current measured value
            target: Target setpoint
            dt: Timestep
            plant_gain: Plant transfer gain (output -> state change)

        Returns:
            (new_position, command)
        """
        error = target - current
        cmd = self.update(error, dt)
        new_pos = current + cmd * plant_gain * dt
        return new_pos, cmd
