"""Longitudinal speed controller for the Modular SDC pipeline."""

from __future__ import annotations

import numpy as np
from typing import Optional

class LongitudinalController:
    """PID-based throttle and brake controller.

    Parameters are tuned for the CarRacing simulator but can be adjusted at
    runtime. The controller keeps a small amount of state to implement the
    integral and derivative terms.
    """

    def __init__(
        self,
        KP: float = 0.08,
        KI: float = 0.01,
        KD: float = 0.02,
        integral_windup_limit: float = 10.0,
        max_gas: float = 0.8,
        max_brake: float = 0.8,
        default_dt: float = 1.0 / 50.0,
    ) -> None:
        self.KP = float(KP)
        self.KI = float(KI)
        self.KD = float(KD)
        self.integral_windup_limit = float(integral_windup_limit)
        self.max_gas = float(max_gas)
        self.max_brake = float(max_brake)
        self.default_dt = float(default_dt)

        self.last_error: float = 0.0
        self.sum_error: float = 0.0
        self.last_control: float = 0.0
        self.speed_history: list[float] = []
        self.target_speed_history: list[float] = []
        self.step_history: list[int] = []

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear controller state before a new episode."""

        self.last_error = 0.0
        self.sum_error = 0.0
        self.last_control = 0.0
        self.speed_history.clear()
        self.target_speed_history.clear()
        self.step_history.clear()

    # ------------------------------------------------------------------
    # PID core
    # ------------------------------------------------------------------

    def PID_step(self, speed: float, target_speed: float, dt: Optional[float] = None) -> float:
        """Compute the raw PID command.

        Args:
            speed: Current vehicle speed (m/s).
            target_speed: Desired vehicle speed (m/s).
            dt: Time delta in seconds. Defaults to ``default_dt`` (1/50 s).

        Returns:
            Control effort interpreted as longitudinal force.
        """

        dt_value = self.default_dt if dt is None else float(dt)
        dt_value = max(dt_value, 1e-3)  # guard against division by zero

        error = float(target_speed) - float(speed)

        # Proportional
        p_term = self.KP * error

        # Integral with clamping
        self.sum_error += error * dt_value
        self.sum_error = float(
            np.clip(self.sum_error, -self.integral_windup_limit, self.integral_windup_limit)
        )
        i_term = self.KI * self.sum_error

        # Derivative (difference quotient)
        d_term = self.KD * (error - self.last_error) / dt_value

        control = p_term + i_term + d_term
        self.last_error = error
        self.last_control = control
        return float(control)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def control(
        self,
        speed: float,
        target_speed: float,
        dt: Optional[float] = None,
    ) -> tuple[float, float]:
        """Translate the PID output into gas and brake commands.

        Args:
            speed: Current speed from the simulator.
            target_speed: Desired speed produced by the planner.
            dt: Optional integration step (seconds).

        Returns:
            ``(gas, brake)`` tuple with values clipped to simulator limits.
        """

        control = self.PID_step(speed, target_speed, dt=dt)

        if control >= 0.0:
            gas = float(np.clip(control, 0.0, self.max_gas))
            brake = 0.0
        else:
            gas = 0.0
            brake = float(np.clip(-control, 0.0, self.max_brake))

        return gas, brake

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_speed(self, speed: float, target_speed: float, step: int, fig) -> None:
        """Plot speed traces for interactive debugging.

        Matplotlib is imported lazily to keep the runtime dependency optional.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - plotting is optional
            return

        self.speed_history.append(float(speed))
        self.target_speed_history.append(float(target_speed))
        self.step_history.append(int(step))

        plt.gcf().clear()
        plt.plot(self.step_history, self.speed_history, c="green")
        plt.plot(self.step_history, self.target_speed_history)
        fig.canvas.flush_events()
