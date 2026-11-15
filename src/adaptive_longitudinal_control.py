"""Adaptive longitudinal speed controller for the Modular SDC pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np


class AdaptiveLongitudinalController:
    """Gain-scheduled, online-adaptive throttle and brake controller.

    Combines speed feedforward, online MIT-style gain adaptation (increasing
    responsiveness during sustained lag and damping on error sign flips),
    conditional integration anti-windup, steering-coupled cornering throttle
    limiting, deadband hysteresis, and actuator slew-rate limiting.
    """

    def __init__(
        self,
        KP: float = 0.08,
        KI: float = 0.01,
        KD: float = 0.001,
        integral_windup_limit: float = 10.0,
        max_gas: float = 0.8,
        max_brake: float = 0.8,
        default_dt: float = 1.0 / 50.0,
        feedforward_gain: float = 0.006,
        adaptation_rate: float = 0.15,
        min_gain_scale: float = 0.6,
        max_gain_scale: float = 1.8,
        deadband: float = 0.5,
        slew_rate: float = 6.0,
        corner_throttle_reduction: float = 0.35,
    ) -> None:
        self.KP = float(KP)
        self.KI = float(KI)
        self.KD = float(KD)
        self.integral_windup_limit = float(integral_windup_limit)
        self.max_gas = float(max_gas)
        self.max_brake = float(max_brake)
        self.default_dt = float(default_dt)
        self.feedforward_gain = float(feedforward_gain)
        self.adaptation_rate = float(adaptation_rate)
        self.min_gain_scale = float(min_gain_scale)
        self.max_gain_scale = float(max_gain_scale)
        self.deadband = float(deadband)
        self.slew_rate = float(slew_rate)
        self.corner_throttle_reduction = float(corner_throttle_reduction)

        self.gain_scale: float = 1.0
        self.sum_error: float = 0.0
        self.last_error: Optional[float] = None
        self.last_speed: Optional[float] = None
        self.last_gas: float = 0.0
        self.last_brake: float = 0.0
        self.last_control: float = 0.0

    def reset(self) -> None:
        """Clear controller state before a new episode."""

        self.gain_scale = 1.0
        self.sum_error = 0.0
        self.last_error = None
        self.last_speed = None
        self.last_gas = 0.0
        self.last_brake = 0.0
        self.last_control = 0.0

    def _adapt_gain(self, error: float, dt_value: float, saturated: bool) -> None:
        """Update online gain scale based on error persistence or oscillation."""

        if self.last_error is None or self.adaptation_rate <= 0.0:
            return

        if error * self.last_error < 0.0 and abs(error) > self.deadband:
            # Error flipped sign -> damp gain to avoid oscillation
            self.gain_scale -= 0.5 * self.adaptation_rate * dt_value
        elif (
            error * self.last_error > 0.0
            and abs(error) > self.deadband
            and not saturated
        ):
            # Sustained tracking lag without saturation -> boost gain
            lag_factor = min(abs(error) / 20.0, 1.0)
            self.gain_scale += self.adaptation_rate * lag_factor * dt_value
        else:
            # Gentle relaxation back toward nominal gain
            self.gain_scale += 0.05 * self.adaptation_rate * (1.0 - self.gain_scale) * dt_value

        self.gain_scale = float(
            np.clip(self.gain_scale, self.min_gain_scale, self.max_gain_scale)
        )

    def _apply_slew_limit(self, target: float, previous: float, dt_value: float) -> float:
        """Limit actuator rate of change to keep chassis transitions smooth."""

        if self.slew_rate <= 0.0:
            return float(target)
        max_delta = self.slew_rate * dt_value
        return float(np.clip(target, previous - max_delta, previous + max_delta))

    def control(
        self,
        speed: float,
        target_speed: float,
        dt: Optional[float] = None,
        steer: float = 0.0,
        waypoints: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Compute adaptive ``(gas, brake)`` commands.

        Args:
            speed: Current vehicle speed (m/s).
            target_speed: Desired speed from planner (m/s).
            dt: Optional integration step (seconds).
            steer: Current lateral steering command used for cornering traction guard.
            waypoints: Unused lookahead waypoints kept for interface compatibility.

        Returns:
            ``(gas, brake)`` tuple clipped to simulator limits.
        """
        _ = waypoints

        dt_value = self.default_dt if dt is None else float(dt)
        dt_value = max(dt_value, 1e-3)

        speed_value = float(speed)
        target_value = float(target_speed)
        error = target_value - speed_value

        steer_ratio = min(abs(float(steer)) / 0.4, 1.0)
        effective_max_gas = self.max_gas * (1.0 - self.corner_throttle_reduction * steer_ratio)
        effective_max_gas = max(0.0, float(effective_max_gas))

        # Feedforward baseline proportional to target speed
        ff_term = self.feedforward_gain * max(target_value, 0.0)

        # Boost braking responsiveness on sharp deceleration demands
        brake_boost = 1.0 + min(abs(error) / 25.0, 0.5) if error < -self.deadband else 1.0

        kp_eff = self.KP * self.gain_scale * brake_boost
        ki_eff = self.KI * self.gain_scale
        kd_eff = self.KD * self.gain_scale

        effective_error = 0.0 if abs(error) <= self.deadband else error
        p_term = kp_eff * effective_error

        if self.last_speed is None:
            d_term = 0.0
        else:
            d_term = -kd_eff * (speed_value - self.last_speed) / dt_value

        pre_control = ff_term + p_term + ki_eff * self.sum_error + d_term
        saturated_high = pre_control >= effective_max_gas and effective_error > 0.0
        saturated_low = pre_control <= -self.max_brake and effective_error < 0.0

        self._adapt_gain(error, dt_value, saturated_high or saturated_low)

        # Conditional integration (anti-windup)
        if not (saturated_high or saturated_low):
            self.sum_error += effective_error * dt_value
            self.sum_error = float(
                np.clip(
                    self.sum_error,
                    -self.integral_windup_limit,
                    self.integral_windup_limit,
                )
            )

        control_effort = ff_term + p_term + ki_eff * self.sum_error + d_term

        if control_effort >= 0.0:
            raw_gas = float(np.clip(control_effort, 0.0, effective_max_gas))
            raw_brake = 0.0
        else:
            raw_gas = 0.0
            raw_brake = float(np.clip(-control_effort, 0.0, self.max_brake))

        gas = float(np.clip(self._apply_slew_limit(raw_gas, self.last_gas, dt_value), 0.0, effective_max_gas))
        brake = float(np.clip(self._apply_slew_limit(raw_brake, self.last_brake, dt_value), 0.0, self.max_brake))

        self.last_error = error
        self.last_speed = speed_value
        self.last_gas = gas
        self.last_brake = brake
        self.last_control = float(control_effort)
        return gas, brake
