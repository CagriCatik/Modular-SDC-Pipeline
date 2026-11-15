"""Receding-horizon Model Predictive Control (MPC) for longitudinal speed."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


class MPCLongitudinalController:
    """Receding-horizon longitudinal MPC over throttle and brake effort.

    Optimizes a sequence of net longitudinal commands ``u_k in [-max_brake, effective_max_gas]``
    using a first-order longitudinal vehicle model with linear drag, asymmetric
    overspeed braking urgency, a coast-before-brake deadband, crisp gas-to-brake
    transitions, steering-coupled cornering throttle limits, and optional
    waypoint-lookahead speed profiling.
    """

    def __init__(
        self,
        max_gas: float = 0.8,
        max_brake: float = 0.8,
        horizon: int = 8,
        dt: float = 0.05,
        accel_gain: float = 35.0,
        drag_coeff: float = 0.02,
        speed_weight: float = 1.0,
        overspeed_weight: float = 3.0,
        effort_weight: float = 0.15,
        effort_rate_weight: float = 0.8,
        coast_band: float = 1.5,
        brake_transition_scale: float = 0.2,
        corner_throttle_reduction: float = 0.35,
        max_iter: int = 20,
    ) -> None:
        self.max_gas = max(0.0, float(max_gas))
        self.max_brake = max(0.0, float(max_brake))
        self.horizon = max(2, int(horizon))
        self.dt = max(1e-3, float(dt))
        self.accel_gain = max(1e-3, float(accel_gain))
        self.drag_coeff = max(0.0, float(drag_coeff))
        self.speed_weight = float(speed_weight)
        self.overspeed_weight = float(overspeed_weight)
        self.effort_weight = float(effort_weight)
        self.effort_rate_weight = float(effort_rate_weight)
        self.coast_band = max(0.0, float(coast_band))
        self.brake_transition_scale = float(np.clip(brake_transition_scale, 0.0, 1.0))
        self.corner_throttle_reduction = float(corner_throttle_reduction)
        self.max_iter = max(5, int(max_iter))

        self.previous_effort: float = 0.0
        self._warm_start: np.ndarray = np.zeros(self.horizon, dtype=float)

    def reset(self) -> None:
        """Reset warm-start sequence and previous effort."""

        self.previous_effort = 0.0
        self._warm_start = np.zeros(self.horizon, dtype=float)

    def _build_target_profile(
        self,
        target_speed: float,
        waypoints: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build a horizon target speed profile using lookahead waypoint curvature."""

        profile = np.full(self.horizon, max(0.0, float(target_speed)), dtype=float)
        if (
            waypoints is None
            or not isinstance(waypoints, np.ndarray)
            or waypoints.ndim != 2
            or waypoints.shape[0] != 2
            or waypoints.shape[1] < 3
            or not np.isfinite(waypoints).all()
        ):
            return profile

        # Measure cumulative heading change across lookahead waypoints
        diffs = np.diff(waypoints.astype(float, copy=False), axis=1)
        seg_norms = np.hypot(diffs[0], diffs[1])
        valid = seg_norms > 1e-3
        if np.sum(valid) < 2:
            return profile

        headings = np.arctan2(diffs[1, valid], diffs[0, valid])
        turn_angles = np.abs(np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings))))
        max_turn = float(np.max(turn_angles)) if turn_angles.size > 0 else 0.0

        # If a sharp bend is visible ahead, taper the far end of the horizon down
        if max_turn > 0.12:
            bend_factor = max(0.45, 1.0 - min(max_turn / 1.2, 0.55))
            far_target = target_speed * bend_factor
            profile = np.linspace(target_speed, far_target, self.horizon)

        return profile

    def _cost(
        self,
        u_seq: np.ndarray,
        speed: float,
        target_profile: np.ndarray,
        dt_value: float,
    ) -> float:
        v = float(speed)
        cost = 0.0
        prev_u = self.previous_effort

        for k in range(self.horizon):
            u = float(u_seq[k])
            v = max(0.0, v + dt_value * (self.accel_gain * u - self.drag_coeff * v))
            v_ref = float(target_profile[k])
            err = v - v_ref

            if err > self.coast_band:
                # Overspeed beyond coast band: strong braking urgency
                excess = err - self.coast_band
                speed_cost = self.overspeed_weight * (excess * excess)
            elif err >= 0.0:
                # Within coast band: let drag slow us gently instead of stabbing brakes
                speed_cost = 0.0
            else:
                speed_cost = self.speed_weight * (err * err)

            du = u - prev_u
            rate_weight = self.effort_rate_weight
            if err > self.coast_band and u < 0.0 and prev_u > 0.0:
                # Allow crisp gas-to-brake transition when overspeeding
                rate_weight *= self.brake_transition_scale

            cost += speed_cost + self.effort_weight * (u * u) + rate_weight * (du * du)
            prev_u = u

        return float(cost)

    def control(
        self,
        speed: float,
        target_speed: float,
        dt: Optional[float] = None,
        steer: float = 0.0,
        waypoints: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Solve the receding-horizon longitudinal MPC problem.

        Args:
            speed: Current vehicle speed (m/s).
            target_speed: Desired speed from planner (m/s).
            dt: Optional integration step override (seconds).
            steer: Current lateral steering command used for cornering traction limit.
            waypoints: Optional ``(2, N)`` lookahead waypoints for curve braking preview.

        Returns:
            ``(gas, brake)`` tuple clipped to simulator limits.
        """

        dt_value = self.dt if dt is None else max(1e-3, float(dt))
        speed_value = max(0.0, float(speed))
        target_value = max(0.0, float(target_speed))
        target_profile = self._build_target_profile(target_value, waypoints)

        steer_ratio = min(abs(float(steer)) / 0.4, 1.0)
        effective_max_gas = max(
            0.0, self.max_gas * (1.0 - self.corner_throttle_reduction * steer_ratio)
        )

        bounds = [(-self.max_brake, effective_max_gas)] * self.horizon
        x0 = np.clip(self._warm_start, -self.max_brake, effective_max_gas)

        result = minimize(
            self._cost,
            x0,
            args=(speed_value, target_profile, dt_value),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-4},
        )

        optimal_seq = result.x if result.success or result.x is not None else x0
        effort = float(np.clip(optimal_seq[0], -self.max_brake, effective_max_gas))

        self.previous_effort = effort
        self._warm_start = np.concatenate([optimal_seq[1:], [optimal_seq[-1]]])

        if effort >= 0.0:
            return effort, 0.0
        return 0.0, -effort
