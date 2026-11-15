"""Kinematic-bicycle Model Predictive Control (MPC) for lateral steering."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


class MPCLateralController:
    """Receding-horizon kinematic bicycle MPC over lookahead waypoints."""

    def __init__(
        self,
        vehicle_center_x: float = 48.0,
        steering_limit: float = 0.4,
        horizon: int = 6,
        wheelbase: float = 30.0,
        dt: float = 0.05,
        cte_weight: float = 1.0,
        heading_weight: float = 2.0,
        steer_weight: float = 0.5,
        steer_rate_weight: float = 3.0,
        max_iter: int = 25,
    ) -> None:
        self.vehicle_center_x = float(vehicle_center_x)
        self.steering_limit = float(abs(steering_limit)) if steering_limit != 0 else 0.4
        self.horizon = max(2, int(horizon))
        self.wheelbase = max(1.0, float(wheelbase))
        self.dt = max(1e-3, float(dt))
        self.cte_weight = float(cte_weight)
        self.heading_weight = float(heading_weight)
        self.steer_weight = float(steer_weight)
        self.steer_rate_weight = float(steer_rate_weight)
        self.max_iter = max(5, int(max_iter))

        self.previous_steering_angle: float = 0.0
        self._warm_start: np.ndarray = np.zeros(self.horizon, dtype=float)

    def reset(self) -> None:
        """Reset warm-start sequence and previous steering angle."""
        self.previous_steering_angle = 0.0
        self._warm_start = np.zeros(self.horizon, dtype=float)

    def _valid_waypoints(self, waypoints: np.ndarray) -> bool:
        return (
            isinstance(waypoints, np.ndarray)
            and waypoints.ndim == 2
            and waypoints.shape[0] == 2
            and waypoints.shape[1] >= 1
            and np.isfinite(waypoints).all()
        )

    def _to_vehicle_frame(self, waypoints: np.ndarray) -> np.ndarray:
        x_image = waypoints[0].astype(np.float64, copy=False)
        y_image = waypoints[1].astype(np.float64, copy=False)
        x_vehicle = y_image
        y_vehicle = self.vehicle_center_x - x_image
        return np.vstack([x_vehicle, y_vehicle])

    def _build_reference(self, vehicle_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_wp = vehicle_frame.shape[1]
        if n_wp == self.horizon:
            ref_pts = vehicle_frame
        elif n_wp == 1:
            ref_pts = np.repeat(vehicle_frame, self.horizon, axis=1)
        else:
            t_src = np.linspace(0.0, 1.0, n_wp)
            t_dst = np.linspace(0.0, 1.0, self.horizon)
            ref_x = np.interp(t_dst, t_src, vehicle_frame[0])
            ref_y = np.interp(t_dst, t_src, vehicle_frame[1])
            ref_pts = np.vstack([ref_x, ref_y])

        ref_x = ref_pts[0]
        ref_y = ref_pts[1]
        dx = np.gradient(ref_x)
        dy = np.gradient(ref_y)
        ref_psi = np.arctan2(dy, np.where(np.abs(dx) + np.abs(dy) > 1e-6, dx, 1.0))
        return ref_x, ref_y, ref_psi

    def compute_steering(self, waypoints: np.ndarray, speed: float) -> float:
        """Solve finite-horizon MPC and return CarRacing steering in [-1, 1]."""
        limit = self.steering_limit if self.steering_limit > 0 else 0.4

        if not self._valid_waypoints(waypoints):
            prev = float(self.previous_steering_angle)
            return -float(np.clip(prev, -limit, limit) / limit)

        v = float(speed) if np.isfinite(speed) else 0.0
        v_eff = max(v, 8.0)  # keep kinematic rollout well-conditioned at low speed

        vehicle_frame = self._to_vehicle_frame(waypoints)
        ref_x, ref_y, ref_psi = self._build_reference(vehicle_frame)

        # Arc-length step sizes between consecutive waypoints
        ds = np.hypot(np.diff(ref_x), np.diff(ref_y))
        ds = np.maximum(ds, 1.0)

        prev_delta = float(self.previous_steering_angle)
        wb_eff = self.wheelbase
        w_cte = self.cte_weight
        w_psi = self.heading_weight
        w_u = self.steer_weight
        w_du = self.steer_rate_weight

        def cost(u: np.ndarray) -> float:
            # Start at the vehicle's longitudinal alignment with wp[0] (y=0, psi=0)
            x = float(ref_x[0])
            y = 0.0
            psi = 0.0
            # Normalize pixel errors by 10 px so CTE and heading/steer terms are O(1)
            px_scale = 0.1
            epsi0 = (psi - float(ref_psi[0]) + np.pi) % (2.0 * np.pi) - np.pi
            total = w_cte * ((float(ref_y[0]) * px_scale) ** 2) + w_psi * (epsi0 * epsi0)
            prev_u = prev_delta
            for k in range(self.horizon - 1):
                delta = float(u[k])
                step_len = float(ds[k])
                x += step_len * np.cos(psi)
                y += step_len * np.sin(psi)
                psi += (step_len / wb_eff) * np.tan(delta)

                ex = (x - float(ref_x[k + 1])) * px_scale
                ey = (y - float(ref_y[k + 1])) * px_scale
                epsi = (psi - float(ref_psi[k + 1]) + np.pi) % (2.0 * np.pi) - np.pi

                du = delta - prev_u
                total += w_cte * (ex * ex + ey * ey) + w_psi * (epsi * epsi) + w_u * (delta * delta) + w_du * (du * du)
                prev_u = delta
            return float(total)

        u0 = np.roll(self._warm_start, -1)
        u0[-1] = u0[-2]
        bounds = [(-limit, limit)] * self.horizon

        res = minimize(
            cost,
            u0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-4},
        )
        u_opt = res.x if np.isfinite(res.x).all() else u0
        self._warm_start = np.clip(u_opt, -limit, limit)

        delta_0 = float(self._warm_start[0])
        self.previous_steering_angle = delta_0
        return -float(np.clip(delta_0, -limit, limit) / limit)

    # Alias so LateralControlModule and existing harnesses work interchangeably
    def stanley(self, waypoints: np.ndarray, speed: float) -> float:
        return self.compute_steering(waypoints, speed)
