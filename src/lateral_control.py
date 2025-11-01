import numpy as np

class LateralController:
    """
    Stanley-based lateral controller with damping and robust guards.

    Args:
        gain_constant: Proportional gain on cross-track error (k in Stanley).
        damping_constant: First-order damping on steering command to reduce jitter.

    Notes:
        - Expects waypoints in vehicle frame: vehicle at origin, heading along +x.
        - Returns steering action scaled to [-1, 1] assuming env steering limit = 0.4 rad.
    """

    def __init__(self, gain_constant: float = 0.025, damping_constant: float = 0.0125):
        self.gain_constant = float(gain_constant)
        self.damping_constant = float(damping_constant)
        self.previous_steering_angle = 0.0  # radians

    def reset(self) -> None:
        """Reset controller internal state."""
        self.previous_steering_angle = 0.0

    def _valid_waypoints(self, waypoints: np.ndarray) -> bool:
        return (
            isinstance(waypoints, np.ndarray)
            and waypoints.ndim == 2
            and waypoints.shape[0] == 2
            and waypoints.shape[1] >= 1
            and np.isfinite(waypoints).all()
        )

    def stanley(self, waypoints: np.ndarray, speed: float) -> float:
        """
        One step of the Stanley controller with damping.

        Args:
            waypoints: array shape [2, N], in vehicle coordinates.
            speed: current vehicle speed (same units as controller tuning).

        Returns:
            Steering command in [-1, 1] assuming +/-0.4 rad physical limit.
        """
        # Guard: invalid or empty waypoints -> hold previous command
        if not self._valid_waypoints(waypoints):
            prev = float(self.previous_steering_angle)
            return float(np.clip(prev, -0.4, 0.4) / 0.4)

        # Numerical guards
        epsilon = 1e-6
        v = float(speed) if np.isfinite(speed) else 0.0
        v = max(v, 0.0)  # no negative speed

        # Vehicle at origin, heading along +x
        vehicle_heading = 0.0

        # Heading error psi_t from first segment if available
        if waypoints.shape[1] >= 2:
            dx = float(waypoints[0, 1] - waypoints[0, 0])
            dy = float(waypoints[1, 1] - waypoints[1, 0])
            path_heading = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 0.0 else 0.0
        else:
            path_heading = 0.0

        psi_t = path_heading - vehicle_heading
        # Wrap to [-pi, pi]
        psi_t = (psi_t + np.pi) % (2.0 * np.pi) - np.pi

        # Cross-track error d_t from first waypoint
        dx_error = float(waypoints[0, 0])
        dy_error = float(waypoints[1, 0])
        d_abs = float(np.hypot(dx_error, dy_error))

        # Sign by lateral offset (y>0 -> positive error)
        sign = 0.0
        if dy_error > 0.0:
            sign = 1.0
        elif dy_error < 0.0:
            sign = -1.0
        d_t = d_abs * sign

        # Stanley control law
        delta_sc = psi_t + np.arctan2(self.gain_constant * d_t, v + epsilon)

        # Damping (discrete first-order filter on delta)
        delta = delta_sc - self.damping_constant * (delta_sc - self.previous_steering_angle)

        # Update memory
        self.previous_steering_angle = float(delta)

        # Clip to physical limit 0.4 rad, then rescale to [-1, 1]
        return float(np.clip(delta, -0.4, 0.4) / 0.4)
