import numpy as np

class LateralController:
    """Stanley-based lateral controller with damping and robust guards."""

    def __init__(
        self,
        gain_constant: float = 0.025,
        damping_constant: float = 0.0125,
        vehicle_center_x: float = 48.0,
        steering_limit: float = 0.4,
    ) -> None:
        """Create a lateral controller.

        Args:
            gain_constant: Proportional gain on cross-track error (k in Stanley).
            damping_constant: First-order damping on steering command to reduce jitter.
            vehicle_center_x: Image column that corresponds to the vehicle centre.
            steering_limit: Physical steering bound in radians used for scaling to [-1, 1].
        """

        self.gain_constant = float(gain_constant)
        self.damping_constant = float(damping_constant)
        self.vehicle_center_x = float(vehicle_center_x)
        self.steering_limit = float(abs(steering_limit)) if steering_limit != 0 else 0.4
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

    def _to_vehicle_frame(self, waypoints: np.ndarray) -> np.ndarray:
        """Rotate image-frame waypoints into the vehicle frame.

        The lane detector returns pixel coordinates where axis 0 is the horizontal image
        index and axis 1 points forward (down the cropped image). The Stanley controller
        assumes a vehicle-centric frame with the vehicle heading along +x and lateral
        offsets on the +y axis. We convert by swapping the axes and centring the lateral
        coordinate around the vehicle column.
        """

        x_image = waypoints[0].astype(np.float32, copy=False)
        y_image = waypoints[1].astype(np.float32, copy=False)

        x_vehicle = y_image  # forward distance increases down the image crop
        y_vehicle = self.vehicle_center_x - x_image  # positive value means track is left

        return np.vstack([x_vehicle, y_vehicle])

    def stanley(self, waypoints: np.ndarray, speed: float) -> float:
        """
        One step of the Stanley controller with damping.

        Args:
            waypoints: array shape [2, N] in image coordinates (x columns, y rows).
            speed: current vehicle speed (same units as controller tuning).

        Returns:
            Steering command in [-1, 1] assuming +/-0.4 rad physical limit.
        """
        limit = self.steering_limit if self.steering_limit > 0 else 0.4

        # Guard: invalid or empty waypoints -> hold previous command
        if not self._valid_waypoints(waypoints):
            prev = float(self.previous_steering_angle)
            return float(np.clip(prev, -limit, limit) / limit)

        # Numerical guards
        epsilon = 1e-6
        v = float(speed) if np.isfinite(speed) else 0.0
        v = max(v, 0.0)  # no negative speed

        # Vehicle at origin, heading along +x
        vehicle_heading = 0.0

        vehicle_frame = self._to_vehicle_frame(waypoints)

        # Heading error psi_t from first segment if available
        if vehicle_frame.shape[1] >= 2:
            dx = float(vehicle_frame[0, 1] - vehicle_frame[0, 0])
            dy = float(vehicle_frame[1, 1] - vehicle_frame[1, 0])
            path_heading = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 0.0 else 0.0
        else:
            path_heading = 0.0

        psi_t = path_heading - vehicle_heading
        # Wrap to [-pi, pi]
        psi_t = (psi_t + np.pi) % (2.0 * np.pi) - np.pi

        # Cross-track error d_t from first waypoint
        lateral_offset = float(vehicle_frame[1, 0])

        d_t = float(lateral_offset)

        # Stanley control law
        delta_sc = psi_t + np.arctan2(self.gain_constant * d_t, v + epsilon)

        # Damping (discrete first-order filter on delta)
        delta = delta_sc - self.damping_constant * (delta_sc - self.previous_steering_angle)

        # Update memory
        self.previous_steering_angle = float(delta)

        # Clip to physical limit, then rescale to [-1, 1]
        return float(np.clip(delta, -limit, limit) / limit)
