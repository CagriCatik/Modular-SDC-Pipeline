import pathlib
import sys

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mpc_lateral_control import MPCLateralController


def test_mpc_steers_left_for_left_curve_and_right_for_right_curve():
    ctrl = MPCLateralController(horizon=6, steering_limit=0.4)

    # Waypoints curving left (x_image < 48) -> CarRacing steer must be negative
    y_fwd = np.linspace(5.0, 55.0, 6)
    wp_left = np.vstack([48.0 - 0.3 * y_fwd, y_fwd])
    steer_left = ctrl.compute_steering(wp_left, speed=25.0)
    assert -1.0 <= steer_left < 0.0

    ctrl.reset()
    # Waypoints curving right (x_image > 48) -> CarRacing steer must be positive
    wp_right = np.vstack([48.0 + 0.3 * y_fwd, y_fwd])
    steer_right = ctrl.compute_steering(wp_right, speed=25.0)
    assert 0.0 < steer_right <= 1.0


def test_mpc_handles_invalid_waypoints():
    ctrl = MPCLateralController(horizon=6, steering_limit=0.4)
    steer = ctrl.compute_steering(np.array([[np.nan], [10.0]]), speed=15.0)
    assert steer == 0.0
