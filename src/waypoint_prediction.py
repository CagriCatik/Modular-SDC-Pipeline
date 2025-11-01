import numpy as np
from scipy.interpolate import splev
from scipy.optimize import minimize
from typing import Optional, Tuple

# -------------------------
# Utilities
# -------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=0)
    norm = np.where(norm == 0.0, 1.0, norm)
    return v / norm

def curvature(waypoints: np.ndarray) -> float:
    """
    Curvature proxy for path smoothing and speed prediction.
    waypoints: array shape [2, N]
    """
    if waypoints.ndim != 2 or waypoints.shape[0] != 2 or waypoints.shape[1] < 3:
        return 0.0
    delta = waypoints[:, 1:] - waypoints[:, :-1]           # [2, N-1]
    delta_n = normalize(delta)                              # [2, N-1]
    if delta_n.shape[1] < 2:
        return 0.0
    dots = np.sum(delta_n[:, :-1] * delta_n[:, 1:], axis=0)
    return float(np.sum(dots))

def _is_valid_tck(tck: Optional[Tuple]) -> bool:
    if tck is None or not isinstance(tck, (tuple, list)) or len(tck) != 3:
        return False
    t, c, k = tck
    try:
        _ = int(k)
    except Exception:
        return False
    # Basic size sanity checks
    try:
        tc = np.asarray(c)
        if tc.ndim < 1 or tc.size == 0:
            return False
    except Exception:
        return False
    return True

def _fallback_waypoints(num_waypoints: int = 6) -> np.ndarray:
    """
    Straight-ahead centerline fallback in cropped image coordinates used by the splines.
    x near mid-column 48, y forward from near the bottom of the crop.
    Returns shape [2, N].
    """
    x = np.full(num_waypoints, 48.0, dtype=np.float32)
    y = np.linspace(5.0, 60.0, num_waypoints, dtype=np.float32)
    return np.vstack([x, y])

# -------------------------
# Smoothing objective
# -------------------------

def smoothing_objective(waypoints_flat: np.ndarray, waypoints_center_flat: np.ndarray, beta: float = 30.0) -> float:
    """
    Objective for path smoothing.
    waypoints_flat: [2*N]
    waypoints_center_flat: [2*N]
    """
    if waypoints_center_flat.ndim != 1 or waypoints_flat.ndim != 1:
        return 0.0
    if waypoints_center_flat.shape[0] != waypoints_flat.shape[0]:
        return 0.0

    num_waypoints = waypoints_center_flat.shape[0] // 2
    waypoints = waypoints_flat.reshape(2, num_waypoints)
    waypoints_center = waypoints_center_flat.reshape(2, num_waypoints)

    ls_tocenter = float(np.sum((waypoints - waypoints_center) ** 2))
    curv = curvature(waypoints)

    # Minimize distance to centerline and encourage smoothness via curvature proxy
    return ls_tocenter - beta * curv

# -------------------------
# Public API
# -------------------------

def waypoint_prediction(
    roadside1_spline: Optional[Tuple],
    roadside2_spline: Optional[Tuple],
    num_waypoints: int = 6,
    way_type: str = "smooth",
) -> np.ndarray:
    """
    Predict waypoints via two methods:
    - way_type="center": midpoints between the two roadside splines
    - way_type="smooth": L-BFGS-B smoothing starting from the midpoints

    Returns: waypoints array with shape [2, num_waypoints].

    Robustness:
    - If either spline is invalid or splev fails, returns a straight-ahead fallback.
    """
    # Guard for invalid splines
    if not (_is_valid_tck(roadside1_spline) and _is_valid_tck(roadside2_spline)):
        return _fallback_waypoints(num_waypoints)

    u = np.linspace(0.0, 1.0, num_waypoints)

    try:
        r1 = np.array(splev(u, roadside1_spline))  # [2, N]
        r2 = np.array(splev(u, roadside2_spline))  # [2, N]
    except Exception:
        return _fallback_waypoints(num_waypoints)

    if r1.shape != (2, num_waypoints) or r2.shape != (2, num_waypoints):
        return _fallback_waypoints(num_waypoints)
    if not (np.isfinite(r1).all() and np.isfinite(r2).all()):
        return _fallback_waypoints(num_waypoints)

    waypoints_center = (r1 + r2) / 2.0  # [2, N]

    if way_type == "center":
        return waypoints_center.astype(np.float32, copy=False)

    if way_type == "smooth":
        # Flatten for optimizer
        w0 = waypoints_center.flatten()
        try:
            res = minimize(
                smoothing_objective,
                w0,
                args=(w0,),                 # FIX: tuple arg
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-6},
            )
            w_opt = res.x if res.success and np.isfinite(res.x).all() else w0
        except Exception:
            w_opt = w0

        waypoints_smoothed = w_opt.reshape(2, num_waypoints)
        return waypoints_smoothed.astype(np.float32, copy=False)

    # Unknown way_type -> default to center
    return waypoints_center.astype(np.float32, copy=False)

def target_speed_prediction(
    waypoints: np.ndarray,
    num_waypoints_used: int = 4,
    max_speed: float = 30.0,
    min_speed: float = 15.0,
    K_v: float = 2.5,
) -> float:
    """
    Predict target speed from path curvature proxy.
    Returns a non-negative scalar speed.
    """
    if waypoints is None or waypoints.ndim != 2 or waypoints.shape[0] != 2 or waypoints.shape[1] < 2:
        return float(min_speed)

    n = min(num_waypoints_used, waypoints.shape[1])
    w = waypoints[:, :n]

    delta = w[:, 1:] - w[:, :-1]
    delta_n = normalize(delta)
    if delta_n.shape[1] < 2:
        curvature_term = 0.0
    else:
        dots = np.sum(delta_n[:, :-1] * delta_n[:, 1:], axis=0)
        curvature_term = float(np.sum(1.0 - dots))

    exponent = -K_v * curvature_term
    target_speed = (max_speed - min_speed) * np.exp(exponent) + min_speed
    return float(max(target_speed, 0.0))
