"""Unit and integration tests for AdaptiveLongitudinalController."""

from __future__ import annotations

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from adaptive_longitudinal_control import AdaptiveLongitudinalController
from configuration import AppConfig
from modular_pipeline import build_pipeline


def test_adaptive_gain_increases_on_sustained_lag_and_damps_on_oscillation() -> None:
    ctrl = AdaptiveLongitudinalController(
        KP=0.05,
        KI=0.0,
        KD=0.0,
        adaptation_rate=1.0,
        slew_rate=0.0,
        deadband=0.5,
    )
    # Sustained positive tracking error without saturation should increase gain_scale
    for _ in range(10):
        ctrl.control(speed=10.0, target_speed=15.0, dt=0.05)
    assert ctrl.gain_scale > 1.0

    # Sign flip (oscillation) should damp gain_scale
    prev_scale = ctrl.gain_scale
    ctrl.control(speed=20.0, target_speed=15.0, dt=0.05)
    assert ctrl.gain_scale < prev_scale

    ctrl.reset()
    assert ctrl.gain_scale == 1.0
    assert ctrl.sum_error == 0.0
    assert ctrl.last_error is None


def test_adaptive_cornering_throttle_reduction_and_slew_limit() -> None:
    ctrl = AdaptiveLongitudinalController(
        KP=0.2,
        KI=0.0,
        KD=0.0,
        max_gas=0.8,
        corner_throttle_reduction=0.5,
        slew_rate=0.0,
    )
    gas_straight, brake_straight = ctrl.control(speed=0.0, target_speed=50.0, dt=0.02, steer=0.0)
    ctrl.reset()
    gas_corner, brake_corner = ctrl.control(speed=0.0, target_speed=50.0, dt=0.02, steer=0.4)

    assert brake_straight == 0.0
    assert brake_corner == 0.0
    assert gas_straight == 0.8
    assert gas_corner == 0.4

    # Slew-rate limiting caps per-step gas jump
    slew_ctrl = AdaptiveLongitudinalController(KP=0.2, slew_rate=5.0, max_gas=0.8)
    gas_step, _ = slew_ctrl.control(speed=0.0, target_speed=50.0, dt=0.02)
    assert abs(gas_step - 0.1) < 1e-6


def test_pipeline_builds_adaptive_longitudinal_controller() -> None:
    cfg = AppConfig.from_mapping(
        {
            "control": {
                "lateral": {"controller_type": "mpc"},
                "longitudinal": {"controller_type": "adaptive"},
            },
            "monitoring": {"dashboard": {"enabled": False}},
        }
    )
    pipeline = build_pipeline(cfg, object(), enable_dashboard=False)
    long_module = pipeline.control[1]
    assert isinstance(long_module._controller, AdaptiveLongitudinalController)
