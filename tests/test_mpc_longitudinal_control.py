"""Unit and integration tests for MPCLongitudinalController and nested config."""

from __future__ import annotations

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
from configuration import AppConfig
from modular_pipeline import build_pipeline
from mpc_longitudinal_control import MPCLongitudinalController


def test_mpc_longitudinal_accelerates_brakes_and_respects_cornering_limit() -> None:
    ctrl = MPCLongitudinalController(
        max_gas=0.8,
        max_brake=0.5,
        corner_throttle_reduction=0.5,
    )
    # Accelerate when speed < target_speed
    gas_straight, brake_straight = ctrl.control(speed=10.0, target_speed=40.0, steer=0.0)
    assert 0.0 < gas_straight <= 0.8
    assert brake_straight == 0.0

    # Cornering steering reduces max gas
    ctrl.reset()
    gas_corner, brake_corner = ctrl.control(speed=10.0, target_speed=40.0, steer=0.4)
    assert 0.0 <= gas_corner <= 0.4 + 1e-6
    assert brake_corner == 0.0

    # Brake when speed > target_speed
    ctrl.reset()
    gas_dec, brake_dec = ctrl.control(speed=50.0, target_speed=15.0, steer=0.0)
    assert gas_dec == 0.0
    assert 0.0 < brake_dec <= 0.5


def test_nested_controller_config_and_pipeline_selection() -> None:
    cfg = AppConfig.from_mapping(
        {
            "control": {
                "lateral": {
                    "controller_type": "mpc",
                    "stanley": {"gain_constant": 0.05},
                    "mpc": {"wheelbase": 14.0, "horizon": 7},
                },
                "longitudinal": {
                    "controller_type": "mpc",
                    "max_gas": 0.7,
                    "max_brake": 0.4,
                    "pid": {"KP": 0.25},
                    "mpc": {"horizon": 6, "accel_gain": 30.0},
                },
            },
            "monitoring": {"dashboard": {"enabled": False}},
        }
    )
    assert cfg.control.lateral.gain_constant == 0.05
    assert cfg.control.lateral.wheelbase == 14.0
    assert cfg.control.lateral.horizon == 7
    assert cfg.control.longitudinal.KP == 0.25
    assert cfg.control.longitudinal.horizon == 6
    assert cfg.control.longitudinal.accel_gain == 30.0

    pipeline = build_pipeline(cfg, object(), enable_dashboard=False)
    assert isinstance(pipeline.control[1]._controller, MPCLongitudinalController)


def test_mpc_longitudinal_coast_band_and_waypoint_curve_preview() -> None:
    ctrl = MPCLongitudinalController(max_gas=0.8, max_brake=0.5, coast_band=2.0)
    # Slight overspeed inside coast_band should not stab the brakes
    gas, brake = ctrl.control(speed=31.0, target_speed=30.0)
    assert brake == 0.0
    assert gas <= 0.1

    # Sharp upcoming turn in waypoints should trigger earlier deceleration even at target_speed
    ctrl.reset()
    straight_wp = np.array([[48.0, 48.0, 48.0, 48.0], [60.0, 50.0, 40.0, 30.0]])
    sharp_turn_wp = np.array([[48.0, 48.0, 56.0, 68.0], [60.0, 50.0, 45.0, 44.0]])

    gas_straight, brake_straight = ctrl.control(
        speed=50.0, target_speed=50.0, waypoints=straight_wp
    )
    ctrl.reset()
    gas_curve, brake_curve = ctrl.control(
        speed=50.0, target_speed=50.0, waypoints=sharp_turn_wp
    )
    assert (brake_curve > brake_straight) or (gas_curve < gas_straight)
