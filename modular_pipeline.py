"""Command-line entry point wiring configuration and pipeline modules."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import gymnasium as gym

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0] / "src"))

from configuration import AppConfig, DEFAULT_CONFIG_PATH, TargetSpeedConfig
from lane_detection import LaneDetection
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
from mpc_lateral_control import MPCLateralController
from pipeline import (
    LaneDetectionModule,
    LateralControlModule,
    LongitudinalControlModule,
    ModularPipeline,
    PySideDashboard,
    TargetSpeedPlanningModule,
    WaypointPlanningModule,
)
from pipeline.core import calculate_score_for_leaderboard, evaluate
from sdc_wrapper import SDC_Wrapper

__all__ = [
    "ModularPipeline",
    "build_pipeline",
    "evaluate",
    "calculate_score_for_leaderboard",
    "main",
]


def build_pipeline(
    config: AppConfig,
    env: gym.Env,
    *,
    enable_dashboard: Optional[bool] = None,
) -> ModularPipeline:
    """Create a :class:`ModularPipeline` from configuration and an environment."""

    lane_kwargs = config.perception.lane_detection.to_kwargs()
    perception_module = LaneDetectionModule(
        factory=lambda: LaneDetection(**lane_kwargs)
    )

    target_speed_settings = TargetSpeedConfig(
        num_waypoints_used=config.planning.target_speed.num_waypoints_used,
        max_speed=config.planning.target_speed.max_speed,
        min_speed=config.planning.target_speed.min_speed,
        curvature_gain=config.planning.target_speed.curvature_gain,
    )

    planning_modules = [
        WaypointPlanningModule(
            num_waypoints=config.planning.waypoints.num_waypoints,
            way_type=config.planning.waypoints.way_type,
            smoothing_beta=config.planning.waypoints.smoothing_beta,
        ),
        TargetSpeedPlanningModule(settings=target_speed_settings),
    ]

    if config.control.lateral.controller_type == "mpc":
        mpc_kwargs = config.control.lateral.to_mpc_kwargs()
        lateral_module = LateralControlModule(
            factory=lambda: MPCLateralController(**mpc_kwargs)
        )
    else:
        lateral_kwargs = config.control.lateral.to_kwargs()
        lateral_module = LateralControlModule(
            factory=lambda: LateralController(**lateral_kwargs)
        )
    longitudinal_kwargs = config.control.longitudinal.to_kwargs()

    control_modules = [
        lateral_module,
        LongitudinalControlModule(
            factory=lambda: LongitudinalController(**longitudinal_kwargs)
        ),
    ]

    observers = []
    dashboard_flag = (
        config.monitoring.dashboard.enabled if enable_dashboard is None else bool(enable_dashboard)
    )
    if dashboard_flag:
        observers.append(
            PySideDashboard(
                max_history=config.monitoring.dashboard.max_history,
                cut_size=config.perception.lane_detection.cut_size,
                gradient_threshold=config.perception.lane_detection.gradient_threshold,
                controller_name=config.control.lateral.controller_type.upper(),
            )
        )

    return ModularPipeline(
        env=env,
        perception=perception_module,
        planning=planning_modules,
        control=control_modules,
        max_steps=config.runtime.max_steps,
        timestep_seconds=config.runtime.timestep_seconds,
        observers=observers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="evaluate for the leaderboard")
    parser.add_argument("--no_display", action="store_true", default=False, help="headless mode")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML configuration file (defaults to config.yml)",
    )
    args = parser.parse_args()

    config = AppConfig.from_file(args.config)

    dashboard_enabled = config.monitoring.dashboard.enabled and not args.no_display

    env_kwargs = dict(config.environment.kwargs)
    if args.no_display or dashboard_enabled:
        env_kwargs["render_mode"] = "rgb_array"

    base_env = gym.make(config.environment.id, **env_kwargs)
    env = SDC_Wrapper(
        base_env,
        **config.environment.wrapper.to_kwargs(),
    )

    pipeline = build_pipeline(config, env, enable_dashboard=dashboard_enabled)

    try:
        if args.score:
            calculate_score_for_leaderboard(pipeline, config.evaluation.score_seeds)
        else:
            evaluate(pipeline, config.evaluation.episodes)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
