"""Configuration loader for the Modular SDC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml


def _as_dict(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if mapping is None:
        return {}
    return {str(k): v for k, v in dict(mapping).items()}


@dataclass
class LaneDetectionConfig:
    cut_size: int = 68
    spline_smoothness: float = 10.0
    gradient_threshold: float = 14.0
    distance_maxima_gradient: int = 3

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "LaneDetectionConfig":
        payload = _as_dict(data)
        return cls(
            cut_size=int(payload.get("cut_size", cls.cut_size)),
            spline_smoothness=float(payload.get("spline_smoothness", cls.spline_smoothness)),
            gradient_threshold=float(payload.get("gradient_threshold", cls.gradient_threshold)),
            distance_maxima_gradient=int(
                payload.get("distance_maxima_gradient", cls.distance_maxima_gradient)
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "cut_size": self.cut_size,
            "spline_smoothness": self.spline_smoothness,
            "gradient_threshold": self.gradient_threshold,
            "distance_maxima_gradient": self.distance_maxima_gradient,
        }


@dataclass
class LateralControllerConfig:
    controller_type: str = "stanley"
    gain_constant: float = 0.025
    damping_constant: float = 0.30
    vehicle_center_x: float = 48.0
    steering_limit: float = 0.4
    horizon: int = 6
    wheelbase: float = 30.0
    dt: float = 0.05
    cte_weight: float = 1.0
    heading_weight: float = 2.0
    steer_weight: float = 0.5
    steer_rate_weight: float = 3.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "LateralControllerConfig":
        payload = _as_dict(data)
        stanley_cfg = _as_dict(payload.get("stanley"))
        mpc_cfg = _as_dict(payload.get("mpc"))
        raw_type = str(payload.get("type", payload.get("controller_type", cls.controller_type))).strip().lower()
        return cls(
            controller_type=raw_type if raw_type in {"stanley", "mpc"} else cls.controller_type,
            gain_constant=float(stanley_cfg.get("gain_constant", payload.get("gain_constant", cls.gain_constant))),
            damping_constant=float(
                stanley_cfg.get("damping_constant", payload.get("damping_constant", cls.damping_constant))
            ),
            vehicle_center_x=float(payload.get("vehicle_center_x", cls.vehicle_center_x)),
            steering_limit=float(payload.get("steering_limit", cls.steering_limit)),
            horizon=int(mpc_cfg.get("horizon", payload.get("horizon", cls.horizon))),
            wheelbase=float(mpc_cfg.get("wheelbase", payload.get("wheelbase", cls.wheelbase))),
            dt=float(mpc_cfg.get("dt", payload.get("dt", cls.dt))),
            cte_weight=float(mpc_cfg.get("cte_weight", payload.get("cte_weight", cls.cte_weight))),
            heading_weight=float(mpc_cfg.get("heading_weight", payload.get("heading_weight", cls.heading_weight))),
            steer_weight=float(mpc_cfg.get("steer_weight", payload.get("steer_weight", cls.steer_weight))),
            steer_rate_weight=float(
                mpc_cfg.get("steer_rate_weight", payload.get("steer_rate_weight", cls.steer_rate_weight))
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "gain_constant": self.gain_constant,
            "damping_constant": self.damping_constant,
            "vehicle_center_x": self.vehicle_center_x,
            "steering_limit": self.steering_limit,
        }

    def to_mpc_kwargs(self) -> Dict[str, Any]:
        return {
            "vehicle_center_x": self.vehicle_center_x,
            "steering_limit": self.steering_limit,
            "horizon": self.horizon,
            "wheelbase": self.wheelbase,
            "dt": self.dt,
            "cte_weight": self.cte_weight,
            "heading_weight": self.heading_weight,
            "steer_weight": self.steer_weight,
            "steer_rate_weight": self.steer_rate_weight,
        }


@dataclass
class LongitudinalControllerConfig:
    controller_type: str = "pid"
    KP: float = 0.08
    KI: float = 0.01
    KD: float = 0.001
    integral_windup_limit: float = 10.0
    max_gas: float = 0.8
    max_brake: float = 0.8
    default_dt: float = 1.0 / 50.0
    feedforward_gain: float = 0.006
    adaptation_rate: float = 0.15
    min_gain_scale: float = 0.6
    max_gain_scale: float = 1.8
    deadband: float = 0.5
    slew_rate: float = 6.0
    corner_throttle_reduction: float = 0.35
    horizon: int = 8
    dt: float = 0.05
    accel_gain: float = 35.0
    drag_coeff: float = 0.02
    speed_weight: float = 1.0
    overspeed_weight: float = 3.0
    effort_weight: float = 0.15
    effort_rate_weight: float = 0.8
    coast_band: float = 1.5
    brake_transition_scale: float = 0.2
    max_iter: int = 20

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "LongitudinalControllerConfig":
        payload = _as_dict(data)
        raw_type = (
            str(payload.get("type", payload.get("controller_type", cls.controller_type)))
            .strip()
            .lower()
        )
        controller_type = raw_type if raw_type in {"pid", "adaptive", "mpc"} else cls.controller_type
        pid_cfg = _as_dict(payload.get("pid"))
        adaptive_cfg = _as_dict(payload.get("adaptive"))
        mpc_cfg = _as_dict(payload.get("mpc"))
        active_cfg = _as_dict(payload.get(controller_type))

        def _get_shared_gain(key: str, default: float) -> float:
            if key in active_cfg:
                return float(active_cfg[key])
            if key in pid_cfg:
                return float(pid_cfg[key])
            return float(payload.get(key, default))

        return cls(
            controller_type=controller_type,
            KP=_get_shared_gain("KP", cls.KP),
            KI=_get_shared_gain("KI", cls.KI),
            KD=_get_shared_gain("KD", cls.KD),
            integral_windup_limit=_get_shared_gain("integral_windup_limit", cls.integral_windup_limit),
            max_gas=float(active_cfg.get("max_gas", payload.get("max_gas", cls.max_gas))),
            max_brake=float(active_cfg.get("max_brake", payload.get("max_brake", cls.max_brake))),
            default_dt=float(active_cfg.get("default_dt", payload.get("default_dt", cls.default_dt))),
            feedforward_gain=float(
                adaptive_cfg.get("feedforward_gain", payload.get("feedforward_gain", cls.feedforward_gain))
            ),
            adaptation_rate=float(
                adaptive_cfg.get("adaptation_rate", payload.get("adaptation_rate", cls.adaptation_rate))
            ),
            min_gain_scale=float(
                adaptive_cfg.get("min_gain_scale", payload.get("min_gain_scale", cls.min_gain_scale))
            ),
            max_gain_scale=float(
                adaptive_cfg.get("max_gain_scale", payload.get("max_gain_scale", cls.max_gain_scale))
            ),
            deadband=float(adaptive_cfg.get("deadband", payload.get("deadband", cls.deadband))),
            slew_rate=float(adaptive_cfg.get("slew_rate", payload.get("slew_rate", cls.slew_rate))),
            corner_throttle_reduction=float(
                active_cfg.get(
                    "corner_throttle_reduction",
                    adaptive_cfg.get(
                        "corner_throttle_reduction",
                        mpc_cfg.get(
                            "corner_throttle_reduction",
                            payload.get("corner_throttle_reduction", cls.corner_throttle_reduction),
                        ),
                    ),
                )
            ),
            horizon=int(mpc_cfg.get("horizon", payload.get("horizon", cls.horizon))),
            dt=float(mpc_cfg.get("dt", payload.get("dt", cls.dt))),
            accel_gain=float(mpc_cfg.get("accel_gain", payload.get("accel_gain", cls.accel_gain))),
            drag_coeff=float(mpc_cfg.get("drag_coeff", payload.get("drag_coeff", cls.drag_coeff))),
            speed_weight=float(mpc_cfg.get("speed_weight", payload.get("speed_weight", cls.speed_weight))),
            overspeed_weight=float(
                mpc_cfg.get("overspeed_weight", payload.get("overspeed_weight", cls.overspeed_weight))
            ),
            effort_weight=float(mpc_cfg.get("effort_weight", payload.get("effort_weight", cls.effort_weight))),
            effort_rate_weight=float(
                mpc_cfg.get("effort_rate_weight", payload.get("effort_rate_weight", cls.effort_rate_weight))
            ),
            coast_band=float(mpc_cfg.get("coast_band", payload.get("coast_band", cls.coast_band))),
            brake_transition_scale=float(
                mpc_cfg.get(
                    "brake_transition_scale",
                    payload.get("brake_transition_scale", cls.brake_transition_scale),
                )
            ),
            max_iter=int(mpc_cfg.get("max_iter", payload.get("max_iter", cls.max_iter))),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "KP": self.KP,
            "KI": self.KI,
            "KD": self.KD,
            "integral_windup_limit": self.integral_windup_limit,
            "max_gas": self.max_gas,
            "max_brake": self.max_brake,
            "default_dt": self.default_dt,
        }

    def to_adaptive_kwargs(self) -> Dict[str, Any]:
        kwargs = self.to_kwargs()
        kwargs.update(
            {
                "feedforward_gain": self.feedforward_gain,
                "adaptation_rate": self.adaptation_rate,
                "min_gain_scale": self.min_gain_scale,
                "max_gain_scale": self.max_gain_scale,
                "deadband": self.deadband,
                "slew_rate": self.slew_rate,
                "corner_throttle_reduction": self.corner_throttle_reduction,
            }
        )
        return kwargs

    def to_mpc_kwargs(self) -> Dict[str, Any]:
        return {
            "max_gas": self.max_gas,
            "max_brake": self.max_brake,
            "horizon": self.horizon,
            "dt": self.dt,
            "accel_gain": self.accel_gain,
            "drag_coeff": self.drag_coeff,
            "speed_weight": self.speed_weight,
            "overspeed_weight": self.overspeed_weight,
            "effort_weight": self.effort_weight,
            "effort_rate_weight": self.effort_rate_weight,
            "coast_band": self.coast_band,
            "brake_transition_scale": self.brake_transition_scale,
            "corner_throttle_reduction": self.corner_throttle_reduction,
            "max_iter": self.max_iter,
        }


@dataclass
class WaypointPlannerConfig:
    num_waypoints: int = 6
    way_type: str = "smooth"
    smoothing_beta: float = 30.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "WaypointPlannerConfig":
        payload = _as_dict(data)
        way_type = str(payload.get("way_type", cls.way_type)).strip()
        return cls(
            num_waypoints=int(payload.get("num_waypoints", cls.num_waypoints)),
            way_type=way_type if way_type else cls.way_type,
            smoothing_beta=float(payload.get("smoothing_beta", cls.smoothing_beta)),
        )


@dataclass
class TargetSpeedConfig:
    num_waypoints_used: int = 4
    max_speed: float = 30.0
    min_speed: float = 15.0
    curvature_gain: float = 2.5

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "TargetSpeedConfig":
        payload = _as_dict(data)
        return cls(
            num_waypoints_used=int(payload.get("num_waypoints_used", cls.num_waypoints_used)),
            max_speed=float(payload.get("max_speed", cls.max_speed)),
            min_speed=float(payload.get("min_speed", cls.min_speed)),
            curvature_gain=float(payload.get("curvature_gain", cls.curvature_gain)),
        )


@dataclass
class PlanningConfig:
    waypoints: WaypointPlannerConfig = field(default_factory=WaypointPlannerConfig)
    target_speed: TargetSpeedConfig = field(default_factory=TargetSpeedConfig)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "PlanningConfig":
        payload = _as_dict(data)
        return cls(
            waypoints=WaypointPlannerConfig.from_mapping(payload.get("waypoints")),
            target_speed=TargetSpeedConfig.from_mapping(payload.get("target_speed")),
        )


@dataclass
class PerceptionConfig:
    lane_detection: LaneDetectionConfig = field(default_factory=LaneDetectionConfig)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "PerceptionConfig":
        payload = _as_dict(data)
        return cls(
            lane_detection=LaneDetectionConfig.from_mapping(payload.get("lane_detection")),
        )


@dataclass
class ControlConfig:
    lateral: LateralControllerConfig = field(default_factory=LateralControllerConfig)
    longitudinal: LongitudinalControllerConfig = field(default_factory=LongitudinalControllerConfig)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "ControlConfig":
        payload = _as_dict(data)
        return cls(
            lateral=LateralControllerConfig.from_mapping(payload.get("lateral")),
            longitudinal=LongitudinalControllerConfig.from_mapping(payload.get("longitudinal")),
        )


@dataclass
class DashboardConfig:
    enabled: bool = False
    max_history: int = 200
    backend: str = "pyside"

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "DashboardConfig":
        payload = _as_dict(data)
        backend = str(payload.get("backend", cls.backend)).strip().lower()
        return cls(
            enabled=bool(payload.get("enabled", cls.enabled)),
            max_history=int(payload.get("max_history", cls.max_history)),
            backend=backend if backend else cls.backend,
        )


@dataclass
class MonitoringConfig:
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "MonitoringConfig":
        payload = _as_dict(data)
        return cls(
            dashboard=DashboardConfig.from_mapping(payload.get("dashboard")),
        )


@dataclass
class EnvironmentWrapperConfig:
    remove_score: bool = True
    return_linear_velocity: bool = True

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "EnvironmentWrapperConfig":
        payload = _as_dict(data)
        return cls(
            remove_score=bool(payload.get("remove_score", cls.remove_score)),
            return_linear_velocity=bool(
                payload.get("return_linear_velocity", cls.return_linear_velocity)
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "remove_score": self.remove_score,
            "return_linear_velocity": self.return_linear_velocity,
        }


@dataclass
class EnvironmentConfig:
    id: str = "CarRacing-v3"
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"render_mode": "human"})
    wrapper: EnvironmentWrapperConfig = field(default_factory=EnvironmentWrapperConfig)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "EnvironmentConfig":
        payload = _as_dict(data)
        kwargs = _as_dict(payload.get("kwargs")) or {"render_mode": "human"}
        kwargs.setdefault("render_mode", "human")
        return cls(
            id=str(payload.get("id", cls.id)),
            kwargs=kwargs,
            wrapper=EnvironmentWrapperConfig.from_mapping(payload.get("wrapper")),
        )


@dataclass
class RuntimeConfig:
    max_steps: int = 600
    timestep_seconds: float = 1.0 / 50.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "RuntimeConfig":
        payload = _as_dict(data)
        return cls(
            max_steps=int(payload.get("max_steps", cls.max_steps)),
            timestep_seconds=float(payload.get("timestep_seconds", cls.timestep_seconds)),
        )


def _default_leaderboard_seeds() -> list[int]:
    return [
        97657630,
        47460391,
        22619914,
        76925063,
        84647422,
        83470445,
        77482096,
        94017676,
        99341122,
        58134947,
    ]


@dataclass
class EvaluationConfig:
    episodes: int = 5
    score_seeds: Sequence[int] = field(default_factory=_default_leaderboard_seeds)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "EvaluationConfig":
        payload = _as_dict(data)
        seeds = payload.get("score_seeds")
        if isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes)):
            score_seeds = [int(s) for s in seeds]
        else:
            score_seeds = _default_leaderboard_seeds()
        return cls(
            episodes=int(payload.get("episodes", cls.episodes)),
            score_seeds=score_seeds,
        )


@dataclass
class AppConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "AppConfig":
        payload = _as_dict(data)
        return cls(
            environment=EnvironmentConfig.from_mapping(payload.get("environment")),
            runtime=RuntimeConfig.from_mapping(payload.get("runtime")),
            perception=PerceptionConfig.from_mapping(payload.get("perception")),
            planning=PlanningConfig.from_mapping(payload.get("planning")),
            control=ControlConfig.from_mapping(payload.get("control")),
            monitoring=MonitoringConfig.from_mapping(payload.get("monitoring")),
            evaluation=EvaluationConfig.from_mapping(payload.get("evaluation")),
        )

    @classmethod
    def from_file(cls, path: Optional[Path]) -> "AppConfig":
        if path is None:
            return cls()
        path = path.expanduser()
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
        return cls.from_mapping(data)


DEFAULT_CONFIG_PATH = Path("config.yml")

