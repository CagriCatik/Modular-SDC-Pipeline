"""Utilities to share dashboard wiring across integration scripts."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from configuration import AppConfig, DEFAULT_CONFIG_PATH
from pipeline import (
    ControlCommand,
    DashboardInputs,
    LiveDashboard,
    build_observer_step,
    make_lane_detections,
)
from sdc_wrapper import SDC_Wrapper


def load_config(path: str | None = None) -> AppConfig:
    """Load the application configuration (defaults to ``config.yml``)."""

    config_path = DEFAULT_CONFIG_PATH if path is None else path
    return AppConfig.from_file(config_path)


def create_env(config: AppConfig, *, render_mode: Optional[str] = None) -> SDC_Wrapper:
    """Instantiate the wrapped simulator as described by ``config``."""

    env_kwargs = dict(config.environment.kwargs)
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    base_env = gym.make(config.environment.id, **env_kwargs)
    return SDC_Wrapper(base_env, **config.environment.wrapper.to_kwargs())


def create_dashboard(config: AppConfig, *, enabled: bool = True) -> Optional[LiveDashboard]:
    """Instantiate :class:`LiveDashboard` if ``enabled`` is ``True``."""

    if not enabled or not config.monitoring.dashboard.enabled:
        return None

    try:
        return LiveDashboard(max_history=config.monitoring.dashboard.max_history)
    except RuntimeError as exc:
        print(f"Live dashboard unavailable: {exc}")
        return None


def reset_dashboard(
    dashboard: Optional[LiveDashboard],
    observation: np.ndarray,
    info: Mapping[str, Any],
) -> None:
    if dashboard is not None:
        dashboard.on_reset(observation, info)


def update_dashboard(
    dashboard: Optional[LiveDashboard],
    *,
    observation: np.ndarray,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    step_index: int,
    speed: float,
    timestep: float,
    lanes: Optional[tuple[np.ndarray, np.ndarray] | np.ndarray] = None,
    waypoints: Optional[np.ndarray] = None,
    target_speed: Optional[float] = None,
) -> None:
    """Emit a new step to the dashboard observer if present."""

    if dashboard is None:
        return

    lane_payload = None
    if isinstance(lanes, tuple) and len(lanes) == 2:
        lane_payload = make_lane_detections(lanes[0], lanes[1])
    elif isinstance(lanes, np.ndarray):
        raise ValueError("lanes must be (left, right) tuple when provided as ndarray")

    command = ControlCommand(
        steer=float(action[0]) if action.size >= 1 else 0.0,
        gas=float(action[1]) if action.size >= 2 else 0.0,
        brake=float(action[2]) if action.size >= 3 else 0.0,
    )

    step = build_observer_step(
        DashboardInputs(
            observation=observation,
            step_index=step_index,
            speed=speed,
            timestep=timestep,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action=action,
            lanes=lane_payload,
            waypoints=waypoints,
            target_speed=target_speed,
            command=command,
        )
    )

    dashboard.on_step(step)


def extract_speed(info: Mapping[str, Any], fallback: float = 0.0) -> float:
    """Extract simulator speed in m/s when available."""

    if isinstance(info, Mapping) and "speed" in info:
        try:
            return float(info["speed"])
        except Exception:
            pass
    return float(fallback)
