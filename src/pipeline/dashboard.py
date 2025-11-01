"""Matplotlib dashboard for real-time inspection of pipeline signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence

import numpy as np

from .interfaces import ObserverStep, PipelineObserver


def _ensure_observation(observation: np.ndarray) -> np.ndarray:
    """Return an RGB image suitable for imshow (values clipped to [0, 1])."""

    obs = np.asarray(observation)
    if obs.ndim == 2:
        obs = np.stack([obs] * 3, axis=-1)
    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)
    obs = obs.astype(np.float32, copy=False)
    if obs.max(initial=1.0) > 1.5:
        obs = obs / 255.0
    return np.clip(obs, 0.0, 1.0)


@dataclass
class LiveDashboard(PipelineObserver):
    """Render perception, planning, and control telemetry in real time."""

    max_history: int = 200
    _plt: Any = field(init=False, repr=False)
    _figure: Any = field(init=False, default=None, repr=False)
    _axes: Mapping[str, Any] = field(init=False, default_factory=dict, repr=False)
    _image_artist: Any = field(init=False, default=None, repr=False)
    _lane_artists: Sequence[Any] = field(init=False, default_factory=tuple, repr=False)
    _waypoint_artist: Any = field(init=False, default=None, repr=False)
    _speed_lines: Sequence[Any] = field(init=False, default_factory=tuple, repr=False)
    _control_lines: Sequence[Any] = field(init=False, default_factory=tuple, repr=False)
    _steps: List[float] = field(init=False, default_factory=list, repr=False)
    _speed_actual: List[float] = field(init=False, default_factory=list, repr=False)
    _speed_target: List[float] = field(init=False, default_factory=list, repr=False)
    _steer: List[float] = field(init=False, default_factory=list, repr=False)
    _gas: List[float] = field(init=False, default_factory=list, repr=False)
    _brake: List[float] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        self._plt = self._import_matplotlib()

    def _import_matplotlib(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - import error handled explicitly
            raise RuntimeError("LiveDashboard requires matplotlib to be installed") from exc

        plt.ion()
        return plt

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return

        plt = self._plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_obs, ax_lanes, ax_speed, ax_control = axes.flatten()

        ax_obs.set_title("Observation")
        ax_obs.axis("off")
        self._image_artist = ax_obs.imshow(np.zeros((96, 96, 3), dtype=np.float32))

        ax_lanes.set_title("Lane detection & planning")
        ax_lanes.set_xlabel("x [px]")
        ax_lanes.set_ylabel("y [px]")
        ax_lanes.set_xlim(0, 95)
        ax_lanes.set_ylim(95, 0)
        (left_line,) = ax_lanes.plot([], [], color="#1f77b4", label="Left lane")
        (right_line,) = ax_lanes.plot([], [], color="#ff7f0e", label="Right lane")
        (way_line,) = ax_lanes.plot([], [], marker="o", color="#2ca02c", label="Waypoints")
        ax_lanes.legend(loc="lower right")

        ax_speed.set_title("Speed tracking")
        ax_speed.set_xlabel("Step")
        ax_speed.set_ylabel("Speed [m/s]")
        (actual_line,) = ax_speed.plot([], [], label="Actual")
        (target_line,) = ax_speed.plot([], [], label="Target")
        ax_speed.legend(loc="upper right")

        ax_control.set_title("Control commands")
        ax_control.set_xlabel("Step")
        ax_control.set_ylabel("Normalized command")
        ax_control.set_ylim(-1.05, 1.05)
        (steer_line,) = ax_control.plot([], [], label="Steer")
        (gas_line,) = ax_control.plot([], [], label="Gas")
        (brake_line,) = ax_control.plot([], [], label="Brake")
        ax_control.legend(loc="upper right")

        fig.tight_layout()

        self._figure = fig
        self._axes = {
            "lanes": ax_lanes,
            "speed": ax_speed,
            "control": ax_control,
        }
        self._lane_artists = (left_line, right_line)
        self._waypoint_artist = way_line
        self._speed_lines = (actual_line, target_line)
        self._control_lines = (steer_line, gas_line, brake_line)

    def _reset_history(self) -> None:
        self._steps.clear()
        self._speed_actual.clear()
        self._speed_target.clear()
        self._steer.clear()
        self._gas.clear()
        self._brake.clear()

        if self._speed_lines:
            for line in self._speed_lines:
                line.set_data([], [])
        if self._control_lines:
            for line in self._control_lines:
                line.set_data([], [])

    def _append(self, buffer: List[float], value: float) -> None:
        buffer.append(value)
        if len(buffer) > self.max_history:
            del buffer[0]

    def _update_curves(self, lanes, waypoints: np.ndarray | None) -> None:
        if self._figure is None:
            return

        left_line, right_line = self._lane_artists

        if (
            lanes is not None
            and isinstance(lanes.left_boundary, np.ndarray)
            and lanes.left_boundary.ndim == 2
            and lanes.left_boundary.shape[0] == 2
        ):
            left_line.set_data(lanes.left_boundary[0], lanes.left_boundary[1])
        else:
            left_line.set_data([], [])

        if (
            lanes is not None
            and isinstance(lanes.right_boundary, np.ndarray)
            and lanes.right_boundary.ndim == 2
            and lanes.right_boundary.shape[0] == 2
        ):
            right_line.set_data(lanes.right_boundary[0], lanes.right_boundary[1])
        else:
            right_line.set_data([], [])

        if (
            waypoints is not None
            and isinstance(waypoints, np.ndarray)
            and waypoints.ndim == 2
            and waypoints.shape[0] == 2
        ):
            self._waypoint_artist.set_data(waypoints[0], waypoints[1])
        else:
            self._waypoint_artist.set_data([], [])

        self._axes["lanes"].relim()
        self._axes["lanes"].autoscale_view()
        self._axes["lanes"].set_xlim(0, 95)
        self._axes["lanes"].set_ylim(95, 0)

    def _update_series(self) -> None:
        if self._figure is None:
            return

        actual_line, target_line = self._speed_lines
        steer_line, gas_line, brake_line = self._control_lines

        actual_line.set_data(self._steps, self._speed_actual)
        target_line.set_data(self._steps, self._speed_target)
        steer_line.set_data(self._steps, self._steer)
        gas_line.set_data(self._steps, self._gas)
        brake_line.set_data(self._steps, self._brake)

        self._axes["speed"].relim()
        self._axes["speed"].autoscale_view()
        self._axes["control"].relim()
        self._axes["control"].autoscale_view()
        self._axes["control"].set_ylim(-1.05, 1.05)

    def _refresh(self) -> None:
        if self._figure is None:
            return
        self._figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def on_reset(self, observation: np.ndarray, info: Mapping[str, Any]) -> None:  # type: ignore[override]
        self._ensure_figure()
        self._reset_history()
        if self._image_artist is not None:
            self._image_artist.set_data(_ensure_observation(observation))
        self._update_curves(None, None)
        self._refresh()

    def on_step(self, step: ObserverStep) -> None:  # type: ignore[override]
        self._ensure_figure()

        if self._image_artist is not None:
            self._image_artist.set_data(_ensure_observation(step.observation))

        self._update_curves(step.perception.lanes, step.planning.waypoints)

        self._append(self._steps, float(step.context.step_index))
        actual_speed = float(step.info.get("speed", step.context.speed))
        target_speed = float(step.planning.target_speed) if step.planning.target_speed is not None else np.nan
        self._append(self._speed_actual, actual_speed)
        self._append(self._speed_target, target_speed)
        self._append(self._steer, float(step.command.steer))
        self._append(self._gas, float(step.command.gas))
        self._append(self._brake, float(step.command.brake))

        self._update_series()
        self._refresh()

