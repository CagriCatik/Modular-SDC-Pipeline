"""Helper utilities for wiring dashboard observers outside the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .interfaces import (
    ControlCommand,
    LaneDetections,
    ObserverStep,
    PerceptionOutput,
    PipelineContext,
    PlanningState,
)


def make_lane_detections(
    left_boundary: Sequence[Sequence[float]] | np.ndarray,
    right_boundary: Sequence[Sequence[float]] | np.ndarray,
) -> LaneDetections:
    """Create :class:`LaneDetections` from array-like inputs."""

    left = np.asarray(left_boundary, dtype=np.float32)
    right = np.asarray(right_boundary, dtype=np.float32)
    return LaneDetections(left_boundary=left, right_boundary=right)


@dataclass
class DashboardInputs:
    """Minimal data required to synthesise an :class:`ObserverStep`."""

    observation: np.ndarray
    step_index: int
    speed: float
    timestep: float
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    action: Sequence[float]
    lanes: Optional[LaneDetections] = None
    waypoints: Optional[np.ndarray] = None
    target_speed: Optional[float] = None
    command: Optional[ControlCommand] = None


def build_observer_step(payload: DashboardInputs) -> ObserverStep:
    """Materialise an :class:`ObserverStep` for dashboard observers."""

    observation = np.asarray(payload.observation)
    action = np.asarray(payload.action, dtype=np.float32)
    command = payload.command or ControlCommand(
        steer=float(action[0]) if action.size >= 1 else 0.0,
        gas=float(action[1]) if action.size >= 2 else 0.0,
        brake=float(action[2]) if action.size >= 3 else 0.0,
    )

    perception = PerceptionOutput(lanes=payload.lanes)
    planning = PlanningState(
        waypoints=None if payload.waypoints is None else np.asarray(payload.waypoints),
        target_speed=payload.target_speed,
    )

    context = PipelineContext(
        step_index=int(payload.step_index),
        speed=float(payload.speed),
        timestep=float(payload.timestep),
    )

    info_mapping: Mapping[str, Any]
    if isinstance(payload.info, Mapping):
        info_mapping = payload.info
    else:
        info_mapping = {}

    return ObserverStep(
        observation=observation,
        perception=perception,
        planning=planning,
        command=command,
        action=action,
        reward=float(payload.reward),
        terminated=bool(payload.terminated),
        truncated=bool(payload.truncated),
        info=info_mapping,
        context=context,
    )
