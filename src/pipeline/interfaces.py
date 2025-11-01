"""Interfaces shared by the modular pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class PipelineContext:
    """State passed to every pipeline stage."""

    step_index: int
    speed: float
    timestep: float


@dataclass
class LaneDetections:
    """Spline control points describing the left/right lane boundaries."""

    left_boundary: np.ndarray
    right_boundary: np.ndarray


@dataclass
class PerceptionOutput:
    """Container for perception artefacts used by downstream stages."""

    lanes: Optional[LaneDetections] = None
    extras: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = {}

    def with_lanes(self, lanes: LaneDetections) -> "PerceptionOutput":
        return replace(self, lanes=lanes)

    def merged(self, **extras: Any) -> "PerceptionOutput":
        payload = dict(self.extras)
        payload.update(extras)
        return replace(self, extras=payload)


@dataclass
class PlanningState:
    """Planning artefacts shared between planning modules."""

    waypoints: Optional[np.ndarray] = None
    target_speed: Optional[float] = None
    extras: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = {}

    def updated(self, *, waypoints: Optional[np.ndarray] = None, target_speed: Optional[float] = None, **extras: Any) -> "PlanningState":
        payload = dict(self.extras)
        payload.update(extras)
        return PlanningState(
            waypoints=self.waypoints if waypoints is None else waypoints,
            target_speed=self.target_speed if target_speed is None else target_speed,
            extras=payload,
        )


@dataclass
class ControlCommand:
    """Final command that will be sent to the simulator."""

    steer: float = 0.0
    gas: float = 0.0
    brake: float = 0.0

    def updated(self, *, steer: Optional[float] = None, gas: Optional[float] = None, brake: Optional[float] = None) -> "ControlCommand":
        return ControlCommand(
            steer=self.steer if steer is None else float(steer),
            gas=self.gas if gas is None else float(gas),
            brake=self.brake if brake is None else float(brake),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.steer, self.gas, self.brake], dtype=np.float32)


@runtime_checkable
class Resettable(Protocol):
    """Simple protocol capturing modules with reset hooks."""

    def reset(self) -> None:
        ...


class PerceptionModule(Protocol):
    """Stages that transform observations into world understanding."""

    def reset(self) -> None:
        ...

    def process(self, observation: np.ndarray, context: PipelineContext) -> PerceptionOutput:
        ...


class PlanningModule(Protocol):
    """Modules that turn perception into reference trajectories or speeds."""

    def reset(self) -> None:
        ...

    def plan(
        self,
        perception: PerceptionOutput,
        context: PipelineContext,
        previous: PlanningState,
    ) -> PlanningState:
        ...


class ControlModule(Protocol):
    """Controllers responsible for turning plans into actuator commands."""

    def reset(self) -> None:
        ...

    def act(
        self,
        plan: PlanningState,
        context: PipelineContext,
        command: ControlCommand,
    ) -> ControlCommand:
        ...


@dataclass
class ObserverStep:
    """Snapshot passed to observers after each environment transition."""

    observation: np.ndarray
    perception: PerceptionOutput
    planning: PlanningState
    command: ControlCommand
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    context: PipelineContext


@runtime_checkable
class PipelineObserver(Protocol):
    """Optional observers that tap into the modular pipeline lifecycle."""

    def on_reset(self, observation: np.ndarray, info: Mapping[str, Any]) -> None:
        ...

    def on_step(self, step: ObserverStep) -> None:
        ...
