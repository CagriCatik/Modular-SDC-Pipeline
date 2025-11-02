"""Control modules translating plans into simulator commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from lateral_control import LateralController
from longitudinal_control import LongitudinalController

from .interfaces import ControlCommand, ControlModule, PipelineContext, PlanningState


@dataclass
class LateralControlModule(ControlModule):
    """Stanley-style steering based on waypoint geometry."""

    factory: Callable[[], LateralController] = LateralController
    _controller: LateralController = field(init=False)

    def __post_init__(self) -> None:
        self._controller = self.factory()

    def reset(self) -> None:
        if hasattr(self._controller, "reset"):
            self._controller.reset()
        else:  # pragma: no cover - backwards compatibility
            self._controller = self.factory()

    def act(
        self,
        plan: PlanningState,
        context: PipelineContext,
        command: ControlCommand,
    ) -> ControlCommand:
        if plan.waypoints is None:
            raise ValueError("Waypoints required before lateral control.")
        steer = self._controller.stanley(plan.waypoints, context.speed)
        return command.updated(steer=steer)


@dataclass
class LongitudinalControlModule(ControlModule):
    """PID-based throttle/brake command from target speeds."""

    factory: Callable[[], LongitudinalController] = LongitudinalController
    _controller: LongitudinalController = field(init=False)

    def __post_init__(self) -> None:
        self._controller = self.factory()

    def reset(self) -> None:
        if hasattr(self._controller, "reset"):
            self._controller.reset()
        else:  # pragma: no cover - backwards compatibility
            self._controller = self.factory()

    def act(
        self,
        plan: PlanningState,
        context: PipelineContext,
        command: ControlCommand,
    ) -> ControlCommand:
        if plan.target_speed is None:
            raise ValueError("Target speed required before longitudinal control.")
        gas, brake = self._controller.control(
            context.speed,
            plan.target_speed,
            dt=context.timestep,
        )
        return command.updated(gas=gas, brake=brake)
