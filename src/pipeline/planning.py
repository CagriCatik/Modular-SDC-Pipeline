"""Planning modules that convert perception into references."""

from __future__ import annotations

from dataclasses import dataclass, field

from configuration import TargetSpeedConfig
from waypoint_prediction import target_speed_prediction, waypoint_prediction

from .interfaces import PerceptionOutput, PipelineContext, PlanningModule, PlanningState


@dataclass
class WaypointPlanningModule(PlanningModule):
    """Generate centreline waypoints from lane detections."""

    num_waypoints: int = 6
    way_type: str = "smooth"
    smoothing_beta: float = 30.0

    def reset(self) -> None:  # pragma: no cover - stateless module
        pass

    def plan(
        self,
        perception: PerceptionOutput,
        context: PipelineContext,
        previous: PlanningState,
    ) -> PlanningState:
        if perception.lanes is None:
            raise ValueError("Lane detections required before waypoint planning.")
        waypoints = waypoint_prediction(
            perception.lanes.left_boundary,
            perception.lanes.right_boundary,
            num_waypoints=self.num_waypoints,
            way_type=self.way_type,
            smoothing_beta=self.smoothing_beta,
        )
        return previous.updated(waypoints=waypoints)


@dataclass
class TargetSpeedPlanningModule(PlanningModule):
    """Compute reference speeds from waypoints."""

    settings: TargetSpeedConfig = field(default_factory=TargetSpeedConfig)

    def reset(self) -> None:  # pragma: no cover - stateless module
        pass

    def plan(
        self,
        perception: PerceptionOutput,
        context: PipelineContext,
        previous: PlanningState,
    ) -> PlanningState:
        if previous.waypoints is None:
            raise ValueError("Waypoints required before target speed planning.")
        target_speed = target_speed_prediction(
            previous.waypoints,
            num_waypoints_used=self.settings.num_waypoints_used,
            max_speed=self.settings.max_speed,
            min_speed=self.settings.min_speed,
            K_v=self.settings.curvature_gain,
        )
        return previous.updated(target_speed=float(target_speed))
