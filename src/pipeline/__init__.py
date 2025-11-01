"""Modular pipeline abstractions for the self-driving car stack."""

from .interfaces import (
    ControlCommand,
    LaneDetections,
    PerceptionOutput,
    PipelineContext,
    PlanningState,
    PipelineObserver,
    ObserverStep,
)
from .control import LateralControlModule, LongitudinalControlModule
from .core import ModularPipeline
from .dashboard import LiveDashboard
from .telemetry import DashboardInputs, build_observer_step, make_lane_detections
from .perception import LaneDetectionModule
from .planning import TargetSpeedPlanningModule, WaypointPlanningModule

__all__ = [
    "ControlCommand",
    "LaneDetections",
    "PerceptionOutput",
    "PipelineContext",
    "PlanningState",
    "PipelineObserver",
    "ObserverStep",
    "LateralControlModule",
    "LongitudinalControlModule",
    "ModularPipeline",
    "LiveDashboard",
    "LaneDetectionModule",
    "TargetSpeedPlanningModule",
    "WaypointPlanningModule",
    "DashboardInputs",
    "build_observer_step",
    "make_lane_detections",
]
