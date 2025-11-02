"""Perception modules used by the modular pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from lane_detection import LaneDetection

from .interfaces import LaneDetections, PerceptionModule, PerceptionOutput, PipelineContext


@dataclass
class LaneDetectionModule(PerceptionModule):
    """Wraps :class:`lane_detection.LaneDetection` in the pipeline interface."""

    factory: Callable[[], LaneDetection] = LaneDetection
    _detector: LaneDetection = field(init=False)

    def __post_init__(self) -> None:
        self._detector = self.factory()

    def reset(self) -> None:
        self._detector = self.factory()

    def process(self, observation: np.ndarray, context: PipelineContext) -> PerceptionOutput:
        left, right = self._detector.lane_detection(observation)
        lanes = LaneDetections(left_boundary=left, right_boundary=right)
        return PerceptionOutput(lanes=lanes)
