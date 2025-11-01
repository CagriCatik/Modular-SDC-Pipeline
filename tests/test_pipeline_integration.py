import argparse
import sys
import pathlib

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pipeline import (
    ControlCommand,
    LaneDetections,
    ModularPipeline,
    ObserverStep,
    PerceptionOutput,
    PipelineContext,
    PipelineObserver,
    PlanningState,
)


class DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(96, 96, 3),
            dtype=np.float32,
        )
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {"speed": 10.0}

    def step(self, action):
        self._steps += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 1.0 - float(abs(action[0]))
        terminated = False
        truncated = self._steps >= 5
        info = {"speed": 10.0 + 0.2 * self._steps}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


class RecordingPerception:
    def __init__(self):
        self.reset_calls = 0
        self.process_calls = 0

    def reset(self):
        self.reset_calls += 1

    def process(self, observation, context: PipelineContext) -> PerceptionOutput:
        self.process_calls += 1
        lanes = LaneDetections(
            left_boundary=np.zeros((2, 4), dtype=float),
            right_boundary=np.ones((2, 4), dtype=float),
        )
        return PerceptionOutput(lanes=lanes)


class RecordingPlanner:
    def __init__(self):
        self.reset_calls = 0
        self.plan_calls = 0

    def reset(self):
        self.reset_calls += 1

    def plan(self, perception: PerceptionOutput, context: PipelineContext, previous: PlanningState) -> PlanningState:
        self.plan_calls += 1
        return previous.updated(waypoints=np.zeros((2, 4), dtype=float))


class RecordingSpeedPlanner:
    def __init__(self):
        self.reset_calls = 0
        self.plan_calls = 0

    def reset(self):
        self.reset_calls += 1

    def plan(self, perception: PerceptionOutput, context: PipelineContext, previous: PlanningState) -> PlanningState:
        self.plan_calls += 1
        return previous.updated(target_speed=12.0)


class RecordingLateralControl:
    def __init__(self):
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self):
        self.reset_calls += 1

    def act(self, plan: PlanningState, context: PipelineContext, command: ControlCommand) -> ControlCommand:
        self.act_calls += 1
        return command.updated(steer=0.1)


class RecordingLongitudinalControl:
    def __init__(self):
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self):
        self.reset_calls += 1

    def act(self, plan: PlanningState, context: PipelineContext, command: ControlCommand) -> ControlCommand:
        self.act_calls += 1
        return command.updated(gas=0.5, brake=0.0)


class RecordingObserver(PipelineObserver):
    def __init__(self):
        self.reset_calls = 0
        self.step_records = []

    def on_reset(self, observation, info):
        self.reset_calls += 1
        self.reset_shape = observation.shape
        self.reset_info = dict(info)

    def on_step(self, step: ObserverStep):
        self.step_records.append((step.context.step_index, float(step.reward)))


def test_modular_pipeline_runs_with_stub_modules():
    env = DummyEnv()
    perception = RecordingPerception()
    planning = [RecordingPlanner(), RecordingSpeedPlanner()]
    control = [RecordingLateralControl(), RecordingLongitudinalControl()]
    observer = RecordingObserver()

    pipeline = ModularPipeline(
        env=env,
        perception=perception,
        planning=planning,
        control=control,
        max_steps=10,
        timestep_seconds=0.1,
        observers=[observer],
    )

    reward = pipeline.run_episode(seed=123)

    assert perception.reset_calls == 1
    assert all(planner.reset_calls == 1 for planner in planning)
    assert all(controller.reset_calls == 1 for controller in control)

    assert perception.process_calls == 5
    assert all(planner.plan_calls == 5 for planner in planning)
    assert all(controller.act_calls == 5 for controller in control)

    assert observer.reset_calls == 1
    assert observer.reset_shape == env.observation_space.shape
    assert len(observer.step_records) == 5
    assert observer.step_records[0][0] == 0

    assert reward == pytest.approx(4.5)

    pipeline.close()


if __name__ == "__main__":
    try:  # pragma: no cover
        from .dashboard_utils import (
            create_dashboard,
            create_env,
            load_config,
        )
    except ImportError:  # pragma: no cover
        from dashboard_utils import (  # type: ignore
            create_dashboard,
            create_env,
            load_config,
        )
    from modular_pipeline import build_pipeline

    parser = argparse.ArgumentParser(description="Run the modular pipeline with the live dashboard")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render_mode", default="human")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()

    config = load_config()
    env = create_env(config, render_mode=args.render_mode)
    dashboard = create_dashboard(config, enabled=not args.no_dashboard)

    pipeline = build_pipeline(config, env, enable_dashboard=False)
    observers = list(pipeline.observers)
    if dashboard is not None:
        observers.append(dashboard)
    pipeline.observers = tuple(observers)

    try:
        for episode in range(args.episodes):
            reward = pipeline.run_episode()
            print(f"episode {episode}\treward {reward:+0.4f}")
    finally:
        pipeline.close()
