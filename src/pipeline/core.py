"""Pipeline orchestrator leveraging modular perception, planning, and control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np

from .interfaces import (
    ControlCommand,
    ControlModule,
    PerceptionModule,
    PipelineContext,
    PlanningModule,
    PlanningState,
)


def _action_contract(command: ControlCommand, action_space) -> np.ndarray:
    """Clip a control command to the environment bounds."""

    act = command.as_array()
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return np.clip(act, low, high).astype(action_space.dtype, copy=False)


def _reset_with_speed(env: gym.Env, **kwargs):
    obs, info = env.reset(**kwargs)
    speed = 0.0
    if isinstance(info, dict):
        speed = float(info.get("speed", 0.0))
    return obs, speed


def _iter_modules(*modules: Iterable) -> Iterable:
    for collection in modules:
        if isinstance(collection, Sequence):
            yield from collection
        else:
            yield collection


@dataclass
class ModularPipeline:
    """High-level orchestrator for the perception-planning-control loop."""

    env: gym.Env
    perception: PerceptionModule
    planning: Sequence[PlanningModule]
    control: Sequence[ControlModule]
    max_steps: int = 600
    timestep_seconds: float = 1.0 / 50.0

    def _reset_modules(self) -> None:
        for module in _iter_modules(self.perception, self.planning, self.control):
            module.reset()

    def run_episode(self, seed: Optional[int] = None) -> float:
        try:
            obs, speed = _reset_with_speed(self.env, seed=seed)
        except Exception:  # pragma: no cover - environment init is external
            print("Please note that you can't use the window on the cluster")
            raise

        self._reset_modules()

        total_reward = 0.0
        for step in range(self.max_steps):
            context = PipelineContext(step_index=step, speed=float(speed), timestep=self.timestep_seconds)

            perception_result = self.perception.process(obs, context)
            plan_state = PlanningState()
            for planner in self.planning:
                plan_state = planner.plan(perception_result, context, plan_state)

            command = ControlCommand()
            for controller in self.control:
                command = controller.act(plan_state, context, command)

            action = _action_contract(command, self.env.action_space)
            obs, reward, terminated, truncated, info = self.env.step(action)

            if isinstance(info, dict) and "speed" in info:
                speed = float(info["speed"])

            total_reward += float(reward)
            if terminated or truncated:
                break

        return float(total_reward)

    def close(self) -> None:
        self.env.close()


def evaluate(pipeline: ModularPipeline, episodes: int = 5) -> None:
    for episode in range(episodes):
        reward = pipeline.run_episode()
        print(f"episode {episode}\t reward {reward:.6f}")


def calculate_score_for_leaderboard(pipeline: ModularPipeline, seeds: Sequence[int]) -> None:
    total_reward = 0.0
    for episode, seed in enumerate(seeds):
        reward = pipeline.run_episode(seed=seed)
        print(f"episode {episode}\t reward {reward:.6f}")
        total_reward += float(np.clip(reward, 0.0, np.inf))

    print("---------------------------")
    print(" total score: %f" % (total_reward / len(seeds)))
    print("---------------------------")
