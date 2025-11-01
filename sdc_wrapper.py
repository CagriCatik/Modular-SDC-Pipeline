import numpy as np
import gymnasium as gym
from gymnasium.core import ActType

class SDC_Wrapper(gym.Wrapper):
    def __init__(self, env, remove_score=True, return_linear_velocity=False):
        super().__init__(env)
        self.remove_score = remove_score
        self.return_linear_velocity = return_linear_velocity

    # ----- internal helpers -----

    def _mask_hud(self, obs):
        # CarRacing observations are (96, 96, 3). The bottom strip contains HUD.
        # Guard against shape drift.
        if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[0] >= 85:
            # Zero a conservative HUD region. Adjust if your pipeline expects a different mask.
            obs[84:, :, :] = 0
        return obs

    def _linear_speed(self):
        # Prefer unwrapped env to access the underlying Box2D car object
        base = getattr(self.env, "unwrapped", self.env)
        car = getattr(base, "car", None)
        if car is None or not hasattr(car, "hull") or not hasattr(car.hull, "linearVelocity"):
            return 0.0
        v = car.hull.linearVelocity
        # Box2D b2Vec2 supports item access and attributes x,y
        try:
            vx, vy = float(v[0]), float(v[1])
        except Exception:
            vx, vy = float(getattr(v, "x", 0.0)), float(getattr(v, "y", 0.0))
        return float(np.hypot(vx, vy))

    # ----- Gymnasium API -----

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if not isinstance(info, dict):
            info = {}

        if self.remove_score:
            obs = self._mask_hud(obs)

        if self.return_linear_velocity:
            info["speed"] = self._linear_speed()

        return obs, info

    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = super().step(action)
        if not isinstance(info, dict):
            info = {}

        # Conservative reward clipping to prevent explosions in controllers/logs
        reward_clipped = float(np.clip(reward, -0.1, 1e8))

        if self.remove_score:
            obs = self._mask_hud(obs)

        if self.return_linear_velocity:
            info["speed"] = self._linear_speed()

        return obs, reward_clipped, terminated, truncated, info
