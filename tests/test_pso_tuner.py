import pathlib
import sys

import numpy as np
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from configuration import AppConfig
from pso_tuner import (
    DEFAULT_CONTROL_PLANNING_SPECS,
    PSOConfig,
    ParameterSpec,
    ParticleSwarmOptimizer,
    apply_params_to_config,
    extract_vector,
    save_config_yaml,
    save_pso_dataset_json,
)


def test_pso_converges_on_quadratic():
    specs = [
        ParameterSpec("x", -5.0, 5.0),
        ParameterSpec("y", -5.0, 5.0),
    ]
    target = np.array([1.75, -2.25])

    def objective(vec: np.ndarray) -> float:
        return -float(np.sum((vec - target) ** 2))

    opt = ParticleSwarmOptimizer(
        specs,
        PSOConfig(num_particles=20, iterations=35, seed=7),
    )
    res = opt.optimize(objective)

    assert res.best_score > -1e-2
    assert res.best_params["x"] == pytest.approx(1.75, abs=0.1)
    assert res.best_params["y"] == pytest.approx(-2.25, abs=0.1)
    assert len(res.evaluations) == 20 * 35


def test_apply_and_save_config_roundtrip(tmp_path: pathlib.Path):
    base = AppConfig()
    vec = extract_vector(base, DEFAULT_CONTROL_PLANNING_SPECS)
    vec[0] = 0.123
    updated = apply_params_to_config(base, DEFAULT_CONTROL_PLANNING_SPECS, vec)

    out_path = tmp_path / "pso_test.yml"
    save_config_yaml(updated, out_path)
    reloaded = AppConfig.from_file(out_path)

    assert reloaded.control.lateral.gain_constant == pytest.approx(0.123)
    assert reloaded.planning.target_speed.min_speed <= reloaded.planning.target_speed.max_speed

    opt = ParticleSwarmOptimizer(
        [ParameterSpec("x", -1.0, 1.0)],
        PSOConfig(num_particles=2, iterations=2, seed=1),
    )
    res = opt.optimize(lambda v: -float(v[0] ** 2))
    dataset_path = tmp_path / "pso_dataset.json"
    save_pso_dataset_json(res, dataset_path)
    assert dataset_path.exists()
