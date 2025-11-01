# Modular Self-Driving Car Pipeline

This repository contains a deterministic perception–planning–control stack for
the Gymnasium `CarRacing-v3` environment. Every component is implemented with
classical computer-vision and control techniques so that algorithmic behaviour
remains explainable and reproducible.

## Repository layout

| Path | Purpose |
|------|---------|
| `modular_pipeline.py` | Command-line entry point that loads configuration and wires together modular stages. |
| `src/pipeline/` | Interfaces and reusable modules for perception, planning, and control composition. |
| `src/lane_detection.py` | Extracts roadside splines from the rendered RGB frame. |
| `src/waypoint_prediction.py` | Generates centreline waypoints and computes a curvature-aware target speed. |
| `src/lateral_control.py` | Stanley lateral controller with damping. |
| `src/longitudinal_control.py` | PID-based longitudinal controller for throttle/brake. |
| `docs/technical_reference.md` | Unified scientific documentation with equations, testing strategy, and tuning advice. |
| `.github/workflows/deploy.yml` | GitHub Actions workflow that builds and publishes the documentation site. |

## Getting started

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the stack without the on-screen viewer:

```bash
python modular_pipeline.py --no_display
```

Pass `--score` to evaluate the leaderboard seeds listed in `config.yml`.

### Configuration-driven tuning

All perception, planning, control, and runtime parameters are declared in
`config.yml`. Update the YAML file to tune the controllers, change the waypoint
generation mode, or adjust the simulator wrapper without touching the Python
code. Use `--config` to point to an alternative configuration file:

```bash
python modular_pipeline.py --config configs/aggressive.yaml
```

Key sections include:

| Section | Purpose |
|---------|---------|
| `environment` | Gymnasium environment ID, render mode, and wrapper toggles. |
| `runtime` | Episode horizon and integration timestep. |
| `perception.lane_detection` | Image-cropping and gradient thresholds for spline extraction. |
| `planning` | Waypoint smoothing strategy and curvature-based speed model. |
| `control` | Stanley gains and longitudinal PID terms, including saturation limits. |
| `evaluation` | Default episode count and leaderboard seed list. |

### Documentation site

The MkDocs site (Material theme) is anchored by a single scientific reference
document covering architecture, derivations, and verification procedures.

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Browse the documentation at <http://127.0.0.1:8000/> while iterating on the
codebase. The same content is published automatically to GitHub Pages.

## Development guidelines

1. **Respect module contracts** – each module returns NumPy data structures with
   documented shapes. Review the docs before changing interfaces.
2. **Reset controller state per episode** – both lateral and longitudinal
   controllers expose a `reset()` method used by the orchestrator.
3. **Validate against reference seeds** – `calculate_score_for_leaderboard`
   ensures regressions are detected quickly.

## Continuous delivery of docs

The `deploy.yml` workflow builds the MkDocs site on pushes to `main` or manual
triggers. It installs only the documentation dependencies and publishes the
static site to the `gh-pages` branch using
[`peaceiris/actions-gh-pages`](https://github.com/peaceiris/actions-gh-pages).

## Licensing

The code is provided for educational and research purposes. Cite this repository
when using it in academic work.
