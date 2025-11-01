# Modular Self-Driving Car Pipeline

This repository contains a deterministic perception–planning–control stack for
the Gymnasium `CarRacing-v3` environment. Every component is implemented with
classical computer-vision and control techniques so that algorithmic behaviour
remains explainable and reproducible.

## Repository layout

| Path | Purpose |
|------|---------|
| `modular_pipeline.py` | Orchestrator that links the environment, perception, planning, and control modules. |
| `src/lane_detection.py` | Extracts roadside splines from the rendered RGB frame. |
| `src/waypoint_prediction.py` | Generates centreline waypoints and computes a curvature-aware target speed. |
| `src/lateral_control.py` | Stanley lateral controller with damping. |
| `src/longitudinal_control.py` | PID-based longitudinal controller for throttle/brake. |
| `docs/` | MkDocs documentation site with architecture notes and module guides. |
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

Pass `--score` to evaluate the ten fixed leaderboard seeds defined in
`modular_pipeline.py`.

### Documentation site

The MkDocs site (Material theme) mirrors the repository structure and contains
architecture diagrams, module deep dives, and testing checklists.

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Browse the documentation at <http://127.0.0.1:8000/> while iterating on the
codebase.

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
