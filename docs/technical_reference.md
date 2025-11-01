# Scientific Reference Manual

The modular self-driving pipeline consists of analytically defined perception,
planning, and control stages operating on the Gymnasium `CarRacing-v3`
simulator. Each stage obeys an explicit interface so that alternative
algorithms can be swapped in without touching the remaining stack.

## Architectural overview

```mermaid
flowchart LR
    I[Observation $x_t$] -->|Perception| P[LaneDetectionModule\\n$\mathcal{L}$]
    P -->|Lanes| W[WaypointPlanningModule\\n$\mathcal{P}_w$]
    W -->|Waypoints| S[TargetSpeedPlanningModule\\n$\mathcal{P}_v$]
    S -->|Reference speed| C1[LateralControlModule\\n$\mathcal{C}_\delta$]
    W --> C1
    S --> C2[LongitudinalControlModule\\n$\mathcal{C}_a$]
    C1 --> A[Action vector $u_t$]
    C2 --> A
    A --> E[Gymnasium environment]
    E -->|reward, speed| M[ModularPipeline]
    M -->|next observation| I
```

At runtime the `ModularPipeline` orchestrator (see `modular_pipeline.py`)
drives these modules according to the configuration loaded from `config.yml`.

### Timing model

Episodes run at a configurable discrete timestep $\Delta t$ (default
$1/50\,\text{s}$). For step index $k$, the pipeline evaluates modules with the
context

$$
\mathcal{C}_k = \{k,\ v_k,\ \Delta t\},
$$

where $v_k$ is the simulator-reported longitudinal speed.

## Perception: lane boundary extraction

The perception stage crops the RGB observation to the region in front of the
vehicle, converts it to intensity, and derives horizontal/vertical gradients
$\partial_x I$ and $\partial_y I$. The gradient magnitude used for peak
searching is

$$
G = |\partial_x I| + |\partial_y I|.
$$

Row-wise maxima separated by a minimum distance form candidate lane points.
The first valid pair above the vehicle seed two sets of points that are tracked
upward row by row. The final polylines are smoothed with a cubic B-spline to
produce left/right boundaries $L$ and $R$.

## Planning: geometric and speed references

### Waypoint generation

The waypoint planner fits a centreline spline by minimizing a curvature
regularised functional

$$
J(C) = \sum_{i=1}^{N} \|C(s_i) - m_i\|_2^2 + \beta \int \kappa(s)^2\, ds,
$$

where $m_i$ are midpoints between $L$ and $R$, $\kappa(s)$ is the spline
curvature, and $\beta$ is the smoothing weight (`planning.waypoints.smoothing_beta`).
The planner outputs $N$ waypoints expressed in the vehicle frame.

### Target-speed prediction

Curvature-driven speed selection uses the reciprocal curvature $\kappa$ of the
centreline:

$$
\hat{v} = \operatorname{clip}\left( v_{\min}, v_{\max}, v_{\max} - K_v\, |\kappa| \right),
$$

with gains configured via `planning.target_speed`. The planner optionally
averages multiple forward-looking waypoints to damp oscillations.

## Control: vehicle dynamics simplifications

### Lateral steering (Stanley method)

The Stanley controller combines heading error $\psi_k$ and cross-track error
$d_k$:

$$
\delta_k = \psi_k + \arctan\left(\frac{k_s d_k}{v_k + \varepsilon}\right) -
\lambda (\delta_k - \delta_{k-1}),
$$

where $k_s$ is the gain (`control.lateral.gain_constant`) and $\lambda$ is the
first-order damping term (`control.lateral.damping_constant`). The command is
clipped to the simulator's $\pm 0.4$ rad steering limit and rescaled to
$[-1,1]$ before submission.

### Longitudinal PID control

Throttle/brake are derived from a discrete PID law applied to the speed error
$e_k = v^*_k - v_k$:

$$
\begin{aligned}
P_k &= K_P e_k,\\
I_k &= I_{k-1} + K_I e_k \Delta t,\\
D_k &= K_D \frac{e_k - e_{k-1}}{\Delta t},\\
u_k &= P_k + I_k + D_k.
\end{aligned}
$$

The integral term is clamped to `control.longitudinal.integral_windup_limit`.
Positive $\nu_k$ commands throttle while negative $\nu_k$ maps to brake
pressure, both bounded by `control.longitudinal.max_gas` and
`control.longitudinal.max_brake`.

## Configuration surface

All tunable parameters live in `config.yml` and mirror the dataclasses in
`src/configuration.py`:

- `perception.lane_detection.*` controls cropping, spline smoothness, and
  gradient thresholds.
- `planning.waypoints` sets waypoint count, interpolation model (`way_type`),
  and smoothing weight.
- `planning.target_speed` configures curvature gain and admissible speed range.
- `control.lateral` and `control.longitudinal` expose the Stanley gains and PID
  coefficients.
- `runtime.*` chooses episode horizon and timestep.
- `environment.*` specifies the Gymnasium environment ID, render mode, and
  wrapper behaviour.

Switch configurations at runtime via

```bash
python modular_pipeline.py --config custom.yml
```

## Verification and testing

Automated tests cover module contracts as well as the orchestrator wiring:

- Unit tests in `tests/` validate perception, planning, and control behaviours.
- `tests/test_pipeline_integration.py` instantiates a synthetic environment and
  stub modules to verify that resets, context propagation, and command
  aggregation all operate correctly.
- `mkdocs build` ensures the documentation renders without warnings.

When integrating with the real simulator, monitor speed traces using the
longitudinal controller's optional plotting utility and inspect the generated
waypoints to confirm spline smoothness.

## Extensibility guidelines

1. Implement the appropriate protocol from `src/pipeline/interfaces.py`
   (`PerceptionModule`, `PlanningModule`, or `ControlModule`).
2. Register the new module in `build_pipeline` or compose it programmatically.
3. Declare any new hyper-parameters in `config.yml` and extend the
   configuration dataclasses.
4. Document the mathematical model and tuning recommendations in this manual.

The modular structure allows researchers to experiment with advanced perception
(e.g. neural lane segmentation), nonlinear model predictive control, or
reinforcement learning components while preserving a reproducible evaluation
harness.
