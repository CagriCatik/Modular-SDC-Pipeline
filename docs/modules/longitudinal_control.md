# Longitudinal Control

`LongitudinalController` uses a PID regulator to map the target speed to gas and
brake commands.

## PID design

* **Proportional (`KP`)** – matches the steady-state acceleration to the speed
  error.
* **Integral (`KI`)** – compensates for drag or road grade; the accumulated error
  is clamped by `integral_windup_limit` to avoid overshoot.
* **Derivative (`KD`)** – damps oscillations by reacting to error changes. The
  implementation guards against division by zero by enforcing a minimum `dt`.

## Command translation

The PID output is interpreted as longitudinal force:

* Positive command → throttle, clipped to `[0, max_gas]` (default 0.8).
* Negative command → brake, clipped to `[0, max_brake]` (default 0.8).
* Both channels cannot be active simultaneously.

## Episode management

Call `reset()` before each episode to clear the integrator and derivative state.
The controller maintains optional histories (`speed_history`,
`target_speed_history`) that can be plotted for debugging with `plot_speed`.

## Recommended gains

| Condition | KP | KI | KD | Notes |
|-----------|----|----|----|-------|
| Default flat track | 0.08 | 0.01 | 0.02 | Aggressive but stable at 50 Hz. |
| High curvature sections | 0.06 | 0.02 | 0.03 | Slightly more damping to handle rapid slowdowns. |
| Slippery surfaces | 0.04 | 0.02 | 0.0 | Avoids derivative noise from wheel slip. |

Tune KP first, then KI, and finally KD. Always re-validate using the
leaderboard seeds defined in `config.yml`.

## Configuration hooks

Adjust PID gains, integrator clamp, and actuator saturation via
`control.longitudinal` in `config.yml`. Reduce `max_gas`/`max_brake` when
testing hardware-in-the-loop rigs to respect actuator limits.
