# Lateral Control

`LateralController` implements a Stanley controller with damping to drive the
steering command.

## Control law

* Uses the first segment of the waypoint path to estimate the centreline
  heading.
* Computes the cross-track error from the nearest waypoint.
* Applies the Stanley formula `δ = ψ + arctan(k·d / v)` with a small damping term
  to suppress oscillations at low speed.
* Clamps the output to ±0.4 rad and scales to the `[-1, 1]` range expected by
  CarRacing.

## Operational notes

* Call `reset()` when starting a new episode to clear the steering memory.
* Provide `speed` in the same units as the longitudinal controller (the wrapper
  returns metres per second).
* Invalid waypoints trigger the previous steering command instead of zero to
  maintain continuity.
