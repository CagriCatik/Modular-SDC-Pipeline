# Testing and Validation

The simulator is the ground truth for this stack. Use the following
recommendations to evaluate changes consistently.

## Unit-level checks

* **Lane detection** – run against stored screenshots and assert spline validity
  (`t`, `c`, `k` shapes, finite coefficients).
* **Waypoint prediction** – feed synthetic splines (straight line, S-curve) and
  verify that the smoothing objective reduces curvature as expected.
* **Longitudinal PID** – simulate a step response with a target speed profile
  and ensure the gas/brake outputs stay within `[0, 0.8]`.

## Integration loops

Run the orchestrator with the standard settings:

```bash
python modular_pipeline.py --no_display
```

Expected behaviour:

1. Vehicle remains centred in the lane without oscillatory steering.
2. Target speed ramps up on straight sections and drops before sharp turns.
3. Episode rewards should be positive and stable across seeds.

For regression testing, the helper `calculate_score_for_leaderboard` evaluates
the seeds specified under `evaluation.score_seeds` in `config.yml` and prints the
clipped reward per episode followed by the mean score.
