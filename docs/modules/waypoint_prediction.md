# Waypoint and Speed Planning

The planning layer converts lane splines into a drivable centreline and assigns
an appropriate target speed.

## `waypoint_prediction`

* Accepts two spline tuples `(t, c, k)` from `LaneDetection`.
* Evaluates both splines across `num_waypoints` samples and averages them to
  obtain a centreline.
* Optionally optimises the points with L-BFGS-B to reduce curvature while
  staying close to the centreline (`way_type="smooth"`).
* Falls back to a straight-ahead path if spline evaluation fails or produces
  non-finite values.

## `target_speed_prediction`

* Computes heading changes across the first `num_waypoints_used` samples and
  maps curvature to a speed using an exponential decay.
* Guarantees non-negative output and enforces a lower bound (`min_speed`).
* `K_v` controls how aggressively the controller slows down for tight bends.

## Tuning checklist

| Symptom | Suggested change |
|---------|------------------|
| Vehicle cuts corners | Increase `num_waypoints` or the smoothing weight `beta`. |
| Oscillatory steering | Reduce `num_waypoints_used` so speed reacts less to far-away curvature. |
| Vehicle too cautious | Raise `max_speed` or lower `K_v`. |
