# Lane Detection

`LaneDetection` extracts left and right roadside splines from the 96×96 RGB
rendered frame.

## Processing stages

1. **Cropping and grayscale conversion** – removes the dashboard HUD and reduces
   dimensionality while keeping gradients intact.
2. **Gradient magnitude** – computes Sobel-style derivatives and thresholds weak
   responses below `gradient_threshold`.
3. **Row-wise peak search** – uses `scipy.signal.find_peaks` with a configurable
   minimum spacing to produce candidate lane pixels.
4. **Sequential association** – tracks the closest peak in each row to grow the
   left and right lane hypotheses upward in the cropped image.
5. **Spline fitting** – fits cubic B-splines (`scipy.interpolate.splprep`) once a
   sufficient number of points is collected; otherwise the previous spline is
   reused.

## Practical guidance

* Tune `cut_size` to balance field of view and computation. A value of 68 keeps
  the road while removing most of the sky.
* Increase `gradient_threshold` when track textures cause false positives.
* The module preserves the last valid splines to prevent sudden controller
  oscillations when the road temporarily disappears.
* `plot_state_lane` is available for debugging; it overlays the predicted lanes
  and optional waypoints on the rendered frame.
