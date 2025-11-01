# Modular Pipeline for Self-Driving Cars

## Overview

This project implements a **rule-based modular pipeline** for autonomous driving using classical computer vision and control theory. The system is structured into four major components:

1. **Lane Detection**
2. **Path Planning**
3. **Lateral Control**
4. **Longitudinal Control**

The implementation avoids any learning-based approaches, relying instead on deterministic algorithms built with **NumPy** and **SciPy**. The project serves as an educational and scientific framework for understanding the fundamentals of self-driving systems.

---

## 1. Setup

Place all Python modules in the project root directory:

```
lane_detection.py
waypoint_prediction.py
lateral_control.py
longitudinal_control.py
modular_pipeline.py
```

Your submission must also include `submission.txt` with author and parameter information.

The system is tested on the **TCML cluster** using a **Singularity container**. Ensure that your code runs within this environment without additional package installations.

Command to execute in the cluster:

```
singularity exec /path/to/singularity/sdc_gym_amd64.simg python3 modular_pipeline.py --score
```

---

## 2. Lane Detection

### Objective

Detect lane boundaries from the simulator’s camera image and represent them as smooth parametric splines.

### Steps

1. **Edge Detection**

   * Convert the RGB image to grayscale.
   * Crop the area above the car.
   * Compute gradient magnitudes and apply a threshold.
   * Identify local maxima per image row using `scipy.signal.find_peaks()`.

2. **Edge Association**

   * Initialize lane boundaries from the lowest image row.
   * Propagate lane points by nearest-neighbor search to form continuous left and right boundaries.

3. **Spline Fitting**

   * Fit B-splines using `scipy.interpolate.splprep` and evaluate with `scipy.interpolate.splev`.
   * Adjust smoothing parameters for robustness.

4. **Testing**

   * Validate using `test_lane_detection.py`.
   * Tune gradient thresholds, cropping, and spline parameters to minimize failure cases.

---

## 3. Path Planning

### Objective

Generate a smooth, drivable trajectory by computing waypoints along the road centerline.

### Steps

1. **Road Center**

   * Evaluate both lane splines at six equally spaced parameter values.
   * Compute the center point between corresponding left and right spline positions.

2. **Path Smoothing**

   * Optimize waypoints by minimizing curvature while staying near the road center:

     ```
     minimize Σ|y_i - x_i|^2 - β Σ((x_{n+1} - x_n) · (x_n - x_{n-1})) / (|x_{n+1} - x_n| |x_n - x_{n-1}|)
     ```
   * The curvature term encourages smooth, natural cornering.

3. **Target Speed Prediction**

   * Compute target velocity based on path curvature:

     ```
     v_target = (v_max - v_min) * exp(-K_v * curvature) + v_min
     ```
   * Recommended parameters: `v_max = 60`, `v_min = 30`, `K_v = 4.5`.

Test with `test_waypoint_prediction.py` and inspect the resulting paths visually.

---

## 4. Lateral Control

### Objective

Control steering to follow the planned path using the **Stanley Controller**.

### Control Law

```
δ_SC(t) = ψ(t) + arctan(k * d(t) / v(t))
```

* ψ(t): orientation error
* d(t): cross-track error
* v(t): vehicle speed
* k: gain parameter (empirically tuned)

### Damping

Smooth steering transitions to prevent oscillations:

```
δ(t) = δ_SC(t) - D * (δ_SC(t) - δ(t - 1))
```

Adjust `D` for stability and responsiveness.

Test with `test_lateral_control.py` to observe steering dynamics.

---

## 5. Longitudinal Control

### Objective

Control acceleration and braking to match the target speed using a **PID controller**.

### PID Formulation

```
e(t) = v_target - v(t)
u(t) = Kp * e(t) + Kd * [e(t) - e(t-1)] + Ki * Σ e(t_i)
```

### Actuator Mapping

```
gas(t)   = max(0, u(t))
brake(t) = max(0, -u(t))
```

### Notes

* Apply upper limits to the integral term to prevent windup.
* Tune parameters incrementally:

  * Start with `Kp = 0.01`, `Ki = 0`, `Kd = 0`.
  * Adjust one parameter at a time while observing plots in `test_longitudinal_control.py`.

---

## 6. Execution and Evaluation

Each module is executed sequentially within `modular_pipeline.py`.
During startup, a brief camera zoom may affect visual algorithms; the variable `t` (time since episode start) can be used to mitigate initialization effects.

The evaluation process:

* Models are tested over multiple validation tracks.
* For each track, reward after 600 frames is computed.
* Mean reward across all tracks determines performance.
* Final competition results are based on 100 unseen tracks.

Reward function:

```
reward = -0.1 per frame + (1000 / N) per tile visited
```

---

## 7. Design Philosophy

This pipeline exemplifies the **Sense–Plan–Act** paradigm:

* **Sense:** Extract structured lane information from raw vision.
* **Plan:** Generate a geometrically smooth and dynamically feasible path.
* **Act:** Use feedback control to achieve stable steering and speed regulation.

Unlike deep learning systems, this modular approach is **transparent, reproducible, and interpretable**, making it ideal for scientific exploration and education in autonomous vehicle engineering.

---

## 8. References

* SciPy Interpolation: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html)
* Stanley Controller (DARPA Grand Challenge): [http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
* PID Control Fundamentals: [https://homepages.inf.ed.ac.uk/mherrman/IVRINVERTED/pdfs/PID_control.pdf](https://homepages.inf.ed.ac.uk/mherrman/IVRINVERTED/pdfs/PID_control.pdf)

---

## 9. License

This project is for academic and educational use under the Autonomous Vision Group, University of Tübingen. Redistribution or adaptation should acknowledge the original course materials.
