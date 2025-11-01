# Path Planning (Scientific Description and Design Rationale)

## Abstract

Path planning is the cognitive layer of the modular autonomous driving pipeline. It transforms geometric lane information into a **continuous, dynamically feasible trajectory** by computing road-center waypoints, smoothing them through optimization, and assigning a **curvature-dependent target speed**. Unlike reactive control strategies, this approach explicitly formulates the geometry of motion, separating perception from decision-making. The system ensures stable navigation, energy efficiency, and safety under varying road geometries.

---

## 1. Introduction

In an autonomous driving stack, path planning bridges **perception** and **control**. It translates the lane geometry derived from vision into actionable spatial and temporal references for the vehicle.
The planner’s role is to generate a **reference path** that minimizes steering effort and maintains stability, while ensuring that the car remains within the detected lane boundaries.

This module decomposes into three sequential subproblems:

1. **Road-center waypoint prediction** – defines a discrete geometric skeleton of the lane center.
2. **Path smoothing** – regularizes the trajectory to ensure curvature continuity and physical plausibility.
3. **Target speed prediction** – modulates velocity based on curvature, enforcing safety and comfort.

This formulation is deterministic, interpretable, and grounded in analytical optimization, adhering to the principles of model-based motion planning.

---

## 2. Road Center Waypoint Prediction

### 2.1 Geometric Motivation

The lane boundaries, represented as parametric splines ( L(s) ) and ( R(s) ), define the feasible driving corridor. The road centerline ( C(s) ) can be estimated as the midpoint between these boundaries:
[
C(s) = \frac{L(s) + R(s)}{2}, \quad s \in [0, 1]
]
This representation approximates the road’s geometric symmetry and serves as the **nominal path**.

---

### 2.2 Discretization into Waypoints

**Step 1: Sampling along spline parameter space**
Let ( s_0, s_1, ..., s_N ) be **equispaced parameters** in [0, 1]. For each ( s_i ):
[
L_i = L(s_i), \quad R_i = R(s_i), \quad C_i = \frac{L_i + R_i}{2}
]
The resulting set ( { C_i }_{i=1}^N ) forms a polyline approximating the lane center.

**Step 2: Parameter selection**
Including ( s_0 = 0 ) ensures that the first waypoint lies immediately ahead of the vehicle. The number of waypoints ( N ) governs the planning horizon—too few reduce path granularity; too many increase computational load without substantial benefit.

**Step 3: Waypoint generation function**
The function `waypoint_prediction()` should therefore produce:
[
\text{waypoints} = [C_0, C_1, ..., C_N]
]
where each ( C_i ) is expressed in vehicle-relative coordinates.

---

### 2.3 Numerical Considerations

* **Consistency:** The sampling interval in ( s )-space should correspond approximately to uniform spatial distances to maintain constant waypoint density.
* **Noise Sensitivity:** Use low-order spline smoothing to mitigate jitter from lane boundary irregularities.
* **Validation:** Visual inspection and runtime testing under diverse lighting and curvature conditions ensure stability.

This stage constructs a geometrically valid path that serves as the baseline for dynamic refinement.

---

## 3. Path Smoothing

### 3.1 Theoretical Background

The raw centerline may contain small discontinuities or oscillations due to imperfect perception. Path smoothing reformulates the trajectory as an **energy minimization problem**, seeking a trade-off between fidelity to the lane center and curvature minimization.

The objective function is defined as:
[
\min_{x_1, \ldots, x_N} \sum_i | y_i - x_i |^2 - \beta \sum_n \frac{(x_{n+1} - x_n) \cdot (x_n - x_{n-1})}{|x_{n+1} - x_n| |x_n - x_{n-1}|}
]
where:

* ( y_i ): raw center points from perception
* ( x_i ): optimized waypoints
* ( \beta ): curvature weighting coefficient

The second term approximates the **cosine of the turning angle**, promoting alignment between consecutive segments. Maximizing this cosine value penalizes abrupt direction changes, enforcing **curvature continuity**.

---

### 3.2 Mathematical Interpretation

The curvature-related term,
[
\kappa_i = 1 - \frac{(x_{i+1} - x_i) \cdot (x_i - x_{i-1})}{|x_{i+1} - x_i| |x_i - x_{i-1}|}
]
acts as a discrete analog of the second derivative along the curve, i.e., curvature energy. Minimizing this measure ensures smoother transitions, akin to minimizing **bending energy** in spline theory.

The balance parameter ( \beta ) controls the trade-off:

* Small ( \beta ): prioritizes lane-center adherence.
* Large ( \beta ): prioritizes geometric smoothness and corner cutting.

This optimization is conceptually similar to elastic band models in robotics, where the path is treated as a flexible curve constrained by environmental boundaries.

---

### 3.3 Implementation and Optimization

The `curvature()` function should compute the curvature penalty across all consecutive triplets of points. Optimization can be achieved via gradient descent or direct analytical adjustments using local smoothing heuristics.

**Empirical tuning:**
Begin with ( \beta \in [0.1, 1.0] ) and incrementally adjust based on visual curvature smoothness. Excessive smoothing may lead to boundary violations; insufficient smoothing can result in jerky steering.

---

## 4. Target Speed Prediction

### 4.1 Physical Motivation

The optimal vehicle speed varies inversely with curvature: tighter turns require slower velocities to maintain lateral stability, while straight paths permit acceleration. This relationship stems from the **centripetal acceleration constraint**:
[
a_c = \frac{v^2}{r} \leq a_{\text{max}}
]
where ( r ) is the turning radius and ( a_{\text{max}} ) is the tire grip limit. Hence, speed must adapt dynamically to curvature magnitude.

---

### 4.2 Curvature-Dependent Speed Model

The target velocity is defined by an exponential decay function:
[
v_{\text{target}} = (v_{\max} - v_{\min}) \exp[-K_v \cdot C] + v_{\min}
]
where ( C ) is the aggregated curvature measure:
[
C = N - 2 - \sum_n \frac{(x_{n+1} - x_n) \cdot (x_n - x_{n-1})}{|x_{n+1} - x_n| |x_n - x_{n-1}|}
]
and ( K_v ) determines the sensitivity of speed to curvature.

This formulation ensures smooth modulation between ( v_{\min} ) and ( v_{\max} ), yielding **velocity continuity** without abrupt acceleration changes.

---

### 4.3 Parameter Significance

* ( v_{\max} ): Upper bound for straight sections (e.g., 60 units).
* ( v_{\min} ): Lower bound for sharp turns (e.g., 30 units).
* ( K_v ): Exponential curvature gain (default ≈ 4.5).

**Interpretation:**
When ( C \to 0 ) (straight path), the exponential term approaches 1, driving ( v_{\text{target}} \approx v_{\max} ).
As ( C ) increases (tight curves), the exponential decays, reducing speed toward ( v_{\min} ).

---

### 4.4 Implementation and Testing

The `target_speed_prediction()` function computes curvature across the smoothed waypoints and applies the above equation. Testing should include:

* **Straight-line motion:** validate constant high-speed behavior.
* **Mild curvature:** ensure smooth deceleration and recovery.
* **Sharp corners:** confirm stability through gradual braking.

Parameter tuning must consider both comfort and safety, ensuring that longitudinal acceleration and jerk remain within physically acceptable limits.

---

## 5. Integration and Evaluation

When integrated into the complete modular pipeline, the path planning subsystem outputs:

1. A spatial path (set of smoothed waypoints).
2. A temporal profile (target velocity along the path).

These outputs feed directly into **lateral** and **longitudinal control** modules, forming the foundation of a model-based motion planner.

Quantitative metrics for evaluation include:

* **Path smoothness:** curvature variance across waypoints.
* **Lateral deviation:** RMS distance from true lane center.
* **Speed adaptation:** correlation between curvature and velocity.

Testing across diverse synthetic tracks establishes generalization and parameter robustness.

---

## 6. Scientific Significance

This module exemplifies the **geometric control paradigm** in robotics: decomposing motion into spatial and temporal optimization governed by physical constraints. The explicit curvature penalty connects perceptual geometry to dynamic feasibility, aligning the mathematical path with the vehicle’s dynamic envelope.

Such deterministic planning frameworks remain crucial in safety-critical contexts where **verifiability and interpretability** outweigh raw adaptability. The combination of spline geometry, discrete curvature optimization, and exponential velocity modulation constitutes a theoretically grounded and computationally efficient solution for real-time autonomous navigation.

---

## 7. Conclusion

The path planning module formalizes the transformation from **detected lanes** to **drivable trajectories**. Through rigorous geometric reasoning, it achieves a balance between adherence to lane structure and dynamic smoothness. The inclusion of curvature-based speed prediction extends spatial reasoning into the temporal domain, enabling motion that is both safe and efficient.

This scientifically structured approach demonstrates that high-performance autonomous behavior can emerge from principled modeling, optimization, and control—without reliance on black-box learning.
