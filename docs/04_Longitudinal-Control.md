# Longitudinal Control of a Vehicle 

## Abstract

Longitudinal control governs the **forward motion** of an autonomous vehicle by modulating throttle and braking to maintain a target velocity derived from the path planner. This section presents a rigorous, scientific formulation of the **PID (Proportional–Integral–Derivative) controller**, its discrete implementation for digital systems, and tuning strategies for stable and responsive speed control.
The objective is to minimize velocity tracking error while ensuring comfort, safety, and dynamic stability under variable road and vehicle conditions.

---

## 1. Introduction

Longitudinal control is the temporal counterpart to lateral control. While lateral control manages the **spatial trajectory**, longitudinal control dictates **how fast** the vehicle progresses along that trajectory.

The design must satisfy two constraints:

1. **Accuracy** — precisely follow the desired speed profile.
2. **Smoothness** — avoid jerky accelerations or overshoot that degrade comfort or traction.

The **PID controller** remains the cornerstone of industrial control due to its simplicity, interpretability, and well-understood dynamics. When properly tuned, it achieves near-optimal performance for a wide range of linear and quasi-linear systems, including vehicle longitudinal dynamics.

---

## 2. The PID Control Law

### 2.1 Control Objective

The goal is to minimize the instantaneous **velocity tracking error**:
[
e(t) = v_{\text{target}}(t) - v(t)
]
where ( v_{\text{target}} ) is the desired velocity profile and ( v(t) ) is the measured velocity.

The controller outputs a unified control signal ( u(t) ), determining throttle or braking intensity. The system continuously updates ( u(t) ) at discrete time intervals ( \Delta t ).

---

### 2.2 Discrete PID Formulation

The control law is implemented in discrete time as:
[
u(t) = K_p e(t) + K_i \sum_{l=0}^{t} e(l) \Delta t + K_d \frac{e(t) - e(t-1)}{\Delta t}
]

Where:

* ( K_p ) controls proportional response to instantaneous error.
* ( K_i ) accumulates past errors to remove steady-state bias.
* ( K_d ) anticipates error change to dampen oscillations.

The throttle and brake commands are derived directly from ( u(t) ):
[
a_{\text{gas}}(t) = \max(0, u(t)), \quad a_{\text{brake}}(t) = \max(0, -u(t))
]

This ensures mutual exclusivity—only one actuator is active at a time, consistent with vehicle hardware constraints.

---

### 2.3 Control-Theoretic Interpretation

The PID controller represents a **closed-loop negative feedback system**. The proportional term reacts to instantaneous deviations, the integral term ensures zero steady-state error, and the derivative term predicts future behavior.

In the Laplace domain, the continuous PID transfer function is:
[
G_c(s) = K_p + \frac{K_i}{s} + K_d s
]

The **proportional** component contributes immediate corrective action.
The **integral** component adds memory, driving long-term accuracy.
The **derivative** component introduces predictive damping, shaping system response akin to velocity feedback in mechanical systems.

Together, they form a controller capable of first-order and second-order dynamic compensation, approximating ideal behavior for a second-order plant—such as vehicle acceleration governed by engine torque and drag forces.

---

## 3. Understanding the Components

### 3.1 Proportional Term ( K_p e(t) )

* Reacts to current deviation.
* Larger ( K_p ) increases responsiveness but risks **oscillations** and **overshoot** if the system becomes underdamped.
* Physically, this term controls the aggressiveness of throttle and braking response.

### 3.2 Integral Term ( K_i \sum e(l) )

* Compensates for **system bias**—for example, aerodynamic drag or rolling resistance—by accumulating past errors.
* Eliminates steady-state offset where the car stabilizes below target speed.
* However, excessive accumulation leads to **integral windup**, a nonlinear phenomenon where the integrator drives the actuator beyond its physical limits.

### 3.3 Derivative Term ( K_d [e(t) - e(t-1)] )

* Responds to the **rate of error change**, providing anticipatory correction.
* Helps prevent overshoot and stabilizes transient response, functioning as a form of **velocity feedback**.
* Sensitivity to measurement noise requires signal filtering (e.g., a first-order low-pass filter).

---

## 4. Integral Windup and Anti-Windup Techniques

### 4.1 Problem Definition

In real systems, actuators saturate. For example, throttle cannot exceed 100%, and braking cannot go below 0. When such limits are reached, the integral term continues to accumulate error, leading to **overshoot** once the actuator is released.

### 4.2 Mitigation Strategies

1. **Integral Clamping**

   * Bound the integral accumulation term within a predefined range:
     [
     I_{\text{clamped}} = \text{clip}(I, -I_{\max}, I_{\max})
     ]
   * Prevents runaway accumulation.

2. **Conditional Integration**

   * Freeze integral updates when the control signal reaches saturation limits.

3. **Back-Calculation Anti-Windup**

   * Introduce a feedback term that subtracts the difference between saturated and unsaturated control signals, effectively "bleeding off" accumulated error.

These techniques ensure numerical stability and enable predictable response recovery after saturation events such as sudden stops or steep inclines.

---

## 5. PID Parameter Tuning

### 5.1 Empirical Tuning Methodology

PID tuning is an **iterative experimental process**, as real vehicle dynamics are nonlinear and time-varying. The general tuning procedure:

1. **Proportional Phase:**

   * Set ( K_i = 0, K_d = 0 ).
   * Increase ( K_p ) until the system responds rapidly without oscillating excessively.

2. **Derivative Phase:**

   * Gradually increase ( K_d ) to reduce overshoot and improve damping.
   * Observe the step response—higher ( K_d ) produces smoother convergence but can amplify sensor noise.

3. **Integral Phase:**

   * Introduce ( K_i ) to eliminate steady-state error.
   * Apply anti-windup logic to constrain integral growth.

---

### 5.2 Quantitative Evaluation

Performance can be measured by:

* **Rise time**: time to reach 90% of the target speed.
* **Settling time**: time to stabilize within ±5% of the target.
* **Overshoot**: percentage by which speed exceeds target.
* **Steady-state error**: residual difference after settling.

Optimization methods such as **Ziegler–Nichols** or **Cohen–Coon** rules provide initial parameter estimates, but fine-tuning is typically empirical due to the vehicle’s nonlinear friction and torque dynamics.

---

### 5.3 Parameter Sensitivity

| Parameter | Function                   | Increasing Value Effects                        |
| --------- | -------------------------- | ----------------------------------------------- |
| ( K_p )   | Immediate error correction | Faster response, higher overshoot               |
| ( K_i )   | Long-term bias elimination | Lower steady-state error, possible oscillations |
| ( K_d )   | Predictive damping         | Reduced overshoot, noise sensitivity            |

The tuning must adapt to external conditions such as road slope, aerodynamic drag, and tire friction. Hence, practical implementations often employ **gain scheduling**, adjusting gains dynamically based on vehicle speed or load.

---

## 6. System Implementation and Validation

### 6.1 Discrete Implementation in Software

The controller can be implemented in `longitudinal_control.py` as follows:

* Maintain persistent variables for ( e(t-1) ) and the integral sum.
* Apply clamping to limit the control signal within actuator bounds.
* Use consistent sampling intervals (e.g., 50–100 Hz) to ensure numerical stability.

### 6.2 Testing and Visualization

`test_longitudinal_control.py` provides time-series plots of target vs. actual velocity.
Through iterative tests, the user can visualize convergence, overshoot, and steady-state performance, tuning gains accordingly.

### 6.3 Safety Considerations

* Avoid excessively high ( K_p ) or ( K_i ) values, which can cause aggressive acceleration or braking.
* Validate under varying conditions—flat, uphill, and downhill—to ensure robustness.

---

## 7. Scientific Context and Discussion

From a control theory perspective, the PID controller embodies a **low-order approximation of optimal feedback**. It can be derived from linear quadratic regulator (LQR) principles under simplified plant dynamics, where proportional, integral, and derivative actions correspond to state, accumulation, and rate feedback respectively.

Despite its simplicity, the PID framework remains prevalent due to:

* **Computational efficiency**—requires no model inversion or matrix operations.
* **Interpretability**—each term has a clear physical meaning.
* **Adaptability**—can be extended into advanced architectures such as **PI-D cascades**, **adaptive PID**, or **model predictive control (MPC)**.

In vehicle dynamics, longitudinal PID control provides a foundation for higher-level adaptive cruise control, traction management, and energy-optimized driving strategies.

---

## 8. Conclusion

The longitudinal control system translates velocity commands into actionable throttle and braking inputs through a **discrete PID controller**. By combining reactive, integrative, and predictive behaviors, it achieves stable speed tracking across diverse driving conditions.

Scientific rigor in design—proper discretization, anti-windup protection, and gain tuning—ensures both robustness and comfort.
This module demonstrates how classical control theory, when precisely implemented and tuned, continues to outperform complex learning-based systems in safety-critical tasks requiring **stability, interpretability, and verifiability**.


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
