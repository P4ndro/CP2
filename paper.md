# Modeling Active Noise-Cancelling Headphones with Implicit ODE Methods

**Computational Project #2 - Numerical Programming**  
**Kutaisi International University**

---

## 1. Introduction

Active Noise Cancellation (ANC) technology reduces unwanted ambient sounds by generating anti-noise signals through a feedback control loop. This paper models ANC headphones as a system of ordinary differential equations (ODEs) and compares two implicit numerical methods for solving the initial value problem.

**Objectives:**
- Formulate an ODE model for ANC headphones
- Implement Implicit Euler and Implicit Midpoint methods
- Compare Fixed-Point Iteration (FPI) and Newton-Gauss-Seidel (NGS) solvers
- Analyze stability, accuracy, and computational efficiency

---

## 2. Mathematical Model

### 2.1 State Variables

| Variable | Description | Units |
|----------|-------------|-------|
| p(t) | External noise pressure at earcup | Pa |
| a(t) | Anti-noise pressure from actuator | Pa |
| e(t) | Residual error (what user hears) | Pa |

### 2.2 ODE System

$$\frac{dp}{dt} = -\alpha p + n(t)$$

$$\frac{da}{dt} = -\beta a + \gamma \tanh(e)$$

$$\frac{de}{dt} = p - a - \delta e$$

The nonlinear term $\tanh(e)$ models actuator saturation. The high feedback gain $\gamma$ makes this system **stiff**, requiring implicit methods for stable integration.

### 2.3 Parameters

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| α | 2.0 | Noise decay rate |
| β | 50.0 | Actuator response speed |
| γ | 100.0 | Feedback gain (stiffness) |
| δ | 1.0 | Error damping |

### 2.4 Initial Conditions and Input

**Initial conditions:** p(0) = a(0) = e(0) = 0

**Noise input:** 
$$n(t) = \begin{cases} 0 & t < 0.5 \\ 1 + 0.3\sin(10\pi t) & t \geq 0.5 \end{cases}$$

---

## 3. Numerical Methods

### 3.1 Implicit Euler (First Order)

For $\dot{y} = f(t, y)$, the implicit Euler scheme is:

$$y_{k+1} = y_k + h \cdot f(t_{k+1}, y_{k+1})$$

This requires solving a nonlinear system at each time step.

### 3.2 Implicit Midpoint (Second Order)

The implicit midpoint rule achieves second-order accuracy:

$$y_{k+1} = y_k + h \cdot f\left(t_k + \frac{h}{2}, \frac{y_k + y_{k+1}}{2}\right)$$

Both methods are A-stable, suitable for stiff problems.

---

## 4. Nonlinear Solvers

### 4.1 Fixed-Point Iteration (FPI)

Rewrite the implicit equation as $y = \Phi(y)$ and iterate:

$$y^{(m+1)} = \Phi(y^{(m)})$$

**Convergence condition:** $\|\partial\Phi/\partial y\| < 1$

For stiff systems, FPI may require many iterations or diverge at large step sizes.

### 4.2 Newton-Gauss-Seidel (NGS)

Apply coordinate-wise Newton updates to solve $G(y) = 0$:

$$x_i \leftarrow x_i - \frac{G_i(x)}{{\partial G_i}/{\partial x_i}}$$

NGS exploits the system structure and typically converges in 3-5 iterations.

### 4.3 Stopping Criteria

- Tolerance: $\|y^{(m+1)} - y^{(m)}\|_\infty < 10^{-8}$
- Maximum iterations: 100

---

## 5. Experimental Setup

- **Time interval:** [0, 5] seconds
- **Step sizes tested:** h ∈ {0.1, 0.05, 0.02, 0.01, 0.005}
- **Reference solution:** h = 0.0005 with NGS solver

**Error metric:** $\|y(T) - y_{ref}(T)\|_\infty$

---

## 6. Results

### 6.1 Implicit Euler Results

| h | Solver | Avg Iter | Error | Status |
|---|--------|----------|-------|--------|
| 0.100 | FPI | 1.3 | DIVERGED | FAIL |
| 0.100 | NGS | 8.2 | 9.5e-03 | OK |
| 0.050 | FPI | 1.2 | DIVERGED | FAIL |
| 0.050 | NGS | 6.3 | 2.7e-03 | OK |
| 0.020 | FPI | 90.5 | 6.5e-04 | OK |
| 0.020 | NGS | 4.8 | 6.5e-04 | OK |
| 0.010 | FPI | 12.0 | 2.5e-04 | OK |
| 0.010 | NGS | 4.0 | 2.5e-04 | OK |

### 6.2 Implicit Midpoint Results

| h | Solver | Avg Iter | Error | Status |
|---|--------|----------|-------|--------|
| 0.100 | FPI | 1.5 | DIVERGED | FAIL |
| 0.100 | NGS | 6.2 | 5.5e-03 | OK |
| 0.050 | FPI | 2.1 | DIVERGED | FAIL |
| 0.050 | NGS | 5.1 | 1.1e-03 | OK |
| 0.020 | FPI | 12.7 | 1.6e-04 | OK |
| 0.020 | NGS | 4.1 | 1.6e-04 | OK |
| 0.010 | FPI | 6.9 | 4.0e-05 | OK |
| 0.010 | NGS | 3.7 | 4.0e-05 | OK |

### 6.3 Method Comparison

| Aspect | Implicit Euler | Implicit Midpoint |
|--------|----------------|-------------------|
| Order of accuracy | 1 | 2 |
| Error at h=0.01 | 2.5e-04 | 4.0e-05 |
| Stability | A-stable | A-stable |
| Computational cost | Lower | Slightly higher |

---

## 7. Visualizations

Both codes generate plots showing:
- p(t): External noise pressure
- a(t): Anti-noise actuator response
- e(t): Residual error at the ear

The plots demonstrate that ANC effectively reduces the residual error e(t) after noise onset at t=0.5s.

---

## 8. Conclusions

1. **Newton-Gauss-Seidel outperforms Fixed-Point Iteration:**
   - NGS converges in 3-8 iterations regardless of step size
   - FPI diverges at large step sizes (h ≥ 0.05) due to stiffness
   - FPI requires 10-90+ iterations when it does converge

2. **Implicit Midpoint is more accurate than Implicit Euler:**
   - Second-order accuracy vs first-order
   - ~6x smaller error at the same step size

3. **Recommendation:** For stiff ODE systems like ANC models, use **Implicit Midpoint with Newton-Gauss-Seidel** for optimal accuracy and reliability.

---

## References

1. Hairer, E., Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer.
2. Ascher, U.M., Petzold, L.R. (1998). *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*. SIAM.

---

## Appendix: Code Structure

```
anc_project/
├── implicit_euler.py      # Code 1: Implicit Euler method
├── implicit_midpoint.py   # Code 2: Implicit Midpoint method
├── paper.md               # This paper
├── README.md              # Usage instructions
├── requirements.txt       # Dependencies (numpy, matplotlib)
└── results/figures/       # Generated plots
```

**Usage:**
```bash
python implicit_euler.py
python implicit_midpoint.py
```
