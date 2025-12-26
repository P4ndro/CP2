# Modeling Active Noise-Cancelling Headphones with Implicit ODE Methods

**Computational Project 2 - Numerical Programming**

---

## Abstract

This paper presents a computational study of Active Noise-Cancelling (ANC) headphones modeled as a nonlinear system of ordinary differential equations. We solve this initial value problem using two implicit numerical schemes (Implicit Euler and Implicit Midpoint) combined with two nonlinear solvers (Fixed-Point Iteration and Newton-Gauss-Seidel). Results demonstrate that Newton-Gauss-Seidel provides superior convergence properties, while Implicit Midpoint achieves higher accuracy than Implicit Euler.

---

## 1. Introduction

Active Noise Cancellation reduces unwanted sound by generating anti-noise signals that destructively interfere with external noise. This feedback control system can be modeled using ordinary differential equations, where the system's stiffness (due to fast actuator response) makes implicit numerical methods essential for stable integration.

**Objectives:**
1. Formulate an ODE model for ANC headphones
2. Implement Implicit Euler and Implicit Midpoint methods
3. Compare Fixed-Point Iteration and Newton-Gauss-Seidel solvers
4. Analyze stability, accuracy, and computational efficiency

---

## 2. Mathematical Model

### State Variables
- p(t): External noise pressure at earcup
- a(t): Anti-noise pressure from actuator
- e(t): Residual error (what the user hears)

### ODE System

$$\frac{dp}{dt} = -\alpha p + n(t)$$

$$\frac{da}{dt} = -\beta a + \gamma \tanh(e)$$

$$\frac{de}{dt} = p - a - \delta e$$

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 2.0 | Noise decay rate |
| β | 50.0 | Actuator response speed |
| γ | 100.0 | Feedback gain (stiffness) |
| δ | 1.0 | Error damping |

The high value of γ makes this system stiff, requiring implicit methods for stable integration at larger step sizes.

### Initial Conditions
p(0) = a(0) = e(0) = 0

### Noise Input
Combined step and sinusoidal noise:
$$n(t) = \begin{cases} 0 & t < 0.5 \\ 1 + 0.3\sin(10\pi t) & t \geq 0.5 \end{cases}$$

---

## 3. Numerical Methods

### 3.1 Implicit Euler (First Order)

$$y_{k+1} = y_k + h \cdot F(t_{k+1}, y_{k+1})$$

This requires solving a nonlinear system at each step.

### 3.2 Implicit Midpoint (Second Order)

$$y_{k+1} = y_k + h \cdot F\left(t_k + \frac{h}{2}, \frac{y_k + y_{k+1}}{2}\right)$$

The midpoint rule achieves second-order accuracy while maintaining A-stability.

---

## 4. Nonlinear Solvers

### 4.1 Fixed-Point Iteration (FPI)

Rewrite the implicit equation as y = Φ(y) and iterate:

$$y^{(m+1)} = \Phi(y^{(m)})$$

**Convergence:** Requires ‖∂Φ/∂y‖ < 1. For stiff systems with large h, FPI may fail to converge or require many iterations.

### 4.2 Newton-Gauss-Seidel (NGS)

Update each component sequentially using 1D Newton steps:

$$x \leftarrow x - \frac{G(x)}{G'(x)}$$

**Advantage:** Exploits system structure; typically converges in 2-4 iterations even for stiff problems.

### Stopping Criteria
- Tolerance: ‖y^(m+1) - y^(m)‖∞ < 10⁻⁸
- Maximum iterations: 100

---

## 5. Experimental Setup

- Time interval: [0, 5] seconds
- Step sizes: h ∈ {0.1, 0.05, 0.02, 0.01, 0.005}
- Reference solution: Implicit Midpoint + NGS with h = 0.0005

**Error metric:** ‖y(T) - y_ref(T)‖∞

**Note:** Large error values (> 10⁶) indicate solver divergence, not truncation error.

---

## 6. Results

### Comparison Table

| h | Method | Avg Iter | Error | Status |
|---|--------|----------|-------|--------|
| 0.1 | Euler+FPI | ~50 | FAILED | Diverged |
| 0.1 | Euler+NGS | 3-4 | ~10⁻² | OK |
| 0.1 | Midpoint+FPI | ~50 | FAILED | Diverged |
| 0.1 | Midpoint+NGS | 3-4 | ~10⁻³ | OK |
| 0.01 | Euler+FPI | 15-25 | ~10⁻³ | OK |
| 0.01 | Euler+NGS | 2-3 | ~10⁻³ | OK |
| 0.01 | Midpoint+FPI | 20-30 | ~10⁻⁴ | OK |
| 0.01 | Midpoint+NGS | 2-3 | ~10⁻⁴ | OK |

### Key Observations

1. **FPI Stability:** Fixed-Point Iteration fails for large step sizes (h ≥ 0.05) due to the stiff feedback term.

2. **NGS Robustness:** Newton-Gauss-Seidel converges reliably across all tested step sizes with 2-4 iterations per step.

3. **Accuracy:** Implicit Midpoint achieves approximately one order of magnitude better accuracy than Implicit Euler for the same step size.

4. **Efficiency:** NGS is more efficient than FPI, especially at larger step sizes where FPI either fails or requires many iterations.

---

## 7. Conclusion

For stiff ODE systems like the ANC headphone model:

1. **Newton-Gauss-Seidel** is the preferred solver due to its robust convergence properties and low iteration count.

2. **Implicit Midpoint** provides better accuracy than Implicit Euler and should be preferred when higher precision is required.

3. **Fixed-Point Iteration** is only viable for small step sizes; it fails or becomes inefficient for stiff problems at larger step sizes.

**Recommendation:** Use Implicit Midpoint with Newton-Gauss-Seidel for optimal accuracy and stability.

---

## References

1. Hairer, E., Wanner, G. (1996). Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems.
2. Ascher, U.M., Petzold, L.R. (1998). Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations.

---

## Appendix A: Code Structure

```
anc_project/
├── config.py           # Parameters and settings
├── models.py           # ODE system definition
├── solvers.py          # FPI and NGS implementations
├── integrators.py      # Implicit Euler and Midpoint
├── experiments.py      # Experiment runner
├── plots.py            # Visualization
├── run_comparison.py   # Main script (all methods)
├── run_implicit_euler.py    # Euler only
├── run_implicit_midpoint.py # Midpoint only
├── requirements.txt
├── README.md
└── results/
    ├── data/           # CSV, JSON output
    └── figures/        # Plots
```

## Appendix B: Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run full comparison
python run_comparison.py

# Run individual methods
python run_implicit_euler.py
python run_implicit_midpoint.py
```


