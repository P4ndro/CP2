# Modeling Active Noise-Cancelling Headphones with Implicit ODE Methods

## Abstract

This paper models Active Noise-Cancelling (ANC) headphones as a nonlinear ODE system. We compare two implicit schemes (Implicit Euler, Implicit Midpoint) with two solvers (Fixed-Point Iteration, Newton-Gauss-Seidel).

## 1. Model

**State Variables:**
- p(t): External noise pressure
- a(t): Anti-noise from actuator  
- e(t): Residual error

**ODE System:**
```
dp/dt = -αp + n(t)
da/dt = -βa + γ·tanh(e)
de/dt = p - a - δe
```

**Parameters:** α=2, β=50, γ=100, δ=1

**Noise Input:** n(t) = 1 + 0.3·sin(10πt) for t ≥ 0.5

## 2. Numerical Methods

**Implicit Euler (1st order):**
```
y_{k+1} = y_k + h·F(t_{k+1}, y_{k+1})
```

**Implicit Midpoint (2nd order):**
```
y_{k+1} = y_k + h·F(t_k + h/2, (y_k + y_{k+1})/2)
```

## 3. Nonlinear Solvers

**Fixed-Point Iteration:** y^(m+1) = Φ(y^(m))

**Newton-Gauss-Seidel:** Coordinate-wise Newton updates

**Tolerance:** 10⁻⁸, Max iterations: 100

## 4. Results

| h | Method | Solver | Avg Iter | Error | Status |
|---|--------|--------|----------|-------|--------|
| 0.1 | Euler | FPI | 1.3 | diverged | FAIL |
| 0.1 | Euler | NGS | 8.2 | 9.5e-3 | OK |
| 0.1 | Midpoint | FPI | 1.5 | diverged | FAIL |
| 0.1 | Midpoint | NGS | 6.2 | 5.5e-3 | OK |
| 0.01 | Euler | FPI | 12 | 2.5e-4 | OK |
| 0.01 | Euler | NGS | 4 | 2.5e-4 | OK |
| 0.01 | Midpoint | FPI | 7 | 4e-5 | OK |
| 0.01 | Midpoint | NGS | 3.6 | 4e-5 | OK |

## 5. Conclusion

1. **Newton-Gauss-Seidel** is more robust than FPI for stiff systems
2. **Implicit Midpoint** achieves higher accuracy than Implicit Euler
3. **FPI fails** at large step sizes due to stiffness

**Recommendation:** Use Implicit Midpoint + NGS for best accuracy and stability.

## References

1. Hairer, E., Wanner, G. (1996). Solving ODEs II: Stiff Problems
2. Ascher, U.M., Petzold, L.R. (1998). Computer Methods for ODEs
