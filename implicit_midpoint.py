"""
CODE 2: Implicit Midpoint Method for ANC Headphones ODE Model
Compares Fixed-Point Iteration (FPI) vs Newton-Gauss-Seidel (NGS) solvers
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# === PARAMETERS ===
ALPHA, BETA, GAMMA, DELTA = 2.0, 50.0, 100.0, 1.0
T_START, T_END = 0.0, 5.0
STEP_SIZES = [0.1, 0.05, 0.02, 0.01, 0.005]
H_REF = 0.0005
TOL, MAX_ITER = 1e-8, 100

def noise(t):
    return (1.0 + 0.3 * np.sin(10 * np.pi * t)) if t >= 0.5 else 0.0

# === SOLVERS ===
def fpi_midpoint(y_old, t_old, h):
    p_old, a_old, e_old = y_old
    t_mid = t_old + h / 2
    n = noise(t_mid)
    p, a, e = y_old.copy()
    for it in range(1, MAX_ITER + 1):
        p_prev, a_prev, e_prev = p, a, e
        pm, am, em = (p_old + p_prev)/2, (a_old + a_prev)/2, (e_old + e_prev)/2
        p = p_old + h * (-ALPHA * pm + n)
        a = a_old + h * (-BETA * am + GAMMA * np.tanh(em))
        e = e_old + h * (pm - am - DELTA * em)
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < TOL:
            return np.array([p, a, e]), it, True
        if abs(p) > 1e6 or abs(a) > 1e6 or abs(e) > 1e6:
            return np.array([p, a, e]), it, False
    return np.array([p, a, e]), MAX_ITER, False

def ngs_midpoint(y_old, t_old, h):
    p_old, a_old, e_old = y_old
    t_mid = t_old + h / 2
    n = noise(t_mid)
    p, a, e = y_old.copy()
    for it in range(1, MAX_ITER + 1):
        p_prev, a_prev, e_prev = p, a, e
        pm = (p_old + p) / 2
        p = p - (p - p_old - h * (-ALPHA * pm + n)) / (1 + h * ALPHA / 2)
        am, em = (a_old + a) / 2, (e_old + e) / 2
        a = a - (a - a_old - h * (-BETA * am + GAMMA * np.tanh(em))) / (1 + h * BETA / 2)
        pm, am, em = (p_old + p) / 2, (a_old + a) / 2, (e_old + e) / 2
        e = e - (e - e_old - h * (pm - am - DELTA * em)) / (1 + h * DELTA / 2)
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < TOL:
            return np.array([p, a, e]), it, True
    return np.array([p, a, e]), MAX_ITER, False

# === INTEGRATOR ===
def solve_midpoint(h, solver='FPI'):
    t = np.arange(T_START, T_END + h/2, h)
    y = np.zeros((len(t), 3))
    iters = []
    solve = fpi_midpoint if solver == 'FPI' else ngs_midpoint
    start = time.perf_counter()
    for k in range(len(t) - 1):
        y[k+1], it, _ = solve(y[k], t[k], h)
        iters.append(it)
    runtime = time.perf_counter() - start
    return t, y, np.mean(iters), runtime

# === MAIN ===
if __name__ == '__main__':
    print("=" * 60)
    print("IMPLICIT MIDPOINT - ANC Headphones Model")
    print("=" * 60)
    
    # Reference solution
    t_ref, y_ref, _, _ = solve_midpoint(H_REF, 'NGS')
    
    print(f"\n{'h':>6} | {'Solver':^6} | {'Avg Iter':>8} | {'Error':>10} | {'Time (ms)':>9}")
    print("-" * 55)
    
    results = []
    for h in STEP_SIZES:
        for solver in ['FPI', 'NGS']:
            t, y, avg_it, runtime = solve_midpoint(h, solver)
            idx = np.argmin(np.abs(t_ref - t[-1]))
            err = np.max(np.abs(y[-1] - y_ref[idx]))
            status = "OK" if err < 1e6 else "FAIL"
            print(f"{h:>6.3f} | {solver:^6} | {avg_it:>8.2f} | {err:>10.2e} | {runtime*1000:>9.2f} [{status}]")
            results.append((h, solver, t, y, err))
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    h_plot = 0.01
    for h, solver, t, y, err in results:
        if h == h_plot and err < 1e6:
            style = '-' if solver == 'FPI' else '--'
            axes[0].plot(t, y[:,0], style, label=solver)
            axes[1].plot(t, y[:,1], style)
            axes[2].plot(t, y[:,2], style)
    
    axes[0].set_ylabel('Noise p(t)'); axes[0].legend(); axes[0].set_title(f'Implicit Midpoint (h={h_plot})')
    axes[1].set_ylabel('Anti-noise a(t)')
    axes[2].set_ylabel('Error e(t)'); axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('results/figures/implicit_midpoint_comparison.png', dpi=150)
    plt.show()
    print("\nPlot saved to results/figures/implicit_midpoint_comparison.png")

