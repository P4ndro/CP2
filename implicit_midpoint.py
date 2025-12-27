"""
=============================================================================
CODE 2: IMPLICIT MIDPOINT METHOD (Implicit Runge-Kutta)
Active Noise-Cancelling Headphones - ODE Model
Solvers: Fixed-Point Iteration (FPI) vs Newton-Gauss-Seidel (NGS)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ========================== MODEL PARAMETERS ==========================
ALPHA = 2.0      # Noise decay rate
BETA = 50.0      # Actuator response speed  
GAMMA = 100.0    # Feedback gain (causes stiffness)
DELTA = 1.0      # Error damping

# ========================== SIMULATION SETTINGS ==========================
T_START, T_END = 0.0, 5.0
STEP_SIZES = [0.1, 0.05, 0.02, 0.01, 0.005]
H_REFERENCE = 0.0005
TOL = 1e-8
MAX_ITER = 100

# ========================== NOISE INPUT ==========================
def noise(t):
    """Combined step + sinusoidal noise input"""
    if t < 0.5:
        return 0.0
    return 1.0 + 0.3 * np.sin(10 * np.pi * t)

# ========================== ODE RIGHT-HAND SIDE ==========================
def f(t, y):
    """ODE system: dy/dt = f(t, y)"""
    p, a, e = y
    dp = -ALPHA * p + noise(t)
    da = -BETA * a + GAMMA * np.tanh(e)
    de = p - a - DELTA * e
    return np.array([dp, da, de])

# ========================== FIXED-POINT ITERATION ==========================
def solve_fpi(y_old, t_old, h):
    """
    Fixed-Point Iteration for Implicit Midpoint:
    y_{k+1} = y_k + h * f(t_k + h/2, (y_k + y_{k+1})/2)
    
    Rearranged as: y = Phi(y)
    """
    p_old, a_old, e_old = y_old
    t_mid = t_old + h / 2
    n = noise(t_mid)
    
    # Initial guess
    p, a, e = p_old, a_old, e_old
    
    for iteration in range(1, MAX_ITER + 1):
        p_prev, a_prev, e_prev = p, a, e
        
        # Midpoint values using previous guess
        p_mid = (p_old + p_prev) / 2
        a_mid = (a_old + a_prev) / 2
        e_mid = (e_old + e_prev) / 2
        
        # Fixed-point update
        p = p_old + h * (-ALPHA * p_mid + n)
        a = a_old + h * (-BETA * a_mid + GAMMA * np.tanh(e_mid))
        e = e_old + h * (p_mid - a_mid - DELTA * e_mid)
        
        # Check convergence
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < TOL:
            return np.array([p, a, e]), iteration, True
        
        # Check divergence
        if abs(p) > 1e10 or abs(a) > 1e10 or abs(e) > 1e10:
            return np.array([p, a, e]), iteration, False
    
    return np.array([p, a, e]), MAX_ITER, False

# ========================== NEWTON-GAUSS-SEIDEL ==========================
def solve_ngs(y_old, t_old, h):
    """
    Newton-Gauss-Seidel for Implicit Midpoint:
    Solve G(y) = y - y_old - h*f(t_mid, y_mid) = 0
    
    Update each component using 1D Newton step
    """
    p_old, a_old, e_old = y_old
    t_mid = t_old + h / 2
    n = noise(t_mid)
    
    # Initial guess
    p, a, e = p_old, a_old, e_old
    
    for iteration in range(1, MAX_ITER + 1):
        p_prev, a_prev, e_prev = p, a, e
        
        # Update p
        p_mid = (p_old + p) / 2
        G1 = p - p_old - h * (-ALPHA * p_mid + n)
        dG1 = 1 + h * ALPHA / 2
        p = p - G1 / dG1
        
        # Update a
        a_mid = (a_old + a) / 2
        e_mid = (e_old + e) / 2
        G2 = a - a_old - h * (-BETA * a_mid + GAMMA * np.tanh(e_mid))
        dG2 = 1 + h * BETA / 2
        a = a - G2 / dG2
        
        # Update e
        p_mid = (p_old + p) / 2
        a_mid = (a_old + a) / 2
        e_mid = (e_old + e) / 2
        G3 = e - e_old - h * (p_mid - a_mid - DELTA * e_mid)
        dG3 = 1 + h * DELTA / 2
        e = e - G3 / dG3
        
        # Check convergence
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < TOL:
            return np.array([p, a, e]), iteration, True
    
    return np.array([p, a, e]), MAX_ITER, False

# ========================== TIME INTEGRATION ==========================
def integrate(h, solver='FPI'):
    """Integrate ODE using Implicit Midpoint with specified solver"""
    t = np.arange(T_START, T_END + h/2, h)
    y = np.zeros((len(t), 3))
    y[0] = [0.0, 0.0, 0.0]  # Initial conditions
    
    iterations = []
    converged_count = 0
    solve_func = solve_fpi if solver == 'FPI' else solve_ngs
    
    start_time = time.perf_counter()
    for k in range(len(t) - 1):
        y[k+1], iters, converged = solve_func(y[k], t[k], h)
        iterations.append(iters)
        if converged:
            converged_count += 1
    runtime = time.perf_counter() - start_time
    
    return {
        't': t, 'y': y,
        'avg_iter': np.mean(iterations),
        'max_iter': np.max(iterations),
        'runtime': runtime,
        'convergence_rate': converged_count / (len(t) - 1)
    }

# ========================== MAIN PROGRAM ==========================
if __name__ == '__main__':
    os.makedirs('results/figures', exist_ok=True)
    
    print("=" * 70)
    print("IMPLICIT MIDPOINT METHOD - Active Noise-Cancelling Headphones")
    print("=" * 70)
    print(f"\nModel: dp/dt = -{ALPHA}p + n(t)")
    print(f"       da/dt = -{BETA}a + {GAMMA}*tanh(e)")
    print(f"       de/dt = p - a - {DELTA}e")
    print(f"\nTime: [{T_START}, {T_END}] seconds")
    print(f"Reference step size: h = {H_REFERENCE}")
    print(f"Tolerance: {TOL}, Max iterations: {MAX_ITER}")
    
    # Compute reference solution
    print("\nComputing reference solution...")
    ref = integrate(H_REFERENCE, 'NGS')
    print(f"Reference computed: {len(ref['t'])} points")
    
    # Run experiments
    print("\n" + "=" * 70)
    print("COMPARISON: Fixed-Point Iteration vs Newton-Gauss-Seidel")
    print("=" * 70)
    print(f"\n{'h':>7} | {'Solver':^5} | {'Avg Iter':>8} | {'Max Iter':>8} | {'Error':>10} | {'Time(ms)':>8} | Status")
    print("-" * 75)
    
    results = {'FPI': [], 'NGS': []}
    
    for h in STEP_SIZES:
        for solver in ['FPI', 'NGS']:
            res = integrate(h, solver)
            
            # Compute error vs reference
            idx = np.argmin(np.abs(ref['t'] - res['t'][-1]))
            error = np.max(np.abs(res['y'][-1] - ref['y'][idx]))
            
            status = "OK" if error < 1e6 and res['convergence_rate'] > 0.5 else "FAIL"
            error_str = f"{error:.2e}" if error < 1e6 else "DIVERGED"
            
            print(f"{h:>7.3f} | {solver:^5} | {res['avg_iter']:>8.2f} | {res['max_iter']:>8} | {error_str:>10} | {res['runtime']*1000:>8.2f} | {status}")
            
            results[solver].append({'h': h, 'result': res, 'error': error, 'status': status})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nImplicit Midpoint is a 2nd-order method (more accurate than Euler)")
    print("FPI: Fixed-Point Iteration - simple but may diverge for stiff systems")
    print("NGS: Newton-Gauss-Seidel - robust, fewer iterations, handles stiffness")
    print("\nConclusion: NGS is more stable and efficient for this stiff ODE system.")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Use h=0.01 for plotting (where both methods work)
    h_plot = 0.01
    for solver in ['FPI', 'NGS']:
        for r in results[solver]:
            if r['h'] == h_plot and r['status'] == 'OK':
                t, y = r['result']['t'], r['result']['y']
                style = '-' if solver == 'FPI' else '--'
                lw = 2 if solver == 'NGS' else 1.5
                axes[0].plot(t, y[:,0], style, linewidth=lw, label=f'{solver}')
                axes[1].plot(t, y[:,1], style, linewidth=lw)
                axes[2].plot(t, y[:,2], style, linewidth=lw)
    
    axes[0].set_ylabel('Noise p(t)', fontsize=11)
    axes[0].set_title(f'Implicit Midpoint Method (h = {h_plot})', fontsize=13)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Anti-noise a(t)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Error e(t)', fontsize=11)
    axes[2].set_xlabel('Time (seconds)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/implicit_midpoint.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved: results/figures/implicit_midpoint.png")
    print("=" * 70)
