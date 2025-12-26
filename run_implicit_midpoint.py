"""
ANC Headphones ODE Model - Implicit Midpoint Method
Compares Fixed-Point Iteration vs Newton-Gauss-Seidel solvers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import config
from models import ANCParameters, noise_combined
from integrators import solve_implicit_midpoint
from experiments import compute_reference_solution, compute_error_vs_reference
from utils import is_solver_failure, get_failure_status


def main():
    os.makedirs(config.OUTPUT_FIGURES_DIR, exist_ok=True)
    
    params = ANCParameters()
    t_span = (config.T_START, config.T_END)
    y0 = np.array([0.0, 0.0, 0.0])
    
    print("=" * 60)
    print("Implicit Midpoint Method - ANC Headphones Model")
    print("=" * 60)
    
    ref_result = compute_reference_solution(params, t_span, noise_combined)
    
    print("\nComparing FPI vs NGS:")
    print("-" * 60)
    
    for h in config.STEP_SIZES:
        result_fpi = solve_implicit_midpoint(y0, t_span, h, params, noise_combined,
                                             solver='FPI', tol=config.TOL_FPI,
                                             max_iter=config.MAX_ITER_FPI)
        result_ngs = solve_implicit_midpoint(y0, t_span, h, params, noise_combined,
                                             solver='NGS', tol=config.TOL_NGS,
                                             max_iter=config.MAX_ITER_NGS)
        
        err_fpi = compute_error_vs_reference(result_fpi, ref_result)
        err_ngs = compute_error_vs_reference(result_ngs, ref_result)
        
        status_fpi = "OK" if not is_solver_failure(err_fpi) else "FAIL"
        status_ngs = "OK" if not is_solver_failure(err_ngs) else "FAIL"
        
        print(f"h={h:5.3f}: FPI(iters={result_fpi['avg_iterations']:5.1f}, err={err_fpi:.2e}) [{status_fpi}] | "
              f"NGS(iters={result_ngs['avg_iterations']:5.1f}, err={err_ngs:.2e}) [{status_ngs}]")
    
    h_demo = 0.01
    result_fpi = solve_implicit_midpoint(y0, t_span, h_demo, params, noise_combined, solver='FPI')
    result_ngs = solve_implicit_midpoint(y0, t_span, h_demo, params, noise_combined, solver='NGS')
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result_fpi['t'], result_fpi['p'], 'b-', label='FPI', linewidth=1.5)
    axes[0].plot(result_ngs['t'], result_ngs['p'], 'r--', label='NGS', linewidth=1.5)
    axes[0].set_ylabel('Noise p(t)')
    axes[0].legend()
    axes[0].set_title(f'Implicit Midpoint: FPI vs NGS (h = {h_demo})')
    
    axes[1].plot(result_fpi['t'], result_fpi['a'], 'b-', linewidth=1.5)
    axes[1].plot(result_ngs['t'], result_ngs['a'], 'r--', linewidth=1.5)
    axes[1].set_ylabel('Anti-noise a(t)')
    
    axes[2].plot(result_fpi['t'], result_fpi['e'], 'b-', linewidth=1.5)
    axes[2].plot(result_ngs['t'], result_ngs['e'], 'r--', linewidth=1.5)
    axes[2].set_ylabel('Error e(t)')
    axes[2].set_xlabel('Time (s)')
    
    fig.tight_layout()
    fig.savefig(f"{config.OUTPUT_FIGURES_DIR}/midpoint_fpi_vs_ngs.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("\nDone. Plot saved to results/figures/midpoint_fpi_vs_ngs.png")


if __name__ == '__main__':
    main()
