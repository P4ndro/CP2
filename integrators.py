"""Time integrators: Implicit Euler and Implicit Midpoint."""

import numpy as np
import time
from typing import Dict, Tuple, Callable
from models import ANCParameters
from solvers import get_solver


def solve_ode(y0: np.ndarray, t_span: Tuple[float, float], h: float,
              params: ANCParameters, noise_func: Callable,
              method: str = 'euler', solver: str = 'FPI',
              tol: float = 1e-8, max_iter: int = 100) -> Dict:
    """Solve ANC ODE system using specified method and solver."""
    
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h/2, h)
    n_steps = len(t_values) - 1
    
    y_values = np.zeros((len(t_values), 3))
    y_values[0] = y0
    
    iterations_per_step = []
    converged_steps = []
    
    solver_func = get_solver(method, solver)
    use_t_new = (method == 'euler')
    
    start_time = time.perf_counter()
    
    for k in range(n_steps):
        y_old = y_values[k]
        t_arg = t_values[k + 1] if use_t_new else t_values[k]
        
        y_new, iters, converged = solver_func(y_old, t_arg, h, params, 
                                               noise_func, tol, max_iter)
        
        y_values[k + 1] = y_new
        iterations_per_step.append(iters)
        converged_steps.append(converged)
    
    elapsed_time = time.perf_counter() - start_time
    iterations_arr = np.array(iterations_per_step)
    
    method_names = {'euler': 'Implicit Euler', 'midpoint': 'Implicit Midpoint'}
    
    return {
        't': t_values,
        'y': y_values,
        'p': y_values[:, 0],
        'a': y_values[:, 1],
        'e': y_values[:, 2],
        'iterations': iterations_arr,
        'avg_iterations': np.mean(iterations_arr),
        'max_iterations': np.max(iterations_arr),
        'min_iterations': np.min(iterations_arr),
        'convergence_rate': np.mean(converged_steps),
        'runtime': elapsed_time,
        'n_steps': n_steps,
        'h': h,
        'solver': solver,
        'method': method_names.get(method, method)
    }


def solve_implicit_euler(y0, t_span, h, params, noise_func, solver='FPI', **kwargs):
    return solve_ode(y0, t_span, h, params, noise_func, 'euler', solver, **kwargs)


def solve_implicit_midpoint(y0, t_span, h, params, noise_func, solver='FPI', **kwargs):
    return solve_ode(y0, t_span, h, params, noise_func, 'midpoint', solver, **kwargs)
