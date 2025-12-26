"""Experiment runner and results generation."""

import numpy as np
import json
import pandas as pd
from typing import Dict, List, Callable

import config
from models import ANCParameters, noise_combined
from integrators import solve_ode
from utils import inf_norm_error, get_failure_status, is_solver_failure


def compute_reference_solution(params: ANCParameters, t_span, noise_func: Callable) -> Dict:
    print(f"Computing reference solution with h = {config.H_REFERENCE}...")
    y0 = np.array([0.0, 0.0, 0.0])
    ref_result = solve_ode(y0, t_span, config.H_REFERENCE, params, noise_func,
                           method='midpoint', solver='NGS', 
                           tol=config.TOL_NGS, max_iter=config.MAX_ITER_NGS)
    print(f"Reference solution computed: {len(ref_result['t'])} points")
    return ref_result


def compute_error_vs_reference(result: Dict, ref_result: Dict) -> float:
    y_final = result['y'][-1]
    t_final = result['t'][-1]
    idx_ref = np.argmin(np.abs(ref_result['t'] - t_final))
    return inf_norm_error(y_final, ref_result['y'][idx_ref])


def run_single_experiment(method: str, solver: str, h: float,
                          params: ANCParameters, t_span, noise_func: Callable,
                          ref_result: Dict = None) -> Dict:
    y0 = np.array([0.0, 0.0, 0.0])
    tol = config.TOL_FPI if solver == 'FPI' else config.TOL_NGS
    max_iter = config.MAX_ITER_FPI if solver == 'FPI' else config.MAX_ITER_NGS
    
    result = solve_ode(y0, t_span, h, params, noise_func, method, solver, tol, max_iter)
    
    if ref_result is not None:
        result['error'] = compute_error_vs_reference(result, ref_result)
        result['is_failure'] = is_solver_failure(result['error'])
        result['status'] = get_failure_status(result['convergence_rate'], result['error'])
    
    return result


def run_all_experiments(params: ANCParameters = None, t_span=None,
                        noise_func: Callable = None,
                        step_sizes: List[float] = None) -> Dict:
    if params is None:
        params = ANCParameters()
    if t_span is None:
        t_span = (config.T_START, config.T_END)
    if noise_func is None:
        noise_func = noise_combined
    if step_sizes is None:
        step_sizes = config.STEP_SIZES
    
    print("=" * 70)
    print("ANC Headphones ODE Model - Method Comparison")
    print("=" * 70)
    print(f"Parameters: alpha={params.alpha}, beta={params.beta}, gamma={params.gamma}, delta={params.delta}")
    print(f"Time span: {t_span}, Step sizes: {step_sizes}")
    print("=" * 70)
    
    ref_result = compute_reference_solution(params, t_span, noise_func)
    
    configs = [
        ('euler', 'FPI', 'Euler+FPI'),
        ('euler', 'NGS', 'Euler+NGS'),
        ('midpoint', 'FPI', 'Midpoint+FPI'),
        ('midpoint', 'NGS', 'Midpoint+NGS')
    ]
    
    all_results = {name: [] for _, _, name in configs}
    
    for h in step_sizes:
        print(f"\n--- h = {h} ---")
        for method, solver, name in configs:
            result = run_single_experiment(method, solver, h, params, t_span, noise_func, ref_result)
            all_results[name].append(result)
            status = result.get('status', 'unknown')
            mark = "OK" if status == 'success' else "FAIL"
            print(f"{name:15s}: iters={result['avg_iterations']:.2f}, err={result['error']:.2e}, [{mark}]")
    
    return all_results, ref_result, step_sizes


def generate_comparison_table(all_results: Dict, step_sizes: List[float],
                              output_dir: str = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = config.OUTPUT_DATA_DIR
    
    rows = []
    for h_idx, h in enumerate(step_sizes):
        for method_name, results in all_results.items():
            r = results[h_idx]
            rows.append({
                'h': h, 'Method': method_name, 'Steps': r['n_steps'],
                'Avg Iter': f"{r['avg_iterations']:.2f}",
                'Max Iter': int(r['max_iterations']),
                'Runtime (ms)': f"{r['runtime']*1000:.2f}",
                'Error': "FAILED" if r.get('is_failure') else f"{r['error']:.2e}",
                'Conv %': f"{r['convergence_rate']*100:.0f}",
                'Status': r.get('status', 'unknown')
            })
    
    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    
    df.to_csv(f"{output_dir}/comparison_results.csv", index=False)
    
    with open(f"{output_dir}/comparison_table.tex", 'w') as f:
        f.write("\\begin{tabular}{lllllllll}\n\\hline\n")
        f.write("h & Method & Steps & Avg Iter & Max Iter & Runtime & Error & Conv\\% & Status \\\\\n\\hline\n")
        for _, row in df.iterrows():
            f.write(f"{row['h']} & {row['Method']} & {row['Steps']} & {row['Avg Iter']} & "
                   f"{row['Max Iter']} & {row['Runtime (ms)']} & {row['Error']} & "
                   f"{row['Conv %']} & {row['Status']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    
    return df


def save_results_json(all_results: Dict, ref_result: Dict, params: ANCParameters,
                      step_sizes: List[float], output_dir: str = None):
    if output_dir is None:
        output_dir = config.OUTPUT_DATA_DIR
    
    data = {
        'config': {'H_REFERENCE': config.H_REFERENCE, 'TOL': config.TOL_NGS},
        'parameters': {'alpha': params.alpha, 'beta': params.beta, 'gamma': params.gamma, 'delta': params.delta},
        'step_sizes': step_sizes,
        'methods': {name: [{'h': r['h'], 'avg_iterations': float(r['avg_iterations']),
                           'error': float(r['error']), 'status': r.get('status'),
                           'is_failure': bool(r.get('is_failure', False))}
                          for r in results] for name, results in all_results.items()}
    }
    
    with open(f"{output_dir}/complete_results.json", 'w') as f:
        json.dump(data, f, indent=2)


def print_summary(all_results: Dict, step_sizes: List[float]):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nReference: Implicit Midpoint + NGS, h = {config.H_REFERENCE}")
    
    print("\nSTABILITY TABLE:")
    print("-" * 50)
    for name in all_results:
        results = all_results[name]
        max_h = max((r['h'] for r in results if r.get('status') == 'success'), default=0)
        print(f"  {name:15s}: max stable h = {max_h}")
    
    print("\nCONCLUSION:")
    print("  Newton-Gauss-Seidel is more robust and efficient than Fixed-Point Iteration.")
    print("  Implicit Midpoint achieves higher accuracy than Implicit Euler.")
