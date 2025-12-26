"""
ANC Headphones ODE Model - Full Comparison
Compares Implicit Euler vs Implicit Midpoint with FPI and NGS solvers.
"""

import os
import sys

import config
from models import ANCParameters
from experiments import run_all_experiments, generate_comparison_table, save_results_json, print_summary
from plots import generate_all_plots


def main():
    os.makedirs(config.OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_FIGURES_DIR, exist_ok=True)
    
    params = ANCParameters()
    t_span = (config.T_START, config.T_END)
    
    all_results, ref_result, step_sizes = run_all_experiments(params, t_span)
    
    generate_comparison_table(all_results, step_sizes, config.OUTPUT_DATA_DIR)
    save_results_json(all_results, ref_result, params, step_sizes, config.OUTPUT_DATA_DIR)
    generate_all_plots(all_results, ref_result, step_sizes)
    print_summary(all_results, step_sizes)
    
    print("\n" + "=" * 70)
    print("DONE - All results saved to results/ folder")
    print("=" * 70)


if __name__ == '__main__':
    main()
