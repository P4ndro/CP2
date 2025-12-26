"""Visualization functions for ANC simulation results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os

import config


def setup_style():
    plt.rcParams.update({
        'figure.figsize': (10, 6), 'axes.grid': True, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10
    })


def save_figure(fig, filename: str, output_dir: str = None):
    if output_dir is None:
        output_dir = config.OUTPUT_FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{filename}", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_solution(result: Dict, title: str = None, save_path: str = None):
    setup_style()
    t, p, a, e = result['t'], result['p'], result['a'], result['e']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(t, p, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Noise p(t)')
    axes[0].set_title(title or f"{result['method']} + {result['solver']}")
    
    axes[1].plot(t, a, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Anti-noise a(t)')
    
    axes[2].plot(t, e, 'r-', linewidth=1.5)
    axes[2].set_ylabel('Error e(t)')
    axes[2].set_xlabel('Time (s)')
    
    fig.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_all_methods_comparison(all_results: Dict, step_sizes: List[float],
                                 h_plot: float = None, save_path: str = None):
    if h_plot is None:
        h_plot = min(step_sizes)
    
    h_idx = step_sizes.index(h_plot)
    setup_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {'Euler+FPI': 'blue', 'Euler+NGS': 'red', 
              'Midpoint+FPI': 'green', 'Midpoint+NGS': 'purple'}
    
    for name, results in all_results.items():
        result = results[h_idx]
        axes[0].plot(result['t'], result['p'], label=name, color=colors.get(name), linewidth=1)
        axes[1].plot(result['t'], result['a'], color=colors.get(name), linewidth=1)
        axes[2].plot(result['t'], result['e'], color=colors.get(name), linewidth=1)
    
    axes[0].set_ylabel('Noise p(t)')
    axes[0].set_title(f'Method Comparison (h = {h_plot})')
    axes[0].legend()
    axes[1].set_ylabel('Anti-noise a(t)')
    axes[2].set_ylabel('Error e(t)')
    axes[2].set_xlabel('Time (s)')
    
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_convergence_order(all_results: Dict, step_sizes: List[float], save_path: str = None):
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = {'Euler+FPI': 'o', 'Euler+NGS': 's', 'Midpoint+FPI': '^', 'Midpoint+NGS': 'd'}
    colors = {'Euler+FPI': 'blue', 'Euler+NGS': 'red', 'Midpoint+FPI': 'green', 'Midpoint+NGS': 'purple'}
    
    for name, results in all_results.items():
        h_vals = [r['h'] for r in results if r.get('status') == 'success']
        errors = [r['error'] for r in results if r.get('status') == 'success']
        if h_vals:
            ax.loglog(h_vals, errors, marker=markers.get(name, 'o'),
                     color=colors.get(name), linewidth=2, label=name)
    
    hs = np.array(step_sizes)
    ax.loglog(hs, 0.5 * hs, 'k--', linewidth=1, label='O(h)')
    ax.loglog(hs, 0.5 * hs**2, 'k:', linewidth=1, label='O(h^2)')
    
    ax.set_xlabel('Step size h')
    ax.set_ylabel('Error (vs reference)')
    ax.set_title('Convergence Order')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_performance_comparison(all_results: Dict, step_sizes: List[float], save_path: str = None):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'Euler+FPI': 'blue', 'Euler+NGS': 'red', 'Midpoint+FPI': 'green', 'Midpoint+NGS': 'purple'}
    
    for name, results in all_results.items():
        runtimes = [r['runtime'] * 1000 for r in results]
        avg_iters = [r['avg_iterations'] for r in results]
        axes[0].plot(step_sizes, runtimes, 'o-', color=colors.get(name), label=name)
        axes[1].plot(step_sizes, avg_iters, 'o-', color=colors.get(name), label=name)
    
    axes[0].set_xlabel('Step size h')
    axes[0].set_ylabel('Runtime (ms)')
    axes[0].set_title('Runtime vs Step Size')
    axes[0].legend()
    
    axes[1].set_xlabel('Step size h')
    axes[1].set_ylabel('Avg Iterations per Step')
    axes[1].set_title('Solver Iterations vs Step Size')
    axes[1].legend()
    
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_solver_comparison(all_results: Dict, step_sizes: List[float], save_path: str = None):
    setup_style()
    h_idx = len(step_sizes) // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(all_results.keys())
    avg_iters = [all_results[m][h_idx]['avg_iterations'] for m in methods]
    max_iters = [all_results[m][h_idx]['max_iterations'] for m in methods]
    
    x = range(len(methods))
    colors = ['blue', 'red', 'green', 'purple']
    
    axes[0].bar(x, avg_iters, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].set_ylabel('Average Iterations')
    axes[0].set_title(f'Solver Iterations (h = {step_sizes[h_idx]})')
    
    axes[1].bar(x, max_iters, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].set_ylabel('Maximum Iterations')
    axes[1].set_title('Maximum Iterations per Step')
    
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def generate_all_plots(all_results: Dict, ref_result: Dict, step_sizes: List[float]):
    print("\nGenerating plots...")
    os.makedirs(config.OUTPUT_FIGURES_DIR, exist_ok=True)
    
    h_small = min(step_sizes)
    h_idx = step_sizes.index(h_small)
    
    for name, results in all_results.items():
        fname = name.lower().replace('+', '_') + '_solution.png'
        plot_solution(results[h_idx], save_path=fname)
    
    plot_all_methods_comparison(all_results, step_sizes, h_small, 'all_methods_solution.png')
    plot_convergence_order(all_results, step_sizes, 'convergence_order.png')
    plot_performance_comparison(all_results, step_sizes, 'performance_comparison.png')
    plot_solver_comparison(all_results, step_sizes, 'solver_comparison.png')
    
    print(f"Plots saved to {config.OUTPUT_FIGURES_DIR}/")
