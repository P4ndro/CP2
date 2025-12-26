"""
Nonlinear Solvers: Fixed-Point Iteration and Newton-Gauss-Seidel
"""

import numpy as np
from typing import Tuple, Callable
from models import ANCParameters
from config import DIVERGENCE_THRESHOLD


def fpi_euler(y_old: np.ndarray, t_new: float, h: float,
              params: ANCParameters, noise_func: Callable,
              tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, int, bool]:
    """Fixed-Point Iteration for Implicit Euler."""
    p_old, a_old, e_old = y_old
    n_new = noise_func(t_new)
    p, a, e = y_old.copy()
    
    for iteration in range(1, max_iter + 1):
        p_prev, a_prev, e_prev = p, a, e
        
        p_new = p_old + h * (-params.alpha * p_prev + n_new)
        a_new = a_old + h * (-params.beta * a_prev + params.gamma * np.tanh(e_prev))
        e_new = e_old + h * (p_prev - a_prev - params.delta * e_prev)
        
        p, a, e = p_new, a_new, e_new
        
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < tol:
            return np.array([p, a, e]), iteration, True
        
        if abs(p) > DIVERGENCE_THRESHOLD or abs(a) > DIVERGENCE_THRESHOLD or abs(e) > DIVERGENCE_THRESHOLD:
            return np.array([p, a, e]), iteration, False
    
    return np.array([p, a, e]), max_iter, False


def ngs_euler(y_old: np.ndarray, t_new: float, h: float,
              params: ANCParameters, noise_func: Callable,
              tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, int, bool]:
    """Newton-Gauss-Seidel for Implicit Euler."""
    p_old, a_old, e_old = y_old
    n_new = noise_func(t_new)
    p, a, e = y_old.copy()
    
    for iteration in range(1, max_iter + 1):
        p_prev, a_prev, e_prev = p, a, e
        
        G1 = p - p_old - h * (-params.alpha * p + n_new)
        p = p - G1 / (1 + h * params.alpha)
        
        G2 = a - a_old - h * (-params.beta * a + params.gamma * np.tanh(e))
        a = a - G2 / (1 + h * params.beta)
        
        G3 = e - e_old - h * (p - a - params.delta * e)
        e = e - G3 / (1 + h * params.delta)
        
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < tol:
            return np.array([p, a, e]), iteration, True
    
    return np.array([p, a, e]), max_iter, False


def fpi_midpoint(y_old: np.ndarray, t_old: float, h: float,
                 params: ANCParameters, noise_func: Callable,
                 tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, int, bool]:
    """Fixed-Point Iteration for Implicit Midpoint."""
    p_old, a_old, e_old = y_old
    t_mid = t_old + h / 2
    n_mid = noise_func(t_mid)
    p, a, e = y_old.copy()
    
    for iteration in range(1, max_iter + 1):
        p_prev, a_prev, e_prev = p, a, e
        
        p_mid = (p_old + p_prev) / 2
        a_mid = (a_old + a_prev) / 2
        e_mid = (e_old + e_prev) / 2
        
        p_new = p_old + h * (-params.alpha * p_mid + n_mid)
        a_new = a_old + h * (-params.beta * a_mid + params.gamma * np.tanh(e_mid))
        e_new = e_old + h * (p_mid - a_mid - params.delta * e_mid)
        
        p, a, e = p_new, a_new, e_new
        
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < tol:
            return np.array([p, a, e]), iteration, True
        
        if abs(p) > DIVERGENCE_THRESHOLD or abs(a) > DIVERGENCE_THRESHOLD or abs(e) > DIVERGENCE_THRESHOLD:
            return np.array([p, a, e]), iteration, False
    
    return np.array([p, a, e]), max_iter, False


def ngs_midpoint(y_old: np.ndarray, t_old: float, h: float,
                 params: ANCParameters, noise_func: Callable,
                 tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, int, bool]:
    """Newton-Gauss-Seidel for Implicit Midpoint."""
    p_old, a_old, e_old = y_old
    t_mid = t_old + h / 2
    n_mid = noise_func(t_mid)
    p, a, e = y_old.copy()
    
    for iteration in range(1, max_iter + 1):
        p_prev, a_prev, e_prev = p, a, e
        
        p_mid = (p_old + p) / 2
        G1 = p - p_old - h * (-params.alpha * p_mid + n_mid)
        p = p - G1 / (1 + h * params.alpha / 2)
        
        a_mid = (a_old + a) / 2
        e_mid = (e_old + e) / 2
        G2 = a - a_old - h * (-params.beta * a_mid + params.gamma * np.tanh(e_mid))
        a = a - G2 / (1 + h * params.beta / 2)
        
        p_mid = (p_old + p) / 2
        a_mid = (a_old + a) / 2
        e_mid = (e_old + e) / 2
        G3 = e - e_old - h * (p_mid - a_mid - params.delta * e_mid)
        e = e - G3 / (1 + h * params.delta / 2)
        
        if max(abs(p - p_prev), abs(a - a_prev), abs(e - e_prev)) < tol:
            return np.array([p, a, e]), iteration, True
    
    return np.array([p, a, e]), max_iter, False


SOLVERS = {
    'euler': {'FPI': fpi_euler, 'NGS': ngs_euler},
    'midpoint': {'FPI': fpi_midpoint, 'NGS': ngs_midpoint}
}


def get_solver(method: str, solver: str) -> Callable:
    return SOLVERS[method][solver]
