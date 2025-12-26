"""Utility functions for timing, metrics, and status tracking."""

import numpy as np
import time
from contextlib import contextmanager
from config import DIVERGENCE_THRESHOLD


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def inf_norm_error(y: np.ndarray, y_ref: np.ndarray) -> float:
    return np.max(np.abs(y - y_ref))


def is_solver_failure(error: float) -> bool:
    return error > DIVERGENCE_THRESHOLD


def get_failure_status(convergence_rate: float, error: float) -> str:
    if convergence_rate < 0.5 or error > DIVERGENCE_THRESHOLD:
        return 'failed'
    elif convergence_rate < 1.0:
        return 'partial'
    return 'success'
