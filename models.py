"""
ANC Model: Active Noise-Cancelling Headphones ODE System

dp/dt = -alpha*p + n(t)
da/dt = -beta*a + gamma*tanh(e)
de/dt = p - a - delta*e
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable
import config


@dataclass
class ANCParameters:
    alpha: float = config.ALPHA
    beta: float = config.BETA
    gamma: float = config.GAMMA
    delta: float = config.DELTA


def noise_step(t: float) -> float:
    return config.NOISE_AMPLITUDE if t >= config.NOISE_ONSET else 0.0


def noise_sine(t: float, freq: float = config.SINE_FREQ) -> float:
    return np.sin(2 * np.pi * freq * t)


def noise_combined(t: float) -> float:
    step = config.NOISE_AMPLITUDE if t >= config.NOISE_ONSET else 0.0
    sine = config.SINE_AMPLITUDE * np.sin(2 * np.pi * config.SINE_FREQ * t)
    return step + sine


def ode_rhs(t: float, y: np.ndarray, params: ANCParameters, 
            noise_func: Callable) -> np.ndarray:
    p, a, e = y
    n = noise_func(t)
    
    dp = -params.alpha * p + n
    da = -params.beta * a + params.gamma * np.tanh(e)
    de = p - a - params.delta * e
    
    return np.array([dp, da, de])
