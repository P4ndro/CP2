# ANC Headphones ODE Model

Modeling Active Noise-Cancelling Headphones using a nonlinear ODE system with implicit numerical methods.

## Model

The system models noise pressure, anti-noise actuator response, and residual error:

```
dp/dt = -α·p + n(t)
da/dt = -β·a + γ·tanh(e)
de/dt = p - a - δ·e
```

Parameters: α=2, β=50, γ=100, δ=1

## Methods

- **Implicit Euler** (1st order)
- **Implicit Midpoint** (2nd order)

## Solvers

- **Fixed-Point Iteration (FPI)**
- **Newton-Gauss-Seidel (NGS)**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run full comparison:
```bash
python run_comparison.py
```

Run individual methods:
```bash
python run_implicit_euler.py
python run_implicit_midpoint.py
```

## Output

- `results/data/` - CSV and JSON data files
- `results/figures/` - Plots and visualizations

## Files

- `config.py` - All parameters and settings
- `models.py` - ODE system definition
- `solvers.py` - FPI and NGS implementations
- `integrators.py` - Time stepping methods
- `experiments.py` - Experiment runner
- `plots.py` - Visualization functions
