# ANC Headphones ODE Model

Modeling Active Noise-Cancelling Headphones using implicit numerical methods.

## Model
```
dp/dt = -α·p + n(t)
da/dt = -β·a + γ·tanh(e)  
de/dt = p - a - δ·e
```

## Two Working Codes

**Code 1:** `implicit_euler.py` - Implicit Euler with FPI vs NGS  
**Code 2:** `implicit_midpoint.py` - Implicit Midpoint with FPI vs NGS

## Usage
```bash
pip install numpy matplotlib
python implicit_euler.py
python implicit_midpoint.py
```

## Output
- Comparison tables printed to console
- Plots saved to `results/figures/`
