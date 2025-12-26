"""Configuration parameters for ANC simulation."""

# Time domain
T_START = 0.0
T_END = 5.0

# Step sizes to test
STEP_SIZES = [0.1, 0.05, 0.02, 0.01, 0.005]
H_REFERENCE = 0.0005

# Solver settings
TOL_FPI = 1e-8
TOL_NGS = 1e-8
MAX_ITER_FPI = 100
MAX_ITER_NGS = 100

# Model parameters (ANC Headphones)
ALPHA = 2.0      # Noise decay rate
BETA = 50.0      # Actuator response speed
GAMMA = 100.0    # Feedback gain (stiffness)
DELTA = 1.0      # Error damping

# Noise settings
NOISE_ONSET = 0.5
NOISE_AMPLITUDE = 1.0
SINE_FREQ = 5.0
SINE_AMPLITUDE = 0.3

# Output paths
OUTPUT_DATA_DIR = 'results/data'
OUTPUT_FIGURES_DIR = 'results/figures'

# Divergence threshold
DIVERGENCE_THRESHOLD = 1e6
