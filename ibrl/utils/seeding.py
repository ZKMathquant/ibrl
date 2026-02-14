"""Random seed utilities."""

import numpy as np
import random


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
