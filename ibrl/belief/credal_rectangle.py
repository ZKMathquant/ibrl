"""Multi-dimensional credal intervals (rectangular credal sets)."""

import numpy as np


class CredalRectangle:
    """
    N-dimensional credal interval: [θ₁_l, θ₁_u] × [θ₂_l, θ₂_u] × ...
    
    Each dimension updated independently using concentration bounds.
    """

    def __init__(self, lower_bounds, upper_bounds, delta=0.05):
        """
        Args:
            lower_bounds: Initial lower bounds for each dimension
            upper_bounds: Initial upper bounds for each dimension
            delta: Confidence parameter
        """
        self.initial_lower = np.array(lower_bounds, dtype=float)
        self.initial_upper = np.array(upper_bounds, dtype=float)
        self.lower = self.initial_lower.copy()
        self.upper = self.initial_upper.copy()
        self.delta = delta
        
        self.n_dims = len(self.lower)
        self.successes = np.zeros(self.n_dims)
        self.trials = 0

    def update(self, outcomes):
        """
        Update credal rectangle based on observations.
        
        Args:
            outcomes: Array of boolean outcomes for each dimension
        """
        self.trials += 1
        self.successes += np.array(outcomes, dtype=float)
        
        if self.trials == 0:
            return
        
        p_hat = self.successes / self.trials
        
        # Bonferroni correction for multiple dimensions
        eps = np.sqrt(np.log(2 * self.n_dims / self.delta) / (2 * self.trials))
        
        self.lower = np.maximum(0.0, p_hat - eps)
        self.upper = np.minimum(1.0, p_hat + eps)
        
        # Intersect with initial bounds
        self.lower = np.maximum(self.lower, self.initial_lower)
        self.upper = np.minimum(self.upper, self.initial_upper)

    def interval(self):
        """Return current credal rectangle as (lower, upper) arrays."""
        return self.lower.copy(), self.upper.copy()

    def width(self):
        """Return average width across dimensions."""
        return np.mean(self.upper - self.lower)

    def reset(self):
        """Reset to initial bounds."""
        self.lower = self.initial_lower.copy()
        self.upper = self.initial_upper.copy()
        self.successes = np.zeros(self.n_dims)
        self.trials = 0
