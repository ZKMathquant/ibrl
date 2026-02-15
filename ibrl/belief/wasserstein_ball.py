"""Wasserstein uncertainty ball for distributionally robust RL."""

import numpy as np
from scipy.optimize import linprog


class WassersteinBall:
    """
    Wasserstein ball around empirical distribution.
    
    Represents all distributions within Wasserstein distance ε
    from the empirical estimate.
    """

    def __init__(self, center_dist, radius, delta=0.05):
        """
        Args:
            center_dist: Empirical distribution (numpy array)
            radius: Wasserstein radius ε
            delta: Confidence parameter
        """
        self.center = np.array(center_dist, dtype=float)
        self.radius = radius
        self.delta = delta
        self.n_outcomes = len(center_dist)
        
        # Track observations
        self.counts = np.zeros(self.n_outcomes)
        self.trials = 0

    def update(self, outcome):
        """
        Update empirical distribution and shrink radius.
        
        Args:
            outcome: Index of observed outcome
        """
        self.trials += 1
        self.counts[outcome] += 1
        
        # Update center to empirical distribution
        self.center = self.counts / self.trials
        
        # Shrink radius with concentration bound
        # Wasserstein distance concentrates at rate O(1/sqrt(n))
        self.radius = max(0.01, np.sqrt(np.log(2/self.delta) / (2 * self.trials)))

    def worst_case_expectation(self, values):
        """
        Compute worst-case expectation over Wasserstein ball.
        
        For discrete distributions, this reduces to:
        E_worst[V] = E_center[V] - ε * ||V||_Lip
        
        Args:
            values: Value function (numpy array)
        
        Returns:
            Worst-case expected value
        """
        values = np.array(values, dtype=float)
        
        # Compute Lipschitz constant (max difference)
        lipschitz = np.max(values) - np.min(values)
        
        # Worst-case: center expectation minus radius * Lipschitz
        center_expectation = self.center @ values
        worst_case = center_expectation - self.radius * lipschitz
        
        return worst_case

    def best_case_expectation(self, values):
        """Compute best-case expectation (for completeness)."""
        values = np.array(values, dtype=float)
        lipschitz = np.max(values) - np.min(values)
        center_expectation = self.center @ values
        return center_expectation + self.radius * lipschitz

    def interval(self):
        """Return (center, radius) for compatibility with credal interval."""
        return self.center, self.radius

    def width(self):
        """Return effective uncertainty width."""
        return 2 * self.radius

    def reset(self):
        """Reset to initial state."""
        self.counts = np.zeros(self.n_outcomes)
        self.trials = 0
        self.center = np.ones(self.n_outcomes) / self.n_outcomes
