"""Credal interval belief representation with concentration bounds."""

import math


class CredalInterval:
    """
    Credal interval over predictor accuracy parameter θ.
    
    Maintains interval [θ_lower, θ_upper] using concentration inequalities.
    Shrinks as more data is observed.
    """

    def __init__(self, lower=0.8, upper=0.99, delta=0.05):
        """
        Args:
            lower: Initial lower bound on θ
            upper: Initial upper bound on θ
            delta: Confidence parameter (1-δ confidence)
        """
        self.initial_lower = lower
        self.initial_upper = upper
        self.lower = lower
        self.upper = upper
        self.delta = delta
        
        self.successes = 0
        self.trials = 0

    def update(self, success):
        """
        Update interval based on new observation.
        
        Uses Hoeffding's inequality:
        P(|θ̂ - θ| > ε) ≤ 2exp(-2nε²)
        
        Args:
            success: Whether predictor was correct (boolean)
        """
        self.trials += 1
        if success:
            self.successes += 1
        
        if self.trials == 0:
            return
        
        # Empirical estimate
        p_hat = self.successes / self.trials
        
        # Concentration bound: ε = sqrt(log(2/δ) / (2n))
        epsilon = math.sqrt(math.log(2 / self.delta) / (2 * self.trials))
        
        # Update interval
        self.lower = max(0.0, p_hat - epsilon)
        self.upper = min(1.0, p_hat + epsilon)
        
        # Intersect with initial bounds
        self.lower = max(self.lower, self.initial_lower)
        self.upper = min(self.upper, self.initial_upper)

    def interval(self):
        """
        Return current credal interval.
        
        Returns:
            (lower, upper): Tuple of interval bounds
        """
        return self.lower, self.upper

    def width(self):
        """Return interval width."""
        return self.upper - self.lower

    def reset(self):
        """Reset to initial interval."""
        self.lower = self.initial_lower
        self.upper = self.initial_upper
        self.successes = 0
        self.trials = 0
