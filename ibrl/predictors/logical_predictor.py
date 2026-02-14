"""Logical predictor that inspects agent policy."""

import numpy as np


class LogicalPredictor:
    """
    Predictor with logical dependence on agent policy.
    
    Inspects agent's greedy action and predicts with accuracy θ.
    This creates policy-dependent transition dynamics.
    """

    def __init__(self, theta, seed=None):
        """
        Args:
            theta: Prediction accuracy (probability of correct prediction)
            seed: Random seed
        """
        self.theta = theta
        self.rng = np.random.default_rng(seed)

    def predict(self, greedy_action):
        """
        Predict agent's action by inspecting policy.
        
        Args:
            greedy_action: Agent's greedy action (0=one-box, 1=two-box)
        
        Returns:
            predicted_action: Prediction with accuracy θ
        """
        if self.rng.random() < self.theta:
            # Correct prediction
            return greedy_action
        else:
            # Incorrect prediction (flip action)
            return 1 - greedy_action
