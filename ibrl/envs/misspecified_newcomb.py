"""Misspecified Newcomb environment for robustness testing."""

import numpy as np
from .base_env import BaseEnv


class MisspecifiedNewcombEnv(BaseEnv):
    """
    Newcomb's Problem where true predictor accuracy is outside agent's belief set.
    
    Tests robustness under model misspecification.
    
    Agent believes: θ ∈ [0.8, 0.99]
    True accuracy: θ = 0.75 (outside belief set)
    """

    def __init__(self, true_theta, predictor, million=1_000_000, small=1_000, seed=None):
        """
        Args:
            true_theta: True predictor accuracy (outside agent's belief)
            predictor: LogicalPredictor (agent's model, will be wrong)
            million: Large box reward
            small: Small box reward
            seed: Random seed
        """
        self.true_theta = true_theta
        self.predictor = predictor
        self.million = million
        self.small = small
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Return initial state."""
        return 0

    def step(self, action, greedy_action=None):
        """
        Execute action with misspecified predictor.
        
        The predictor uses its internal theta (e.g., 0.95),
        but we override with true_theta for actual outcome.
        
        Args:
            action: Agent's action
            greedy_action: Agent's greedy action (for predictor)
        
        Returns:
            state, reward, done, info
        """
        if greedy_action is None:
            greedy_action = action
        
        # Agent's predictor thinks it has accuracy self.predictor.theta
        # But true accuracy is self.true_theta
        
        # Use TRUE accuracy for actual prediction
        if self.rng.random() < self.true_theta:
            predicted_action = greedy_action
        else:
            predicted_action = 1 - greedy_action
        
        # Box B is full if predictor predicted one-boxing
        box_b_full = (predicted_action == 0)
        
        # Compute reward
        if action == 0:  # one-box
            reward = self.million if box_b_full else 0
        else:  # two-box
            reward = self.small + (self.million if box_b_full else 0)
        
        # For agent's update: report what agent's model would predict
        # (This is the key: agent gets feedback but model is wrong)
        agent_predicted = self.predictor.predict(greedy_action)
        predictor_correct = (agent_predicted == greedy_action)
        
        info = {
            "predictor_correct": predictor_correct,
            "box_b_full": box_b_full,
            "true_theta": self.true_theta,
            "model_theta": self.predictor.theta,
            "misspecified": True
        }
        
        return 0, float(reward), True, info


class AdversarialNewcombEnv(BaseEnv):
    """
    Adversarial Newcomb where predictor actively tries to minimize agent reward.
    
    Extreme test of robustness.
    """

    def __init__(self, million=1_000_000, small=1_000, seed=None):
        """
        Args:
            million: Large box reward
            small: Small box reward
            seed: Random seed
        """
        self.million = million
        self.small = small
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Return initial state."""
        return 0

    def step(self, action, greedy_action=None):
        """
        Adversarial predictor: always predicts opposite of greedy action.
        
        This is worst-case scenario for the agent.
        """
        if greedy_action is None:
            greedy_action = action
        
        # Adversarial: always predict opposite
        predicted_action = 1 - greedy_action
        
        box_b_full = (predicted_action == 0)
        
        if action == 0:
            reward = self.million if box_b_full else 0
        else:
            reward = self.small + (self.million if box_b_full else 0)
        
        info = {
            "predictor_correct": False,
            "box_b_full": box_b_full,
            "adversarial": True
        }
        
        return 0, float(reward), True, info
