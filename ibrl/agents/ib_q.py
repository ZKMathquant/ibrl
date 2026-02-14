"""Infrabayesian Q-learning agent with credal interval beliefs."""

import numpy as np
from .base_agent import BaseAgent


class IBQAgent(BaseAgent):
    """
    Infrabayesian Q-learning agent.
    
    Maintains credal interval over predictor accuracy.
    Uses worst-case expected value for action selection.
    """

    def __init__(self, credal_interval, n_actions=2, alpha=0.1, gamma=0.99, 
                 million=1_000_000, small=1_000, seed=None):
        """
        Args:
            credal_interval: CredalInterval object for belief updating
            n_actions: Number of available actions (typically 2 for Newcomb)
            alpha: Learning rate
            gamma: Discount factor
            million: Large box reward
            small: Small box reward
            seed: Random seed
        """
        self.credal = credal_interval
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.million = million
        self.small = small
        self.q = np.zeros(n_actions)
        self.rng = np.random.default_rng(seed)

    def worst_case_value(self, action):
        """
        Compute worst-case expected value over credal interval.
        
        For Newcomb:
        - Action 0 (one-box): reward = θ * million
        - Action 1 (two-box): reward = small + (1-θ) * million
        
        Returns minimum over [θ_lower, θ_upper].
        """
        theta_lower, theta_upper = self.credal.interval()
        
        def expected_value(theta):
            if action == 0:  # one-box
                return theta * self.million
            else:  # two-box
                return self.small + (1 - theta) * self.million
        
        # For 1D interval, minimum is at one of the endpoints
        val_lower = expected_value(theta_lower)
        val_upper = expected_value(theta_upper)
        
        return min(val_lower, val_upper)

    def greedy_action(self):
        """Return action with highest worst-case value."""
        values = [self.worst_case_value(a) for a in range(self.n_actions)]
        return int(np.argmax(values))

    def select_action(self, state):
        """Select action using worst-case optimization (no exploration)."""
        return self.greedy_action()

    def update(self, state, action, reward, predictor_correct, next_state=None, done=True):
        """
        Update Q-values and credal interval.
        
        Args:
            state: Current state
            action: Action taken
            reward: Observed reward
            predictor_correct: Whether predictor was correct (for credal update)
            next_state: Next state
            done: Episode termination flag
        """
        # Update credal interval
        self.credal.update(predictor_correct)
        
        # Update Q-value (standard Q-learning)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q)
        
        self.q[action] += self.alpha * (target - self.q[action])

    def reset(self):
        """Reset Q-values (credal interval persists across episodes)."""
        self.q = np.zeros(self.n_actions)
