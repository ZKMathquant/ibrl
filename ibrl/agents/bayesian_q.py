"""Bayesian Q-learning agent with posterior belief updating."""

import numpy as np
from .base_agent import BaseAgent


class BayesianQAgent(BaseAgent):
    """
    Bayesian Q-learning agent.
    
    Maintains posterior distribution over reward probabilities.
    Uses Thompson sampling for exploration.
    """

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, seed=None):
        """
        Args:
            n_actions: Number of available actions
            alpha: Learning rate (for Q-values)
            gamma: Discount factor
            seed: Random seed
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros(n_actions)
        
        # Beta distribution parameters for each action
        self.alpha_params = np.ones(n_actions)
        self.beta_params = np.ones(n_actions)
        
        self.rng = np.random.default_rng(seed)

    def greedy_action(self):
        """Return action with highest expected Q-value."""
        return int(np.argmax(self.q))

    def select_action(self, state):
        """Thompson sampling: sample from posterior and choose best."""
        sampled_values = self.rng.beta(self.alpha_params, self.beta_params)
        return int(np.argmax(sampled_values))

    def update(self, state, action, reward, next_state=None, done=True):
        """Update Q-values and posterior beliefs."""
        # Update Q-value
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q)
        
        self.q[action] += self.alpha * (target - self.q[action])
        
        # Update Beta posterior
        if reward > 0:
            self.alpha_params[action] += 1
        else:
            self.beta_params[action] += 1

    def reset(self):
        """Reset Q-values and beliefs."""
        self.q = np.zeros(self.n_actions)
        self.alpha_params = np.ones(self.n_actions)
        self.beta_params = np.ones(self.n_actions)
