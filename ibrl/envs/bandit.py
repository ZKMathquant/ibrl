"""Classical multi-armed bandit environment."""

import numpy as np
from .base_env import BaseEnv


class BanditEnv(BaseEnv):
    """
    Classical multi-armed bandit.
    
    Each arm has independent success probability.
    No policy dependence.
    """

    def __init__(self, probs=(0.7, 0.5), rewards=(1.0, 1.0), seed=None):
        """
        Args:
            probs: Success probability for each arm
            rewards: Reward value for each arm on success
            seed: Random seed
        """
        self.probs = np.array(probs)
        self.rewards = np.array(rewards)
        self.n_actions = len(probs)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Return initial state (stateless bandit)."""
        return 0

    def step(self, action):
        """
        Pull arm and observe reward.
        
        Returns:
            state: Always 0 (stateless)
            reward: Stochastic reward based on arm probabilities
            done: Always True (one-shot)
            info: Empty dict
        """
        if self.rng.random() < self.probs[action]:
            reward = float(self.rewards[action])
        else:
            reward = 0.0
        
        return 0, reward, True, {}
