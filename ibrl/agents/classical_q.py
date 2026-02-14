"""Classical Q-learning agent with epsilon-greedy exploration."""

import numpy as np
from .base_agent import BaseAgent


class ClassicalQAgent(BaseAgent):
    """
    Standard Q-learning agent.
    
    Uses point estimates and epsilon-greedy exploration.
    """

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, seed=None):
        """
        Args:
            n_actions: Number of available actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            seed: Random seed
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros(n_actions)
        self.rng = np.random.default_rng(seed)

    def greedy_action(self):
        """Return action with highest Q-value."""
        return int(np.argmax(self.q))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return self.greedy_action()

    def update(self, state, action, reward, next_state=None, done=True):
        """Q-learning update rule."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q)
        
        self.q[action] += self.alpha * (target - self.q[action])

    def reset(self):
        """Reset Q-values."""
        self.q = np.zeros(self.n_actions)
