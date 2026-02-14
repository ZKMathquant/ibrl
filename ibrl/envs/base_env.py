"""Base environment interface."""

from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """Abstract base class for RL environments."""

    @abstractmethod
    def reset(self):
        """Reset environment and return initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).
        
        Returns:
            next_state: Next state
            reward: Scalar reward
            done: Whether episode is complete
            info: Additional information dict
        """
        pass
