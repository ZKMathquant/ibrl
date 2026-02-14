"""Base agent interface for all RL agents."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def select_action(self, state):
        """Select action given current state."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update agent's internal state after observing outcome."""
        pass

    def greedy_action(self):
        """Return greedy action (for predictors to inspect)."""
        raise NotImplementedError("Subclass must implement greedy_action")

    def reset(self):
        """Reset agent state (optional)."""
        pass
