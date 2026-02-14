"""Transparent Newcomb's Problem (agent can see box contents)."""

from .newcomb import NewcombEnv


class TransparentNewcombEnv(NewcombEnv):
    """
    Transparent Newcomb's Problem.
    
    Agent can observe whether Box B is full before choosing.
    This tests whether agents can resist the temptation to two-box
    even when they see the million dollars.
    """

    def step(self, action, greedy_action=None):
        """
        Execute action with visible box state.
        
        Returns same as NewcombEnv but info includes box visibility.
        """
        state, reward, done, info = super().step(action, greedy_action)
        info["transparent"] = True
        return state, reward, done, info
