"""Twin Prisoner's Dilemma with policy-dependent opponent."""

import numpy as np
from .base_env import BaseEnv


class TwinPDEnv(BaseEnv):
    """
    Twin Prisoner's Dilemma.
    
    Agent plays against a copy of itself (twin).
    Twin's action is predicted based on agent's policy.
    
    Actions:
    - 0: Cooperate
    - 1: Defect
    
    Payoff matrix (agent's reward):
              Twin C  Twin D
    Agent C     3       0
    Agent D     5       1
    
    Key: Twin predictor sees agent's greedy policy.
    """

    def __init__(self, predictor, payoffs=None, seed=None):
        """
        Args:
            predictor: LogicalPredictor for twin's action
            payoffs: 2x2 payoff matrix (default: standard PD)
            seed: Random seed
        """
        self.predictor = predictor
        
        if payoffs is None:
            # Standard PD payoffs
            self.payoffs = np.array([
                [3, 0],  # Agent cooperates
                [5, 1]   # Agent defects
            ])
        else:
            self.payoffs = np.array(payoffs)
        
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Return initial state."""
        return 0

    def step(self, action, greedy_action=None):
        """
        Execute action after twin chooses.
        
        Args:
            action: Agent's actual action (0=cooperate, 1=defect)
            greedy_action: Agent's greedy action (what twin sees)
        
        Returns:
            state: Always 0
            reward: Payoff based on (agent_action, twin_action)
            done: Always True
            info: Dict with twin's action and predictor correctness
        """
        if greedy_action is None:
            greedy_action = action
        
        # Twin predicts agent's greedy action
        twin_action = self.predictor.predict(greedy_action)
        
        # Get reward from payoff matrix
        reward = float(self.payoffs[action, twin_action])
        
        # Check predictor correctness
        predictor_correct = (twin_action == greedy_action)
        
        info = {
            "predictor_correct": predictor_correct,
            "twin_action": twin_action,
            "agent_action": action
        }
        
        return 0, reward, True, info
