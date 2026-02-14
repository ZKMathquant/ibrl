"""Newcomb's Problem environment with policy-dependent predictor."""

import numpy as np
from .base_env import BaseEnv


class NewcombEnv(BaseEnv):
    """
    Newcomb's Problem.
    
    Two boxes:
    - Box A (small): Always contains $1,000
    - Box B (large): Contains $1,000,000 if predictor predicts one-boxing
    
    Actions:
    - 0: Take only Box B (one-box)
    - 1: Take both boxes (two-box)
    
    Key: Predictor inspects agent's policy before boxes are filled.
    """

    def __init__(self, predictor, million=1_000_000, small=1_000, seed=None):
        """
        Args:
            predictor: LogicalPredictor instance
            million: Large box reward
            small: Small box reward
            seed: Random seed
        """
        self.predictor = predictor
        self.million = million
        self.small = small
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Return initial state."""
        return 0

    def step(self, action, greedy_action=None):
        """
        Execute action after predictor fills boxes.
        
        Args:
            action: Actual action taken (may differ from greedy due to exploration)
            greedy_action: Agent's greedy action (what predictor sees)
        
        Returns:
            state: Always 0
            reward: Total reward from boxes taken
            done: Always True
            info: Dict with predictor correctness
        """
        if greedy_action is None:
            greedy_action = action
        
        # Predictor predicts based on greedy policy
        predicted_action = self.predictor.predict(greedy_action)
        
        # Box B is full if predictor predicted one-boxing
        box_b_full = (predicted_action == 0)
        
        # Compute reward based on actual action
        if action == 0:  # one-box
            reward = self.million if box_b_full else 0
        else:  # two-box
            reward = self.small + (self.million if box_b_full else 0)
        
        # Check if predictor was correct
        predictor_correct = (predicted_action == greedy_action)
        
        info = {
            "predictor_correct": predictor_correct,
            "box_b_full": box_b_full,
            "predicted_action": predicted_action
        }
        
        return 0, float(reward), True, info
