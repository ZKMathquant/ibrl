"""Tests for Twin Prisoner's Dilemma environment."""

from ibrl.envs import TwinPDEnv
from ibrl.predictors import LogicalPredictor


def test_twin_pd_mutual_cooperation():
    predictor = LogicalPredictor(theta=1.0, seed=42)
    env = TwinPDEnv(predictor, seed=42)
    env.reset()
    
    # Both cooperate
    _, reward, _, info = env.step(action=0, greedy_action=0)
    assert reward == 3
    assert info["twin_action"] == 0


def test_twin_pd_mutual_defection():
    predictor = LogicalPredictor(theta=1.0, seed=42)
    env = TwinPDEnv(predictor, seed=42)
    env.reset()
    
    # Both defect
    _, reward, _, info = env.step(action=1, greedy_action=1)
    assert reward == 1
    assert info["twin_action"] == 1


def test_twin_pd_policy_dependence():
    predictor = LogicalPredictor(theta=1.0, seed=42)
    env = TwinPDEnv(predictor, seed=42)
    
    # Greedy cooperates, twin cooperates
    env.reset()
    _, reward1, _, _ = env.step(action=1, greedy_action=0)
    assert reward1 == 5  # Defect against cooperator
    
    # Greedy defects, twin defects
    env.reset()
    _, reward2, _, _ = env.step(action=1, greedy_action=1)
    assert reward2 == 1  # Defect against defector
