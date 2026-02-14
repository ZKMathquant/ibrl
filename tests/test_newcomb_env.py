"""Tests for Newcomb environment."""

from ibrl.envs import NewcombEnv
from ibrl.predictors import LogicalPredictor


def test_newcomb_one_box_perfect_predictor():
    predictor = LogicalPredictor(theta=1.0, seed=42)
    env = NewcombEnv(predictor, seed=42)
    env.reset()
    
    # One-boxing with perfect predictor
    _, reward, _, info = env.step(action=0, greedy_action=0)
    assert reward == 1_000_000
    assert info["predictor_correct"] is True


def test_newcomb_two_box_perfect_predictor():
    predictor = LogicalPredictor(theta=1.0, seed=42)
    env = NewcombEnv(predictor, seed=42)
    env.reset()
    
    # Two-boxing with perfect predictor
    _, reward, _, info = env.step(action=1, greedy_action=1)
    assert reward == 1_000
    assert info["predictor_correct"] is True


def test_newcomb_policy_dependence():
    predictor = LogicalPredictor(theta=1.0, seed=42)
    env = NewcombEnv(predictor, seed=42)
    
    # Predictor sees greedy action, not actual action
    env.reset()
    _, reward1, _, _ = env.step(action=1, greedy_action=0)  # Greedy one-boxes
    assert reward1 == 1_001_000  # Box full + small box
    
    env.reset()
    _, reward2, _, _ = env.step(action=1, greedy_action=1)  # Greedy two-boxes
    assert reward2 == 1_000  # Only small box
