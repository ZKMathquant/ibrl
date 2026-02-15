"""Tests for misspecified environments."""

from ibrl.envs import MisspecifiedNewcombEnv, AdversarialNewcombEnv
from ibrl.predictors import LogicalPredictor


def test_misspecified_newcomb():
    # Agent thinks θ=0.95, but true θ=0.75
    predictor = LogicalPredictor(theta=0.95, seed=42)
    env = MisspecifiedNewcombEnv(true_theta=0.75, predictor=predictor, seed=42)
    
    env.reset()
    _, reward, _, info = env.step(action=0, greedy_action=0)
    
    assert info["misspecified"] is True
    assert info["true_theta"] == 0.75
    assert info["model_theta"] == 0.95


def test_adversarial_newcomb():
    env = AdversarialNewcombEnv(seed=42)
    env.reset()
    
    # Adversarial predictor always predicts opposite
    _, reward, _, info = env.step(action=0, greedy_action=0)
    
    assert info["adversarial"] is True
    assert info["predictor_correct"] is False
