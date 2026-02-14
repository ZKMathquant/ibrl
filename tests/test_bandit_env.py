"""Tests for bandit environment."""

import numpy as np
from ibrl.envs import BanditEnv


def test_bandit_reset():
    env = BanditEnv()
    state = env.reset()
    assert state == 0


def test_bandit_step():
    env = BanditEnv(probs=(1.0, 0.0), rewards=(1.0, 1.0), seed=42)
    env.reset()
    
    # Arm 0 always succeeds
    _, reward, done, _ = env.step(0)
    assert reward == 1.0
    assert done is True
    
    # Arm 1 always fails
    _, reward, done, _ = env.step(1)
    assert reward == 0.0


def test_bandit_stochastic():
    env = BanditEnv(probs=(0.7, 0.5), seed=42)
    rewards = []
    
    for _ in range(100):
        env.reset()
        _, reward, _, _ = env.step(0)
        rewards.append(reward)
    
    # Should be approximately 70% success rate
    success_rate = np.mean(rewards)
    assert 0.6 < success_rate < 0.8
