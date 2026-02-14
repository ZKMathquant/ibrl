"""Tests for IB agent."""

import numpy as np
from ibrl.agents import IBQAgent
from ibrl.belief import CredalInterval


def test_ib_worst_case_value():
    credal = CredalInterval(lower=0.9, upper=0.95)
    agent = IBQAgent(credal, n_actions=2)
    
    # One-box worst case
    val_one = agent.worst_case_value(0)
    
    # Two-box worst case
    val_two = agent.worst_case_value(1)
    
    # One-box should be better with high theta
    assert val_one > val_two


def test_ib_greedy_action():
    credal = CredalInterval(lower=0.9, upper=0.95)
    agent = IBQAgent(credal, n_actions=2)
    
    # Should choose one-box
    action = agent.greedy_action()
    assert action == 0


def test_ib_credal_update():
    credal = CredalInterval(lower=0.8, upper=0.99, delta=0.05)
    agent = IBQAgent(credal, n_actions=2)
    
    initial_width = agent.credal.width()
    
    # Update with many successes
    for _ in range(100):
        agent.update(0, 0, 1_000_000, predictor_correct=True)
    
    final_width = agent.credal.width()
    
    # Interval should shrink
    assert final_width < initial_width
