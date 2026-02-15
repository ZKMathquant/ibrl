"""Tests for Wasserstein ball belief."""

import numpy as np
from ibrl.belief import WassersteinBall


def test_wasserstein_initialization():
    center = [0.5, 0.5]
    ball = WassersteinBall(center, radius=0.1)
    
    assert ball.n_outcomes == 2
    assert np.allclose(ball.center, [0.5, 0.5])
    assert ball.radius == 0.1


def test_wasserstein_update():
    ball = WassersteinBall([0.5, 0.5], radius=0.5)
    
    initial_radius = ball.radius
    
    for _ in range(100):
        ball.update(0)
    
    # Radius should shrink
    assert ball.radius < initial_radius
    
    # Center should shift toward outcome 0
    assert ball.center[0] > 0.9


def test_worst_case_expectation():
    ball = WassersteinBall([0.5, 0.5], radius=0.1)
    values = np.array([1.0, 0.0])
    
    worst = ball.worst_case_expectation(values)
    best = ball.best_case_expectation(values)
    center = ball.center @ values
    
    # Worst-case should be below center
    assert worst < center
    # Best-case should be above center
    assert best > center
