"""Tests for multi-dimensional credal intervals."""

import numpy as np
from ibrl.belief import CredalRectangle


def test_rectangle_initialization():
    credal = CredalRectangle([0.8, 0.7], [0.99, 0.95])
    lower, upper = credal.interval()
    
    assert len(lower) == 2
    assert len(upper) == 2
    assert np.allclose(lower, [0.8, 0.7])
    assert np.allclose(upper, [0.99, 0.95])


def test_rectangle_update():
    credal = CredalRectangle([0.5, 0.5], [1.0, 1.0], delta=0.05)
    
    initial_width = credal.width()
    
    # Update with observations
    for _ in range(100):
        credal.update([True, True])
    
    final_width = credal.width()
    
    # Width should shrink
    assert final_width < initial_width


def test_rectangle_convergence():
    credal = CredalRectangle([0.0, 0.0], [1.0, 1.0], delta=0.05)
    
    # Simulate 90% and 80% accuracy
    np.random.seed(42)
    for _ in range(500):
        outcomes = [
            np.random.random() < 0.9,
            np.random.random() < 0.8
        ]
        credal.update(outcomes)
    
    lower, upper = credal.interval()
    
    # Should contain true values
    assert lower[0] < 0.9 < upper[0]
    assert lower[1] < 0.8 < upper[1]
    
    # Should be reasonably tight
    assert upper[0] - lower[0] < 0.15
    assert upper[1] - lower[1] < 0.15
