"""Tests for convergence properties."""

import numpy as np
from ibrl.belief import CredalInterval


def test_credal_interval_shrinks():
    credal = CredalInterval(lower=0.8, upper=0.99, delta=0.05)
    
    widths = []
    for i in range(200):
        credal.update(True)
        widths.append(credal.width())
    
    # Width should decrease
    assert widths[-1] < widths[0]
    
    # Should converge toward true value (1.0 with all successes)
    lower, upper = credal.interval()
    assert 0.85 < lower < 1.0  
    assert 0.95 < upper <= 1.0


def test_credal_concentration_bound():
    credal = CredalInterval(lower=0.0, upper=1.0, delta=0.05)
    
    # Simulate 95% accuracy
    np.random.seed(42)
    for _ in range(1000):
        success = np.random.random() < 0.95
        credal.update(success)
    
    lower, upper = credal.interval()
    
    # Should contain true value with high probability
    assert lower < 0.95 < upper
    
    # Should be reasonably tight
    assert upper - lower < 0.1
