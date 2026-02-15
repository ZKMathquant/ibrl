#!/bin/bash

echo "=========================================="
echo "IBRL Framework - Complete Evaluation"
echo "=========================================="
echo ""

# Run tests
echo "Running tests..."
pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed. Exiting."
    exit 1
fi
echo ""

# Run bandit experiment
echo "Running bandit experiment..."
python -m ibrl.experiments.run_bandit
echo ""

# Run Newcomb experiment
echo "Running Newcomb experiment..."
python -m ibrl.experiments.run_newcomb
echo ""

# Run Twin PD experiment
echo "Running Twin PD experiment..."
python -m ibrl.experiments.run_twin_pd
echo ""

# Run misspecified experiment
echo "Running misspecified experiment..."
python -m ibrl.experiments.run_misspecified
echo ""

# Run Wasserstein comparison
echo "Running Wasserstein comparison..."
python -m ibrl.experiments.run_wasserstein
echo ""

# Run comprehensive comparison
echo "Running comprehensive comparison..."
python -m ibrl.experiments.compare_all
echo ""

echo "=========================================="
echo "✓ All experiments complete!"
echo "=========================================="
