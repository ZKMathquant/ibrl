.PHONY: install test test-wasserstein test-misspecified clean run-bandit run-newcomb run-twin-pd run-misspecified run-wasserstein run-experiments run-all format full

install:
	pip install -e .

test:
	pytest tests/ -v

test-wasserstein:
	pytest tests/test_wasserstein.py -v

test-misspecified:
	pytest tests/test_misspecified.py -v

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	#rm -f ibrl_comparison.png

run-bandit:
	python -m ibrl.experiments.run_bandit

run-newcomb:
	python -m ibrl.experiments.run_newcomb

run-twin-pd:
	python -m ibrl.experiments.run_twin_pd

run-misspecified:
	python -m ibrl.experiments.run_misspecified

run-wasserstein:
	python -m ibrl.experiments.run_wasserstein

run-experiments: run-bandit run-newcomb run-twin-pd run-misspecified run-wasserstein
	@echo "✓ All individual experiments complete"

run-all:
	python -m ibrl.experiments.compare_all

format:
	black ibrl/ tests/ 2>/dev/null || echo "black not installed, skipping"

full: test run-experiments run-all
	@echo ""
	@echo "=========================================="
	@echo "✓ Full validation complete!"
	@echo "=========================================="
