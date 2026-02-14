.PHONY: install test clean run-bandit run-newcomb run-twin-pd run-all run-experiments

install:
	pip install -e .

test:
	pytest tests/ -v

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run-bandit:
	python -m ibrl.experiments.run_bandit

run-newcomb:
	python -m ibrl.experiments.run_newcomb

run-twin-pd:
	python -m ibrl.experiments.run_twin_pd

run-experiments: run-bandit run-newcomb run-twin-pd
	@echo "✓ All individual experiments complete"

run-all:
	python -m ibrl.experiments.compare_all

format:
	black ibrl/ tests/ 2>/dev/null || echo "black not installed, skipping"
