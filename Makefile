# Makefile for rbyrct-gtc-2025

PYTHON := python

.PHONY: help bench demo test lint format clean

help:
	@echo "Available targets:"
	@echo "  make bench    - run benchmark grid and save results in assets/"
	@echo "  make demo     - launch Streamlit demo (demo/app.py)"
	@echo "  make test     - run pytest"
	@echo "  make lint     - run flake8 linting"
	@echo "  make format   - run black code formatter"
	@echo "  make clean    - remove build/test artifacts"

bench:
	$(PYTHON) examples/bench_grid.py

demo:
	streamlit run demo/app.py

test:
	pytest -q

lint:
	flake8 rbyrct_core demo examples tests

format:
	black rbyrct_core demo examples tests

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
	rm -f assets/bench_grid.csv assets/bench_table.md assets/runtime_vs_size.png

