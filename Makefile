# ROSClaw v1.0 - Development Makefile

PYTHON := python3
PIP := pip
SRC := src/rosclaw
TESTS := tests/

.PHONY: install test lint format clean help

help:
	@echo "ROSClaw v1.0 - Development Commands"
	@echo ""
	@echo "  make install    Install package in editable mode with dev dependencies"
	@echo "  make test       Run the test suite"
	@echo "  make lint       Run ruff linter"
	@echo "  make format     Run ruff formatter"
	@echo "  make clean      Remove build artifacts"
	@echo "  make all        Run lint + test"

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest $(TESTS) -v

lint:
	ruff check $(SRC) $(TESTS)

format:
	ruff format $(SRC) $(TESTS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

all: lint test
