.PHONY: help install install-dev sync test lint mypy pylint format clean build publish act-list act-pylint act-test act-publish

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make install-dev  - Install project with dev dependencies"
	@echo "  make sync         - Sync dependencies (same as install)"
	@echo "  make test         - Run tests with pytest"
	@echo "  make lint         - Run all linting (mypy + pylint)"
	@echo "  make mypy         - Run mypy type checking"
	@echo "  make pylint       - Run pylint"
	@echo "  make format       - Format code (if ruff/black installed)"
	@echo "  make build        - Build the package"
	@echo "  make clean        - Clean build artifacts and caches"
	@echo "  make publish      - Publish to PyPI"
	@echo ""
	@echo "Local workflow testing with act:"
	@echo "  make act-list     - List all workflow jobs"
	@echo "  make act-pylint   - Run pylint workflow locally"
	@echo "  make act-test     - Run test workflow locally"
	@echo "  make act-publish  - Run publish workflow locally (dry-run)"

# Install dependencies
install:
	uv sync

# Install with dev dependencies
install-dev:
	uv sync

# Alias for install
sync:
	uv sync

# Run tests
test:
	uv run pytest tests/ -v

# Run all linting
lint: mypy pylint

# Run mypy type checking
mypy:
	uv run mypy src/tdfpy

# Run pylint
pylint:
	uv run pylint src/tdfpy

# Format code (add ruff or black to dev dependencies if needed)
format:
	@if uv run ruff --version > /dev/null 2>&1; then \
		uv run ruff format src/ tests/; \
	elif uv run black --version > /dev/null 2>&1; then \
		uv run black src/ tests/; \
	else \
		echo "No formatter found. Install ruff or black."; \
	fi

# Build the package
build:
	uv build

# Clean build artifacts and caches
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Publish to PyPI (requires UV_PUBLISH_TOKEN or interactive auth)
publish: build
	uv publish

# Run test workflow locally (using medium ubuntu image with Python pre-installed)
act-test:
	act -j build -W .github/workflows/python-package.yml -P ubuntu-latest=catthehacker/ubuntu:act-latest