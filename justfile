# Default target
default:
  @just --list

# Install dependencies
install:
  uv sync --no-dev

# Install with dev dependencies
install-dev:
  uv sync

# Alias for install
sync:
  uv sync

# Run tests
test:
  uv run pytest tests/ -v

# Run linter
lint:
    uv run ruff check src/

# Format code
format:
    uv run ruff check --select I --fix src/ tests/
    uv run ruff check --select F401 --fix src/ tests/
    uv run ruff format src/ tests/

# ty type checking
ty:
    uv run ty check src/

# Run lint and tests
check:
    just lint
    just ty
    just test



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
  find . -type f -name "*.so" ! -name "libtimsdata.so" -delete

# Build and serve docs
docs:
    uv run --group docs mkdocs serve --dev-addr=localhost:8002

# Build docs to site/
docs-build:
    uv run --group docs mkdocs build

# Publish to PyPI (requires UV_PUBLISH_TOKEN or interactive auth)
publish: build
  uv publish

# Upgrade Python syntax to 3.11+
upgrade:
  @echo "Upgrading Python syntax to 3.12+..."
  @find src/tdfpy tests -name "*.py" -type f -exec uv run pyupgrade --py312-plus {} +
  @echo "Python syntax upgraded to 3.12+"

# Run tests with coverage
test-cov:
    uv run pytest tests --cov=src/tdfpy --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml

codecov-tests:
    uv run pytest tests --cov --junitxml=junit.xml -o junit_family=legacy
