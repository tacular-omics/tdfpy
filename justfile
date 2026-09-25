# Default target
default:
  @just --list

# Install dependencies (dev included)
install:
  uv sync

# Alias for install — the name CONTRIBUTING.md and docs/getting-started.md use
install-dev:
  uv sync

# Install runtime dependencies only, no dev group
install-prod:
  uv sync --no-dev

# Alias for install
sync:
  uv sync

# Run tests (fast default: skips @pytest.mark.slow, small Hypothesis profile)
test:
  uv run pytest tests/ -v

# Run every test, slow ones included, with the CI Hypothesis profile
test-all:
  RUN_SLOW=1 HYPOTHESIS_PROFILE=thorough uv run pytest tests/ -v

# Lint src, tests and scripts (CI runs the same)
lint:
    uv run ruff check src/ tests/ scripts/

# Sort imports, drop unused imports, then format src, tests and scripts
format:
    uv run ruff check --select I --fix src/ tests/ scripts/
    uv run ruff check --select F401 --fix src/ tests/ scripts/
    uv run ruff format src/ tests/ scripts/

# Verify formatting without changing files
format-check:
    uv run ruff format --check src/ tests/ scripts/

# Type check src, including the optional MCP interface
ty:
    uv run --extra mcp ty check src/

# Lint, format check, type check, then tests
check:
    just lint
    just format-check
    just ty
    just test

# --- release (standard tacular-omics recipes; canonical copy in the workspace templates/) ---

# Set the version everywhere and date the [Unreleased] changelog section
set-version version:
    python scripts/release_version.py sync --set {{version}}

# Copy __version__ to CITATION.cff / .zenodo.json after editing it by hand
sync-version:
    python scripts/release_version.py sync

# Fail if version metadata disagrees
check-version:
    python scripts/release_version.py check

# Build the sdist and wheel into dist/
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
  find . -type f -name "*.so" -delete

# Build and serve docs
docs:
    uv run --group docs mkdocs serve --dev-addr=localhost:8002

# Build docs to site/
docs-build:
    uv run --group docs mkdocs build

# The docs copy ships at /llms-full.txt on the docs site; both copies are committed.
# Regenerate llms-full.txt and docs/llms-full.txt, expanding ::: API directives
llms-full:
    uv run python scripts/build_llms_full.py

# Build the JOSS paper PDF to papers/paper.pdf (requires Docker; mirrors the Draft PDF CI)
paper:
    docker run --rm --volume "$PWD":/data --user "$(id -u):$(id -g)" \
        --env JOURNAL=joss openjournals/inara -o pdf papers/paper.md
    @echo "Wrote papers/paper.pdf"

# Upgrade Python syntax to 3.12+ (pyupgrade)
upgrade:
  @echo "Upgrading Python syntax to 3.12+..."
  @find src/tdfpy tests -name "*.py" -type f -exec uv run pyupgrade --py312-plus {} +
  @echo "Python syntax upgraded to 3.12+"

# Run tests with coverage
test-cov:
    RUN_SLOW=1 uv run pytest tests --cov=src/tdfpy --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml --junitxml=junit.xml

# Tests with coverage and a legacy-format junit.xml, for Codecov
codecov-tests:
    RUN_SLOW=1 uv run pytest tests --cov --junitxml=junit.xml -o junit_family=legacy

# Test the optional MCP interface, including its stdio protocol
test-mcp:
    RUN_SLOW=1 uv run --extra mcp pytest tests/test_mcp.py -v
