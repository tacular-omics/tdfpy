# Contributing to tdfpy

Thank you for your interest in contributing to **tdfpy**, a Python package for
parsing and centroiding Bruker timsTOF mass spectrometry data. Whether you are
fixing a bug, adding a feature, or improving documentation, your help is
welcome. A JOSS submission is in preparation (the manuscript lives in
`papers/`), so clear and well-tested contributions are especially valuable.

All participants are expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant 2.0).

## Reporting Bugs and Requesting Features

Please use [GitHub Issues](https://github.com/tacular-omics/tdfpy/issues) for
bug reports and feature requests.

**Bug reports** should include:

- Python version and tdfpy version (`python --version`, `pip show tdfpy`).
- Operating system and architecture. tdfpy is pure Python and runs anywhere
  Python 3.12+ does, but decoding behaviour can still be platform-sensitive.
- A minimal reproducing example. If the bug is data-dependent, please indicate
  the acquisition mode (DDA / DIA / PRM) and, if you can share, attach a small
  `.d` directory or use `tdfpy.slice_d_folder` to extract a frame range.
- Expected vs. actual behavior.

**Feature requests** should describe the scientific or practical use case, not
just the desired API.

## Development Setup

Prerequisites: **Python 3.12+**, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/).

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/tdfpy.git
cd tdfpy

# Install all dependencies (dev + extras)
just install-dev

# Verify everything works
just check   # runs lint, tests, and type checking
```

Tests that exercise real Bruker data depend on small example `.d` directories
under `tests/data/`. They are skipped automatically when the data is absent.

There are no pre-commit hooks configured; please run `just check` before
pushing.

## Making Changes

1. Create a branch from `main` with a descriptive name: `fix/dia-scan-range`,
   `feature/watershed-leash`, `docs/pipeline-example`.
2. Keep each pull request focused on a single change.
3. Write clear commit messages that explain *why*, not just *what*.

## Code Style

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/)
(Python 3.12 target):

```bash
just format   # auto-format code
just lint     # check for lint errors
```

**Docstrings** use Google or Sphinx style; match the surrounding module.

**Type annotations** are required on all public functions. Run `just ty` to
check.

## Testing

- All new code needs tests. Tests live in `tests/`.
- Use `pytest.approx` for floating-point comparisons.
- Use the `tmp_path` fixture for any file I/O; do not mock library internals.
- When adding a new centroider, noise filter, or pipeline op, include a test
  that exercises both the Numba and pure-Python code paths where applicable.
- Aim to maintain at least the current branch coverage.

```bash
just test       # run tests
just test-cov   # run tests with coverage report
```

## Documentation

- Add docstrings to all public functions and classes.
- Build the docs locally with `just docs` (serves on `localhost:8002`) and
  verify your changes render correctly.
- When introducing new public API, add or extend the corresponding page under
  `docs/api/`.

## Submitting a Pull Request

Before opening a PR, confirm:

- [ ] `just check` passes (lint + tests + type checking).
- [ ] New code has tests and docstrings.
- [ ] PR description explains the change and its motivation.
- [ ] Any related issue is referenced (e.g., "Closes #42").

CI (GitHub Actions) will run lint, type checking, and tests automatically when
you push. A maintainer will review your PR and may request changes.

## Questions?

Open an [issue](https://github.com/tacular-omics/tdfpy/issues) or reach out to
the maintainers. We are happy to help.
