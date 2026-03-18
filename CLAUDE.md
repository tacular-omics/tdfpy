# tdfpy — Claude Code Guide

## Project Overview

Python package for parsing and processing Bruker timsTOF mass spectrometry data (.tdf and .tdf_bin files). Provides a high-level API for DDA, DIA, and PRM acquisition modes with Numba-accelerated centroiding.

**Package:** `tdfpy` | **Version:** 1.0.0 | **Python:** 3.12+ | **License:** MIT

## Common Commands

```bash
just install-dev   # Install with dev dependencies (uv sync)
just test          # Run pytest
just lint          # ruff check src/
just ty            # ty type check src/
just format        # ruff format + import sort
just check         # lint + test + ty (full QC)
just build         # uv build → dist/
just clean         # Remove build artifacts (preserves libtimsdata.so)
just docs          # Serve docs at localhost:8002
```

## Architecture

```
src/tdfpy/
├── __init__.py        # Public API exports (DDA, DIA, PRM, merge_peaks, etc.)
├── reader.py          # High-level DDA/DIA/PRM classes
├── elems.py           # Frame, Precursor, DiaWindow dataclasses
├── centroiding.py     # merge_peaks() + get_centroided_spectrum() + Numba kernel
├── timsdata.py        # ctypes wrapper around libtimsdata.so (Bruker C library)
├── tdf.py             # PandasTdf — SQLite metadata via pandas
├── noise.py           # estimate_noise_level() — 5 filtering strategies
├── lookup.py          # Lookup tables for index↔m/z conversion
├── constants.py       # Physical constants
├── elems.py           # Data element dataclasses
└── libtimsdata.so     # Bruker native library (Linux; .dll on Windows)
```

## Key Design Decisions

- **Centroiding backend:** `centroiding.py` uses a Numba `@njit(cache=True)` kernel (`_merge_peaks_numba_kernel`) with a pure-Python NumPy fallback. `_HAS_NUMBA` flag controls dispatch. The `use_numba=True` parameter in `merge_peaks()` / `get_centroided_spectrum()` can force the Python path.
- **No Rust:** The previous Rust extension (`_tdfpy_rust`) was replaced with Numba in v1.0.0. Do not reintroduce maturin or PyO3.
- **Build system:** hatchling (pure-Python wheel, `py3-none-any`). No compilation step needed.
- **Bruker library:** `libtimsdata.so` is bundled in the package. Access via `TimsData` class in `timsdata.py`.

## Testing

Tests require real Bruker `.d` data at `tests/data/200ngHeLaPASEF_1min.d`. Many tests are skipped if the data is absent.

```bash
just test           # all tests
just test-cov       # with coverage report
```

Key test file: `tests/test_spectra.py` — covers `merge_peaks`, Numba/Python equivalence.

## Dependencies

| Dependency | Role |
|---|---|
| `numpy` | Array operations throughout |
| `pandas` | SQLite metadata access via `PandasTdf` |
| `numba>=0.59` | JIT-compiled centroiding kernel |

Dev: `ruff`, `ty`, `pytest`, `pytest-cov`, `pyupgrade`

## Release Process

1. Update version in `src/tdfpy/__init__.py`
2. Update `CHANGELOG.md`
3. Commit and push to `main`
4. Create and push a `vX.Y.Z` tag
5. Create a GitHub release — this triggers `.github/workflows/python-publish.yml` which builds and publishes to PyPI

## Repo Notes

- `benchmark/` is git-ignored (local only)
- `papers/` contains the JOSS manuscript (`paper.md`, `paper.bib`)
- Docs are MkDocs + mkdocstrings, auto-deployed to GitHub Pages on push to `main`
