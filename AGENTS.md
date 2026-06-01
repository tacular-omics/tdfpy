# tdfpy — Agent Guide

This file is the cross-platform instruction set for coding agents (Codex,
Aider, Devin, Cursor, Claude Code, and similar tools) working on this
repository. Use it as the source of truth; project-specific helpers
(`CLAUDE.md`, `.github/copilot-instructions.md`) duplicate or extend this
content for their respective tools.

## Project Overview

**tdfpy** is a Python package for parsing and centroiding Bruker timsTOF
mass spectrometry data (`.tdf` SQLite metadata + `.tdf_bin` binary
spectra). It exposes a high-level object-oriented API for the three
principal acquisition modes — DDA, DIA, and PRM — including the PASEF and
diaPASEF scan strategies. It ships a Numba-accelerated centroiding stack
with two centroiders (greedy m/z merge and watershed region growing) and a
composable peak-processing pipeline (region exclusion → noise filter →
centroider).

- **Package:** `tdfpy`
- **Python:** 3.12+
- **License:** MIT
- **Repository:** https://github.com/tacular-omics/tdfpy
- **Docs:** https://tacular-omics.github.io/tdfpy
- **PyPI:** https://pypi.org/project/tdfpy/

## Common Commands

```bash
just install-dev   # install with dev dependencies (uv sync)
just test          # run pytest
just test-cov      # pytest + coverage report
just lint          # ruff check src/
just ty            # ty type check src/
just format        # ruff format + import sort
just check         # lint + test + ty (full QC)
just build         # uv build → dist/
just clean         # remove build artifacts (preserves libtimsdata.so)
just docs          # serve docs at localhost:8002
```

Run `just check` before pushing. There are no pre-commit hooks.

## Repository Layout

```
src/tdfpy/
├── __init__.py        Public API exports
├── reader.py          High-level DDA / DIA / PRM reader classes
├── elems.py           Frame, Precursor, DiaWindow, PrmTransition dataclasses
├── lookup.py          Index ↔ m/z lookup tables; query-by-mz/rt helpers
├── centroiding.py     get_raw_peaks, get_centroided_spectrum, merge_peaks
├── pipeline.py        RawSpectrum + composable ops (read_spectrum,
│                       exclude_region, apply_noise, convert) + Centroider
│                       ABC + MergePeaksCentroider + WatershedCentroider
├── noise/             NoiseFilter ABC + intensity filters (MadThreshold etc.)
│                       + structural VerticalNoiseFilter
├── regions.py         ChargeStateRegion — drop singly-charged contamination
├── slicer.py          slice_d_folder — extract a frame-range subset of a .d
├── viz.py             plot_centroiding — 2x2 diagnostic panel
├── tdf.py             PandasTdf — pandas wrapper around analysis.tdf SQLite
├── timsdata.py        ctypes wrapper around libtimsdata.so
├── constants.py       Physical constants (proton mass, table names)
├── libtimsdata.so     Bruker native library (Linux)
└── timsdata.dll       Bruker native library (Windows)
```

## Key Design Decisions

- **No Rust toolchain.** The previous Rust extension (`_tdfpy_rust`, v0.3.x)
  was replaced by a Numba `@njit(cache=True)` kernel in v1.0.0. Do not
  reintroduce maturin or PyO3.
- **Pure-Python wheel.** Build backend is hatchling; wheels are
  `py3-none-any`. The Bruker native library is bundled in the wheel
  (`libtimsdata.so` on Linux, `timsdata.dll` on Windows).
- **Numba is a hard dependency.** Every JIT-compiled kernel has a
  pure-Python NumPy fallback gated on `_HAS_NUMBA`. When adding a new
  kernel, write both paths and add a Numba/Python equivalence test.
- **Lazy spectral access.** Metadata is loaded eagerly when a reader is
  opened; spectral binary data is read on demand by `.peaks` /
  `.raw_peaks()` / `.centroid()`. Frame-element objects hold a reference to
  the open `TimsData` connection — they cannot be used after the reader's
  `with` block exits.
- **Composable pipeline.** Every centroiding entry point orchestrates the
  same ordered ops: `read_spectrum → subset_scans → exclude_region →
  apply_noise → centroider`. Each op consumes and produces a `RawSpectrum`
  in integer-index space; conversion to float (m/z, 1/K0) happens once at
  the end via `convert`. Power users compose the ops directly.
- **Two centroiders.** `MergePeaksCentroider` (default) operates on float
  m/z; `WatershedCentroider` operates on integer (scan, TOF-index) space
  and avoids float binning. Both implement the `Centroider` ABC and are
  swappable per call site via `frame.centroid(centroid=…)`.
- **Region exclusion ≠ noise filtering.** `ChargeStateRegion` answers
  "which part of the (m/z, 1/K0) plane are we interested in?" — a physical
  knowledge claim. Noise filters answer "of what's left, what's real
  signal?" — a statistical or structural claim. They are separate pipeline
  stages.

## Testing

Tests live in `tests/`. Some tests require real Bruker `.d` data at
`tests/data/example_dda.d`, `tests/data/example_dia.d`, and
`tests/data/example_prm.d`; tests are skipped automatically when the data
is absent.

```bash
just test           # all tests
just test-cov       # with coverage report
```

When adding a new centroider, noise filter, or pipeline op, include a test
that exercises both the Numba and pure-Python code paths where applicable.
Use `pytest.approx` for floating-point comparisons.

## Dependencies

| Dependency | Role |
|---|---|
| `numpy>=2.0` | Array operations throughout |
| `pandas>=2.0` | SQLite metadata access via `PandasTdf` |
| `numba>=0.59` | JIT-compiled centroiding and watershed kernels |

Dev / docs: `ruff`, `ty`, `pytest`, `pytest-cov`, `pyupgrade`, `mkdocs`,
`mkdocs-material`, `mkdocstrings`.

## Code Style

- Ruff handles formatting and linting; configuration is in `pyproject.toml`.
- Type annotations are required on all public functions. Run `just ty`.
- Public surface lives in `src/tdfpy/__init__.py`; add new exports there.
- Frozen dataclasses are the preferred pattern for tunable algorithm
  configurations (`Centroider` subclasses, `NoiseFilter` subclasses,
  `ChargeStateRegion`). Frozen makes them hashable so they can be used as
  cache keys (e.g. Streamlit `@cache_data`).
- Docstrings use Google or Sphinx style — match the surrounding module.

## Release Process

1. Update version in `src/tdfpy/__init__.py`.
2. Roll `[Unreleased]` to `[X.Y.Z]` in `CHANGELOG.md` with the release date.
3. Commit and push to `main`.
4. Create and push a `vX.Y.Z` tag.
5. Create a GitHub release — this triggers
   `.github/workflows/python-publish.yml` to build and publish to PyPI, and
   (if Zenodo integration is enabled) Zenodo mints an archival DOI.

## Repo Notes

- `benchmark/` is git-ignored (local only).
- `papers/` contains the JOSS manuscript (`paper.md`, `paper.bib`).
- `apps/` contains internal Streamlit dashboards used during algorithm
  development. Not part of the public package; not installed by `pip
  install tdfpy`.
- Docs are MkDocs + mkdocstrings, auto-deployed to GitHub Pages on push to
  `main`.
- `libtimsdata.so` and `timsdata.dll` are committed to git (and explicitly
  un-ignored in `.gitignore`) so they ship in every PyPI wheel.

## Public API Cheat Sheet

```python
# Acquisition-mode readers
from tdfpy import DDA, DIA, PRM, get_acquisition_type, slice_d_folder

# Frame elements (dataclasses, do not construct directly)
from tdfpy import (
    Frame, DDAMs1Frame, DIAMs1Frame, PRMMs1Frame,
    DiaWindow, DiaWindowGroup, Precursor,
    PrmTarget, PrmTransition,
    MetaData, Calibration,
)

# Lookups (frame[id], precursor[id], window[id], target[id])
from tdfpy import (
    DiaWindowLookup, Ms1FrameLookup, PrecursorLookup,
    PrmTargetLookup, PrmTransitionLookup,
)

# Peak extraction — convenience
from tdfpy import get_raw_peaks, get_centroided_spectrum, merge_peaks

# Peak extraction — composable pipeline ops
from tdfpy import (
    RawSpectrum, read_spectrum, subset_scans,
    exclude_region, apply_noise, convert, centroid_peaks,
)

# Centroiders
from tdfpy import Centroider, MergePeaksCentroider, WatershedCentroider

# Region exclusion
from tdfpy import ChargeStateRegion

# Noise filters
from tdfpy import (
    NoiseFilter, NoiseSpec, coerce_filters,
    IntensityThreshold, AbsoluteThreshold, MadThreshold,
    PercentileThreshold, HistogramThreshold, BaselineThreshold,
    IterativeMedianThreshold, VerticalNoiseFilter,
)

# Visualization
from tdfpy import plot_centroiding

# Low-level
from tdfpy import PandasTdf, TimsData, timsdata_connect
```

## What Not To Do

- Don't add a new build backend or compilation step. The wheel is
  pure-Python by design.
- Don't reintroduce a Rust extension. Numba covers the perf needs.
- Don't `import numba` at module top level outside the existing
  `_HAS_NUMBA` try/except — Numba imports are slow.
- Don't break the `with` context-manager contract. Spectral access after
  the reader closes must raise `RuntimeError`, not return stale data.
- Don't write tests that mock `TimsData` internals. Use the example `.d`
  fixtures under `tests/data/` instead.
- Don't add post-centroid noise filters. Noise filtering belongs
  *pre-centroid* in the pipeline — that's where filters can usefully
  suppress satellites before the centroider sees them.
