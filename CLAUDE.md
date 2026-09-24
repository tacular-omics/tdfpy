# tdfpy — agent guide

Canonical instructions for coding agents (Claude Code, Codex, Copilot, Cursor)
working in this repo. `AGENTS.md` and `.github/copilot-instructions.md` point
here. For agents *using* the package, see `llms.txt` / `llms-full.txt`.

## Project overview

**tdfpy** reads and centroids Bruker timsTOF data (`.d` folders: `analysis.tdf`
SQLite metadata + `analysis.tdf_bin` binary spectra) in pure Python/NumPy. It has
mode-aware readers for DDA, DIA and PRM (PASEF / diaPASEF), a Numba-accelerated
centroiding stack with two centroiders (greedy m/z merge and watershed region
growing), and a composable peak-processing pipeline (region exclusion →
smoothing → noise filter → centroider). Users are proteomics developers who need
timsTOF spectra without Bruker's SDK.

- Python 3.12+, MIT, `src/` layout, hatchling, pure-Python wheel.
- Repo https://github.com/tacular-omics/tdfpy · docs https://tacular-omics.github.io/tdfpy
  · PyPI https://pypi.org/project/tdfpy/ · DOI 10.5281/zenodo.19100532
- tacular-omics graph: tier 0, **no sibling dependencies**. Downstream:
  `spxtacular` (optional extra `tdfpy`) and `tdfextractor` (outside the workspace).
  A breaking API change here needs a note for both.

Runtime deps: `numpy>=2.0`, `pandas>=2.0` (SQLite metadata via `PandasTdf`),
`numba>=0.59` (JIT kernels), `zstandard>=0.22` only on Python < 3.14 (3.14+ uses
stdlib `compression.zstd`). Extras: `viz` (matplotlib, for `plot_centroiding`),
`mcp` (MCP SDK, for `tdfpy-mcp`).

## Commands

```bash
just install        # uv sync (alias: install-dev, sync); install-prod = --no-dev
just test           # uv run pytest tests/ -v   (~40 s, 500+ tests)
just test-cov       # pytest + branch coverage (term, html, xml, junit.xml)
just test-mcp       # MCP interface tests, with --extra mcp
just lint           # ruff check src/ tests/ scripts/
just format         # ruff import sort + unused-import fix + format
just format-check   # ruff format --check src/ tests/ scripts/
just ty             # uv run --extra mcp ty check src/
just check          # lint + format-check + ty + test
just docs           # mkdocs serve on localhost:8002 (docs dependency group)
just docs-build     # mkdocs build to site/
just llms-full      # regenerate llms-full.txt + docs/llms-full.txt (scripts/build_llms_full.py)
just build          # uv build -> dist/
just check-version  # version metadata agrees across __init__/CITATION.cff/.zenodo.json
```

CLI entry points: `tdfpy validate <run.d> [--full]` (JSON report, exit 1 if
invalid) and `tdfpy-mcp --data-root DIR --output-dir DIR` (needs `tdfpy[mcp]`;
also `python -m tdfpy.mcp`).

Distribution check (see `docs/maintenance.md`): `uv build --out-dir dist` then
`uv run python scripts/verify_distribution.py dist`.

Pre-commit hooks (`.pre-commit-config.yaml`, install with `uvx pre-commit install`)
run ruff check + ruff format --check on `src tests scripts` at commit time. CI
(`.github/workflows/ci.yml`) runs ruff on the same paths as `just lint` /
`just format-check`.

## Architecture

```
src/tdfpy/
├── __init__.py      public API exports and __version__ (hatch version source)
├── reader.py        DDA / DIA / PRM reader classes (context managers), get_acquisition_type
├── elems.py         Frame, DDA/DIA/PRM MS1 frames, Precursor, PasefFrameMsmsInfo,
│                    DiaWindow(Group), PrmTarget, PrmTransition, MetaData, Calibration
├── lookup.py        *Lookup containers: index by id, .query() by m/z and RT
├── centroiding.py   get_raw_peaks, get_centroided_spectrum, merge_peaks,
│                    get_mobility_collapsed_spectrum (Numba kernel + Python fallback)
├── pipeline.py      RawSpectrum + ops (read_spectrum, subset_scans, exclude_region,
│                    smooth/box_smooth/Smooth, apply_noise, convert, centroid_peaks),
│                    Centroider ABC, MergePeaksCentroider, WatershedCentroider
├── noise/           NoiseFilter ABC, coerce_filters, NoiseSpec
│   ├── intensity.py   statistical thresholds (Mad, Percentile, Histogram, Baseline, ...)
│   ├── structural.py  VerticalNoiseFilter, HorizontalHaloFilter (Numba kernels)
│   └── gates.py       SelectionPolygonGate (ddaPASEF), DiaMs1WindowGate (diaPASEF)
├── regions.py       ChargeStateRegion: drop the singly-charged band
├── processing.py    iter_window_spectra / iter_precursor_spectra: decode a shared frame once
├── validation.py    validate_acquisition, ValidationReport/Issue (backs `tdfpy validate`)
├── __main__.py      `tdfpy` console script (validate subcommand)
├── slicer.py        slice_d_folder: copy a frame range of a .d into a new .d
├── viz.py           plot_centroiding: 2x2 diagnostic figure (needs matplotlib)
├── tdf.py           PandasTdf: pandas view of analysis.tdf tables
├── timsdata.py      TimsData / timsdata_connect: pure-Python .tdf_bin frame decoder,
│                    format guards, UnsupportedTdfError
├── calibration.py   TOF index <-> m/z and scan <-> 1/K0 models (no I/O),
│                    UnsupportedCalibrationError
├── constants.py     physical constants, table names
├── types.py         Polarity, ToleranceUnit (same as tacular.types)
├── _validation.py   shared argument checks before array ops / JIT kernels
├── _diagnostics.py  internal pipeline instrumentation, no stable API
└── mcp/             optional MCP server (models, service, server); importing
                     tdfpy never imports the MCP SDK
```

Data flow: reader opens `analysis.tdf` (all frame/precursor/window metadata loaded
eagerly into dataclasses and lookups) and a `TimsData` handle on `analysis.tdf_bin`.
Spectral access (`.merged_peaks()`, `.scan_peaks()`, `.raw_peaks()`, `.centroid()`) decodes on demand through
`read_spectrum → subset_scans → exclude_region → smooth → apply_noise → centroider`,
all in integer (scan, TOF-index) space; `convert` maps to m/z and 1/K0 once at the end.

Other directories: `tests/` (fixtures in `tests/data/`), `docs/` (MkDocs +
mkdocstrings, deployed to GitHub Pages on push to main), `scripts/` (benchmarks,
golden-data generators, llms-full builder, release and distribution checks),
`papers/` (JOSS manuscript), `apps/` (internal Streamlit dashboards, not packaged),
`benchmark/` (git-ignored, local only).

## Public API

All names below are exported from `tdfpy` (`__all__`); add new exports there.

- Readers: `DDA`, `DIA`, `PRM`, `get_acquisition_type` (returns `AcquisitionType`, a
  `StrEnum`), `slice_d_folder`
- Frame elements (returned by readers, not constructed by users): `Frame`,
  `DDAMs1Frame`, `DIAMs1Frame`, `PRMMs1Frame`, `Precursor`, `PasefFrameMsmsInfo`,
  `DiaWindow`, `DiaWindowGroup`, `PrmTarget`, `PrmTransition`, `MetaData`,
  `Calibration`, `FrameMetadata`, plus `MsMsType`, `Polarity` and `ToleranceUnit` (`Literal`s in `types.py`, same as `tacular.types`), `MetaValue`. Elements are
  frozen, slotted, keyword-only dataclasses; ids end in `_id`, retention time is `rt`,
  ion mobility (1/K0) is `ook0`
- Lookups: `Ms1FrameLookup`, `PrecursorLookup`, `DiaWindowLookup`,
  `PrmTargetLookup`, `PrmTransitionLookup`
- Convenience extraction: `get_raw_peaks`, `get_centroided_spectrum`,
  `merge_peaks`, `get_mobility_collapsed_spectrum`, `iter_window_spectra`, `iter_precursor_spectra`
- Pipeline ops: `RawSpectrum`, `read_spectrum`, `subset_scans`, `exclude_region`,
  `Smooth`, `smooth`, `box_smooth`, `apply_noise`, `convert`, `centroid_peaks`
- Centroiders: `Centroider` (ABC), `MergePeaksCentroider` (default), `WatershedCentroider`
- Region exclusion: `ChargeStateRegion`
- Noise: `NoiseFilter` (ABC), `NoiseSpec`, `coerce_filters`, `IntensityThreshold`,
  `AbsoluteThreshold`, `MadThreshold`, `PercentileThreshold`, `HistogramThreshold`,
  `BaselineThreshold`, `IterativeMedianThreshold`, `VerticalNoiseFilter`,
  `HorizontalHaloFilter`, `SelectionPolygonGate`, `DiaMs1WindowGate`
- Validation: `validate_acquisition`, `ValidationReport`, `ValidationIssue`
- Visualization: `plot_centroiding`
- Low level: `PandasTdf`, `TimsData`, `timsdata_connect`, `ook0_to_ccs`, `ccs_to_ook0`
- Errors (`errors.py`): `TdfpyError(ValueError)` is the base; `TdfpyKeyError` (+`KeyError`)
  for a missing id or key, `ReaderClosedError` (+`RuntimeError`) for spectral access
  or `reader.timsdata` / `metadata` / `calibration` / `pandas_tdf` after close,
  `AcquisitionTypeError` for opening the wrong reader for the run (e.g. `DDA` on DIA
  data), `UnsupportedTdfError` / `UnsupportedCalibrationError`
  (+`NotImplementedError`). SQLite errors are wrapped as `TdfpyError`.

Output shapes: `raw_peaks()` / `centroid()` return `(N, 3)` `[m/z, intensity,
ion_mobility]`; `Precursor.merged_peaks()` is `(N, 2)`; `scan_peaks()` (Frame, DiaWindow, PrmTransition, PasefFrameMsmsInfo) is a *list* of
per-scan `(N, 2)` arrays. Every spectral accessor is a method; `merged_peaks()` is the
expensive one (greedy m/z merge). Every `*_range` (m/z, 1/K0, CCS, voltage, RT) is
`(low, high)`, whatever the scan order.

## Conventions

- Ruff formats and lints (config in `pyproject.toml`, line length 160). `ty`
  type-checks `src/`. Type annotations are required on public functions.
- Docstrings: Google style (`Args:` / `Returns:`), with Sphinx cross-reference
  roles (`:class:`, `:func:`) in prose. Match the surrounding module. mkdocstrings renders
  them with `docstring_style: google`.
- Frozen dataclasses for tunable algorithm configs (`Centroider` and
  `NoiseFilter` subclasses, `ChargeStateRegion`, `Smooth`): hashable, so usable as
  cache keys (e.g. Streamlit `@cache_data`).
- Errors: validate arguments up front with `_validation.py` helpers before arrays
  reach a JIT kernel. Unsupported formats raise `UnsupportedTdfError` /
  `UnsupportedCalibrationError`; spectral access on a closed reader raises
  `ReaderClosedError`. New errors subclass `TdfpyError`; methods are snake_case.
- Logging: module-level `logger = logging.getLogger(__name__)`; never configure
  logging from the library.
- Tests: `tests/test_<area>.py`. Real fixtures `tests/data/example_{dda,dia,prm}.d`
  are committed; use them instead of mocking `TimsData` internals. Use
  `pytest.approx` for floats. New kernels, centroiders, filters or ops need a
  test covering both the Numba and pure-Python paths. `tests/test_docs.py` runs
  every code block in `docs/getting-started.md` and `docs/analysis.md`
  (pytest-examples), so keep those examples runnable.

## Gotchas and design rules

- **No native library.** `analysis.tdf_bin` is decoded in Python/NumPy
  (since v3.0.0); Bruker's `libtimsdata` is gone. Do not reintroduce it or any
  ctypes binding. Wheels are `py3-none-any`; `scripts/verify_distribution.py`
  fails if one contains a `.so`/`.dll`/`.dylib`/`.pyd`.
- **No Rust, no build step.** The old Rust extension (`_tdfpy_rust`, v0.3.x) was
  replaced by Numba `@njit(cache=True)` kernels in v1.0.0. No maturin/PyO3, no
  new build backend.
- **Unvalidated formats must raise, never approximate.** A wrong calibration gives
  plausible numbers nothing downstream can detect. Legacy `TimsCompressionType` 1
  (`SUPPORTED_COMPRESSION_TYPE = 2` in `timsdata.py`), unknown calibration
  `ModelType`s, `use_recalibrated_state=True` and pressure compensation all raise.
  Only add support after validating against Bruker's library and extending
  `tests/test_calibration_golden.py`. Do not loosen a guard to make a file "work".
- **Never regenerate the golden JSON to make a test pass.**
  `tests/data/calibration_golden.json` and `peaks_golden.json` were captured from
  Bruker's library while it was vendored and are the only record of its
  behaviour. A diff means the reader changed. The generators
  (`scripts/generate_*_golden.py`) need a Bruker SDK that is no longer in the repo.
- **Numba is a hard dependency, but every kernel has a pure-Python fallback**
  gated on `_HAS_NUMBA` (in `centroiding.py`, `pipeline.py`,
  `noise/structural.py`). Import numba only inside that existing try/except;
  numba imports are slow. The first centroid call JIT-compiles (seconds).
- **Lazy spectral access.** Frame elements hold the reader's `TimsData`; after the
  `with` block, spectral access must raise `ReaderClosedError`, never return stale data.
  Do not break this contract.
- **Region exclusion is not noise filtering.** `ChargeStateRegion` says which part
  of the (m/z, 1/K0) plane matters (physical knowledge); noise filters say what
  of the remainder is real signal. Separate pipeline stages.
- **Noise filtering is pre-centroid only.** Do not add post-centroid noise
  filters; that is where filters can suppress satellites before the centroider.
- **DIA/PRM MS2 ion mobility is the precursor's**, because TIMS sits before the
  collision cell. Do not document it as a fragment property.
- **`llms-full.txt` is generated.** `just llms-full` writes the root and `docs/`
  copies: the orientation text in `HEADER` of `scripts/build_llms_full.py`, then
  the docs pages with each `::: tdfpy.X` directive expanded from the installed
  package. Edit the header or the docs, not the output, and re-run it after
  changing a public signature or docstring.

## Releasing

Only the tacular-omics overseer bumps versions or publishes. See `just --list`
(`set-version`, `sync-version`, `check-version`) and `docs/maintenance.md`.
Version source: `__version__` in `src/tdfpy/__init__.py` (`[tool.hatch.version]`),
mirrored in `CITATION.cff` and `.zenodo.json` by `scripts/release_version.py`.
`CHANGELOG.md` keeps an `[Unreleased]` section. Publishing runs from
`.github/workflows/publish.yml` when a GitHub release is published (PyPI trusted
publishing); Zenodo archives the release. There is no local publish recipe.

## Workspace note

This repo is also developed inside the tacular-omics uv workspace
(`~/Repos/tacular-omics/packages/tdfpy`); there `uv run` uses the shared `.venv`
and the root `uv.lock`, not this repo's. See the workspace CLAUDE.md.
