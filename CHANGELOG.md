# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`TimsData.read_frame_arrays`** — reads a scan range as three flat parallel
  arrays (`scan_indices`, `tof_indices`, `intensities`) instead of one array
  pair per scan. Peaks for a contiguous scan range are already contiguous in the
  decoded frame, so this slices where `readScans` has to split. Prefer it
  wherever you would have concatenated `readScans` output back together.

### Performance

- **`read_spectrum` is up to 2x faster** — it no longer splits a frame into
  per-scan arrays only to immediately concatenate them back. The gain is largest
  on sparse frames, where the per-scan overhead dominated: 200 mixed DDA frames
  went from 340 ms to 168 ms, while 5 dense MS1 frames improved ~10%.
- **`get_mobility_collapsed_spectrum` is 3.6x faster on a whole frame**
  (74 ms → 21 ms for an MS1 frame) and ~1.3x on small PASEF ranges. The per-peak
  Python dict rollup is now a vectorised aggregation that picks between
  `bincount` and `unique` based on how the peak count compares to the TOF grid
  width — the two run opposite ways, and picking wrongly costs ~10x either way.
- **Frame decoding avoids one full copy** of every payload by viewing the
  de-interleaved bytes as `uint32` rather than round-tripping through
  `ascontiguousarray().tobytes()`.

Frame decoding remains bit-exact against Bruker over all 1710 fixture frames and
29,399,513 peaks.

## [3.0.0] - 2026-08-04

Bruker's `libtimsdata` is gone. tdfpy now reads `analysis.tdf_bin` itself, which
removes 16 MB of proprietary binaries from the wheel, drops the redistribution
question, and lifts the Linux/Windows-x86-64 restriction — macOS and ARM work.

### Removed

- **`libtimsdata.so` / `timsdata.dll` are no longer bundled or required.** All
  ctypes is gone. CI now fails if a wheel contains any native binary.
- **Native entry points with no callers**: `extractProfileForFrame`,
  `extractChromatograms`, `readPasefProfileMsMs`,
  `readPasefProfileMsMsForFrame`, `readPasefMsMsForFrame`, and
  `readScansDllBuffer`. `readPasefMsMs` and
  `extractCentroidedSpectrumForFrame` are also gone; see *Changed*.
- **`TimsData.dll` and `TimsData.initial_frame_buffer_size`** attributes.

### Changed

- **BREAKING — `Precursor.peaks` and `PasefFrameMsmsInfo.peaks` no longer call
  Bruker's peak picker.** They sum intensities per TOF index over the relevant
  scan ranges (collapsing ion mobility) and centroid with `merge_peaks` at
  30 ppm. Bruker's algorithm is proprietary and appears to smooth before
  picking, so peak lists differ slightly. Measured per item over 10 precursors
  and 12 DIA windows: strong peaks agree to 0.0–1.9 ppm, total ion current to
  within 4%, 92.8–100% of Bruker's intensity falls within 10 ppm of one of our
  centroids, and peak counts run 0.95–1.09×. `tests/test_peaks_divergence.py`
  enforces those bounds.
- **`TimsData.handle`** is now the open `analysis.tdf_bin` file object rather
  than a native handle. It is still `None` after `close()`, which is all any
  caller checked.
- Wheel gains a `zstandard` dependency on Python < 3.14; 3.14+ uses the
  standard library's `compression.zstd`.

### Added

- **`tdfpy.calibration`** — the TOF-index↔m/z and scan↔1/K0 models, as pure
  functions over the `MzCalibration` / `TimsCalibration` tables. Reproduces
  Bruker to ~1e-10 relative for m/z and ~1e-15 for mobility. Note that no other
  open-source reader uses these tables; they approximate from `GlobalMetadata`
  instead, which is off by 6.35 Th (5.4%) on the bundled DDA fixture.
- **`get_mobility_collapsed_spectrum`** — the mobility-collapse + greedy-merge
  helper backing the two `.peaks` properties.
- **`UnsupportedTdfError` / `UnsupportedCalibrationError`.** Formats that have
  not been validated against Bruker's library now raise instead of returning
  plausible-looking numbers: legacy `TimsCompressionType` 1 (per-scan LZF),
  unknown calibration `ModelType`s, `use_recalibrated_state=True`, and any
  pressure-compensation strategy other than `NoPressureCompensation`.
- **Golden regression tests.** `tests/data/calibration_golden.json` pins the
  conversions at 0.01 ppm across 21 frames spanning both PRM calibration rows,
  the temperature extremes driving `dC1`/`dC2`, and every `MsMsType`. Frame
  decoding is separately verified bit-exact against `tims_read_scans_v2` over
  all 1710 frames and 29,399,513 peaks of the three fixtures. Previously a
  deliberately injected 1e-3 relative m/z error passed the entire suite.

## [2.2.0] - 2026-07-07

### Added

- **Precursor-space MS1 gates (`tdfpy.noise.gates`).** Two acquisition-aware `NoiseFilter`s that drop MS1 signal the instrument never fragments — signal that cannot become an identification. Both convert their `(m/z, 1/K0)` region once to per-scan integer TOF-index intervals (via the run calibration) and test membership with a vectorised binary search; both no-op (keep everything) when the run carries no region. Ported from the `dnoise` Rust tool. Compose like any filter, e.g. `noise=[SelectionPolygonGate(), MadThreshold(k=3)]`.
  - **`SelectionPolygonGate`** (ddaPASEF) — keeps only MS1 points inside the run's PASEF selection polygon (the "IMS PolygonFilter" read from `analysis.tdf`'s `GroupProperties`). A generalisation of `ChargeStateRegion` from a single line to the real acquisition polygon. Skipped on diaPASEF (where the same property stores window quads). Padded in physical units (`mz_pad` default 5 Da, `im_pad` default 0.05 1/K0) so an edge precursor keeps its isotopic envelope / mobility spread rather than being clipped at a hard polygon boundary; pass `mz_pad=0.0, im_pad=0.0` for a hard cutoff.
  - **`DiaMs1WindowGate`** (diaPASEF) — keeps only MS1 points inside the union of the isolation windows (`DiaFrameMsMsWindows`); everything outside is a precursor the method never isolates. Windows are padded in physical units (`mz_pad` default 5 Da, `im_pad` default 0.05 1/K0). No-op on ddaPASEF.
  - Both gates now **no-op (keep everything) on non-MS1 frames** rather than testing fragment peaks against the MS1 precursor region (which would empty an MS2 spectrum), so they are safe to leave in a `noise=[...]` list applied across frame types.
  - `build_window_intervals` **clamps a negative `scan_lo` to 0 and skips boxes lying wholly outside the scan range** instead of letting a negative bound wrap around via Python indexing.

## [2.1.0] - 2026-07-02

### Fixed

- **`convert(..., ion_mobility_type="voltage")` returned garbage.** The voltage branch fed per-peak 1/K0 values into `scanNumToVoltage`, which expects scan numbers, so every peak's voltage was looked up at ~scan 0. Now passes the peak scan indices.
- **`Calibration.mode` and `Calibration.std_ppm` always raised `KeyError`.** They read non-existent table keys (`CalibrationMode` / `CalibrationStdPpm`) instead of the real `MzCalibrationMode` / `MzStandardDeviationPPM`.
- **`DiaWindowLookup.query_range` filtered on the wrong field.** It matched against `window_index` (and, for a `DiaWindowGroup` argument, mixed id spaces) instead of `window_group`, which is how the lookup is keyed everywhere else.
- **DIA reader truncated `t1`/`t2` to `int`.** The `Frame.t1`/`t2` fields are `float`, and the DDA/PRM readers already read them as such; the DIA reader now matches.
- **`merge_peaks(..., max_peaks=0)` diverged between the numba and pure-Python kernels** (numba capped after one peak, Python treated it as unlimited). Both now treat `None`/non-positive as "no limit".
- **Pure-Python centroiding produced `NaN` m/z on an all-zero-intensity cluster** (divide-by-zero) where the numba kernel fell back to the seed peak; the Python path now guards the same way.
- **`plot_centroiding` "% of intensity" could exceed 100 %.** The lost-intensity fraction now divides by the pre-noise total (kept + rejected).

### Added

- **Actionable exception messages throughout.** Lookup misses (frames, precursors, DIA windows, PRM targets/transitions) now report the requested id *and* the loaded range; "frame not found", scan-range, unknown-`MsMsType`, unknown-polarity, native-timsdata, and frame-size-cap errors now name the offending value and the valid set/constraint. Aimed at making failures self-explanatory to both humans and LLM/agent callers.
- **Library-wide logging/observability.** Module loggers added to `reader`, `pipeline`, `elems`, `slicer`, and the `noise` filters. Opening a `.d` folder logs a one-line summary; `apply_noise`, `exclude_region`, `read_spectrum`, and the intensity-threshold filters emit debug traces and **warn when an operation removes all peaks or hits a degenerate/non-finite threshold** (so "why is my result empty?" is answerable from the logs). Slicing a folder logs progress and warns before overwriting an existing destination.
- **`ChargeStateRegion` now rejects a negative m/z-vs-1/K0 slope** in `__post_init__` instead of silently excluding the wrong half-plane.

### Changed

- **`IntensityThreshold.keep_mask` keeps all points (with a warning) on a non-finite threshold** instead of silently dropping every peak on degenerate input.
- **`extractChromatograms` re-raises exceptions from the user-supplied generator/sink** (chained) rather than swallowing them behind a generic native error.
- **`slice_d_folder` now closes its SQLite connections** (via `contextlib.closing`).
- Corrected docstrings that contradicted the code: `scan_num_end` is exclusive (`[begin, end)`), `DIA.windows` indexes by window *group*, and `Precursor.peaks` / `PasefFrameMsmsInfo.peaks` are documented as native-centroided 2-D arrays (vs the raw per-scan lists returned by `Frame`/`DiaWindow`).

## [2.0.0] - 2026-06-08

### Added

- **Pipeline module (`tdfpy.pipeline`).** Composable ops for raw peak extraction and centroiding — `RawSpectrum` data carrier plus `read_spectrum`, `subset_scans`, `exclude_region`, `smooth`, `box_smooth`, `apply_noise`, `convert`, `centroid_peaks`. The convenience `get_raw_peaks` and `get_centroided_spectrum` are now thin orchestrators over these ops; custom pipelines call the ops directly. See [docs/api/pipeline.md](docs/api/pipeline.md).
- **`smooth` / `box_smooth` pipeline ops + `Smooth` config.** Position-preserving box sum or mean of intensities over a `(±scan_half_width, ±mz_idx_half_width)` window (default `±5` scans, `±2` TOF indices, `mode="sum"`). Summing amplifies genuine ion-mobility streaks ahead of noise filtering while leaving scattered single-hit noise unchanged; the `mode="mean"` variant backs `WatershedCentroider`'s seed-stabilising smoother. Vectorised prefix-sum implementation, promoted from the tuning dashboards. The frozen `Smooth` dataclass carries the knobs so the convenience entry points accept `smooth=Smooth(...)` — wired into `get_raw_peaks`, `get_centroided_spectrum`, and `Frame`/`DiaWindow`/`PrmTransition` `.raw_peaks()` / `.centroid()`.
- **Noise filter subpackage (`tdfpy.noise`).** `NoiseFilter` ABC plus per-method `IntensityThreshold` subclasses (`AbsoluteThreshold`, `MadThreshold`, `PercentileThreshold`, `HistogramThreshold`, `BaselineThreshold`, `IterativeMedianThreshold`) and the structural filters `VerticalNoiseFilter` and `HorizontalHaloFilter`. Each filter exposes its tunable knobs as dataclass fields; frozen so they're hashable (Streamlit-cacheable). String/numeric shorthand (`noise="mad"`, `noise=500.0`) coerced via `coerce_filters`. Filters compose: `noise=[VerticalNoiseFilter(...), HorizontalHaloFilter(), MadThreshold(k=3)]`.
- **`HorizontalHaloFilter`.** Structural filter that clears the weak m/z halo flanking bright peaks **to the left and right only — never above or below.** Operating in integer `(scan, TOF index)` space, it compares each peak to the maximum intensity in its surrounding box `(±scan_half_width, ±mz_idx_half_width)` *excluding the peak's own m/z column* and drops it if it falls below `peak_fraction` of that reference. Excluding the own column is what guarantees a bright peak directly above/below (the vertical ion-mobility streak of a real ion) can never trigger removal. Defaults `peak_fraction=0.15`, `mz_idx_half_width=100` (≈0.4 Da), `scan_half_width=2`; set `scan_half_width=0` for strictly per-row behaviour.
- **`VerticalNoiseFilter` Numba kernel.** The single-pass vertical-streak scan is now JIT-compiled (`@njit(cache=True)`) with a forward two-pointer window and an incremental per-scan intensity profile; the pure-NumPy reference is retained as the fallback and for the diagnostics histogram. Behaviour is unchanged (covered by an equivalence test).
- **Region exclusion (`tdfpy.regions`).** New `ChargeStateRegion` dataclass for dropping the singly-charged contamination band in timsTOF MS1 — defined by a `(m/z, 1/K0)` line, capped at the upper endpoint. Applied in integer TOF-index space (one vectorized comparison, no per-peak unit conversion). Distinct from noise filters — answers "which part of the data plane?" rather than "what's real signal?". See [docs/api/regions.md](docs/api/regions.md).
- **Centroider hierarchy.** New `Centroider` ABC with two implementations:
  - `MergePeaksCentroider` (default, replaces `CentroidConfig`) — wraps the existing greedy tolerance-based centroider.
  - `WatershedCentroider` — intensity-ordered region growing in integer index space, ported from `apps/ALGORITHM.md` Stage 3. Avoids float-m/z binning; Numba-JIT'd kernel (~2.5× faster than pure Python on real frames).
- **Scan-range subsetting.** New `scan_range=(begin, end)` parameter on `get_raw_peaks` / `get_centroided_spectrum`, exposed automatically by `DiaWindow.centroid()` and `PrmTransition.centroid()` — fixes a long-standing bug where those methods centroided the entire parent frame instead of just the isolation window.
- **`VerticalNoiseDiagnostics`** returned by `VerticalNoiseFilter.run(..., diagnostics=True)`. Carries the keep-mask plus per-pass attrition trace, column counts, and feature-intensity histogram — used by the IM-feature-filter dashboard.
- **PEP 561 `py.typed` marker.** Downstream type checkers now pick up tdfpy's annotations.

### Changed

- **`get_centroided_spectrum`, `get_raw_peaks`, `Frame.centroid()`, `Frame.raw_peaks()`, `DiaWindow.centroid()`, `PrmTransition.centroid()` API.** Old kwargs collapsed into the new composable system. **Breaking** for the affected call sites:

  | Old | New |
  |---|---|
  | `noise_filter="mad" \| float \| None` | `noise="mad" \| MadThreshold(k=3) \| 500.0 \| [filters] \| None` |
  | `min_intensity="mad" \| float \| None` (on `get_raw_peaks`) | `noise=…` (same as above) |
  | `ms1_filter=((350.0, 0.7), (1200.0, 1.4))` | `exclude=ChargeStateRegion()` (defaults to that line) |
  | `centroid=CentroidConfig(mz_tolerance=10)` | `centroid=MergePeaksCentroider(mz_tolerance=10)` |
  | post-centroid `noise_filter` on `get_centroided_spectrum` | **dropped** — chain filters via `noise=` (pre-centroid) instead |
  | `Frame.centroid(mz_tolerance=10, ...)` flat kwargs | `Frame.centroid(centroid=MergePeaksCentroider(mz_tolerance=10, ...))` |
- **`DiaWindow.centroid()` / `PrmTransition.centroid()` now honor `scan_num_begin/end`** — previously they centroided the whole parent frame. Output will differ for these classes; MS1 (`Frame.centroid()`) is unaffected.
- **`apps/_im_filter.py` removed.** The vertical-noise filter algorithm is now canonical in `tdfpy.noise.structural`; the timsTOF viewer uses `VerticalNoiseFilter.run(..., diagnostics=True)`.
- **`VerticalNoiseFilter` / `WatershedCentroider` field names normalized.** Every Chebyshev half-extent on the scan or TOF-index axis now follows `<purpose>_<axis>_half_width` (`mz_idx_half_width`, `attach_scan_half_width`, `smooth_scan_half_width`, `attach_mz_idx_half_width`, `smooth_mz_idx_half_width`). Streak length / gap fields renamed to `min_streak_scans` / `max_gap_scans` / `min_streak_intensity`. `min_centroid_total` renamed to `min_centroid_intensity`. **Breaking** for any code that constructed these dataclasses by keyword.
- **`VerticalNoiseFilter` defaults shifted.** `mz_idx_half_width` now `3` (was `2`), `min_streak_intensity` now `50.0` (was `0.0`), `num_iterations` now `2` (was `1`). Pass explicit values to keep the previous behaviour.
- **`WatershedCentroider` defaults shifted.** Box smoothing is now on by default — `smooth_scan_half_width=5`, `smooth_mz_idx_half_width=3` (previously `0`, off). `max_mz_idx_from_seed` now defaults to `10` (previously `None`, unbounded). Pass `smooth_scan_half_width=0` and `max_mz_idx_from_seed=None` to disable.

### Removed

- **`tdfpy.noise.estimate_noise_level`** (single-file module). Replaced by the `tdfpy.noise` subpackage's `IntensityThreshold` subclasses + `coerce_filters`. The five string-method names (`"mad"`, `"percentile"`, `"histogram"`, `"baseline"`, `"iterative_median"`) still work as shorthand wherever `noise=` is accepted.
- Post-centroid noise filtering on `get_centroided_spectrum` — noise filters now run pre-centroid via `noise=`, where they more usefully suppress satellites before the centroider sees them.
- **Convolution-style smoothing.** The old expand-and-aggregate `smooth()` op and the `im_smoothing_window` / `mz_smoothing_window` kwargs on `get_raw_peaks` / `get_centroided_spectrum` / `Frame.raw_peaks()` / `Frame.centroid()` are gone — it created new positions whenever the window > 1, ballooning the point count. It is replaced by the position-preserving box-sum/mean `smooth` / `box_smooth` ops (see Added).
- **`VerticalNoiseFilter.min_window_intensity`** field. Was a per-scan summed-intensity floor inside the column window — confusingly named and rarely tuned (defaulted to 0). The `min_streak_intensity` total-intensity floor is the kept knob.
- **Standalone tuning dashboards** (`apps/im_feature_filter_dashboard.py`, `apps/merged_frames_dashboard.py`, `apps/raw_spectrum_dashboard.py`, `apps/raw_ms2_dashboard.py`). Their algorithms graduated into the package (`smooth` / `box_smooth`, `HorizontalHaloFilter`, `VerticalNoiseFilter`, `WatershedCentroider`); the multi-page `apps/timstof_viewer/` app is the single remaining dev tool and now calls those package APIs directly. `apps/ALGORITHM.md` is retained as the algorithm reference.

### Fixed

- DIA / PRM centroiding now scopes to the isolation window's scan range. Previously every `DiaWindow.centroid()` and `PrmTransition.centroid()` call read the whole parent frame.

## [1.2.0]

### Added
- `viz` module with `plot_centroiding()` for visual inspection of centroiding quality (2x2 panel: raw peaks, centroids, noise-rejected, 1D spectrum comparison).
- `plot_centroiding` exported from the top-level `tdfpy` namespace.
- Example DDA and PRM datasets under `tests/data/`.

### Changed
- Centroiding now uses dynamic IM region growing instead of a fixed-radius tolerance, improving peak grouping for adjacent ion mobility peaks.
- Default `im_tolerance` changed from `0.05` to `0.1` across all centroiding functions and `centroid()` methods.

### Fixed
- Centroiding edge cases where floating-point precision caused peaks at the exact tolerance boundary to be excluded.
- `timsdata.dll` now included in wheel artifacts for Windows support.
- Validated SQL table names in `convert_table_to_df()` against `TableNames` enum.
- Replaced `assert` with explicit `ValueError` in centroiding data integrity check.
- Narrowed overly broad `except Exception` in Numba fallback paths.
- Improved error message for unsupported platforms in `timsdata.py`.
- Added minimum version bounds (`>=2.0`) for numpy and pandas dependencies.

## [1.1.0]

### Added
- PRM (Parallel Reaction Monitoring) acquisition mode support: new `PRM` reader class, `PrmTarget`, `PrmTransition`, and `PRMMs1Frame` data elements, and `PrmTargetLookup` / `PrmTransitionLookup` lookup tables.
- Comprehensive DIA acquisition mode support: `DIAMs1Frame`, `DiaWindow`, and `DiaWindowGroup` data elements, with `DIA` reader exposing per-window MS1/MS2 access.
- `slice_d_folder` utility (`tdfpy.slicer`) for extracting a frame-range subset of a Bruker `.d` folder into a new `.d` folder, including the `FrameProperties` table.
- Example DIA dataset under `tests/data/example_dia.d/` and tests covering DIA, PRM, and slicer behavior (`tests/test_dia.py`, `tests/test_prm.py`, `tests/test_slicer.py`).
- Documentation pages `docs/api/prm.md` and `docs/utilities.md`, plus expanded `docs/api/centroiding.md`, `docs/api/lookup.md`, `docs/getting-started.md`, and `README.md`.

### Changed
- `slice_d_folder` now overwrites an existing destination directory rather than failing.

## [1.0.2]

### Fixed
- `libtimsdata.so` is now committed to git and negated in `.gitignore` so it is present during CI checkout and correctly included in the PyPI wheel.
- Added wheel content check in CI to assert both `libtimsdata.so` and `timsdata.dll` are present in every built wheel.
- `metadata` property was incorrectly loading `CalibrationInfo` instead of `GlobalMetadata`; `calibration` property was also missing the key-indexed Series transformation.
- Suppressed `ty` `unresolved-import` diagnostic for optional `numba` dependency.

### Added
- Unit tests for `constants.py` (`PROTON_MASS`, `TableNames`), `noise.py` (all five `estimate_noise_level` strategies), `elems.py` (`MsMsType`, `Polarity.from_str`, `DiaWindowGroup` properties), and `centroiding.py` (`calculate_nmass`, `batch_iterator`).
- `test_metadata.py` with 62 tests covering `MetaData` and `Calibration` properties, using a single module-scoped `DDA` fixture for fast execution.

### Changed
- `test_metadata.py` refactored to share a single `DDA` instance across all 62 tests (module-scoped fixture) instead of opening a new connection per test.
- `just test-cov` now excludes `test_docs.py` from coverage measurement.
- README expanded with overview, quick-start examples, lookup/query usage, and centroiding parameter documentation.

## [1.0.1]

### Fixed
- `libtimsdata.so` (Linux Bruker native library) was missing from the PyPI wheel due to a `*.so` entry in `.gitignore`; fixed by declaring it as a hatchling build artifact.

## [1.0.0]

### Added
- New high-level object-oriented API for interacting with TDF data (`DDA`, `DIA`, `PRM` classes)
- `reader.py` module containing the new class hierarchy
- Support for extracting Precursor and PASEF MS/MS data via object properties
- `get_centroided_spectrum` function returning high-performance 2D numpy arrays

### Changed
- **BREAKING**: `merge_peaks` now returns a `numpy.ndarray` of shape (N, 3) (mz, intensity, mobility) instead of a list of `Peak` objects.
- **BREAKING**: Refactored spectrum extraction to return pure numpy arrays instead of `Ms1Spectrum` objects for improved performance.
- `Ms1Spectrum` and `Peak` NamedTuples are replaced by raw numpy array access in the high-level API.
- Replaced Rust extension (`_tdfpy_rust`) with Numba JIT-compiled centroiding backend; no Rust toolchain required.
- `merge_peaks()` and `get_centroided_spectrum()` parameter `use_rust` renamed to `use_numba`.

## [0.3.0]
- Rust backend for centroiding (11x faster)
- uv / ty / ruff
- python 3.11 +
- namesspace (import tdfpy as td)

## [0.2.0]

### Added
- High-level API with `Peak` and `Ms1Spectrum` NamedTuples
- `get_centroided_ms1_spectrum()` and `get_centroided_ms1_spectra()` functions
- `merge_peaks()` for peak centroiding with m/z and ion mobility tolerances
- Noise filtering module (`noise.py`) with `estimate_noise_level()` function
- CCS support via `ion_mobility_type` parameter ("ook0" or "ccs")
- Type annotations throughout (Python 3.8+)
- Test suite with test data included
- Modern build system using `pyproject.toml` and `uv`
- Logging support

### Changed
- Migrated to src-based layout
- Generator-based API for memory efficiency
- High-level API returns retention time in minutes
- Relaxed dependency version requirements

## [0.1.7]

### Added
- PRM (Parallel Reaction Monitoring) related database tables
- `is_dda` and `is_prm` properties to distinguish acquisition modes
- GitHub Actions workflows for pytest and pylint

## [0.1.6]

### Changed
- Updated numpy and pandas version requirements

### Removed
- Unicode import from numpy (deprecated)

## [0.1.3]

### Added
- Logging support throughout the package
- Test data moved into repository for easier testing
- Updated numpy and pandas dependencies

## [0.1.2]

### Added
- Context manager support (`timsdata_connect()`) for automatic resource cleanup
- `with` statement support for `TimsData` class

## [0.1.0]

Initial release with basic functionality.

### Added
- `TimsData` class for low-level access to Bruker `.tdf` and `.tdf_bin` files
- `PandasTdf` class for DataFrame interface to SQLite metadata
- ctypes bindings to Bruker's native libraries
- Cross-platform support (Windows DLL, Linux SO)
- Basic reading of frames, scans, and PASEF MS/MS data
