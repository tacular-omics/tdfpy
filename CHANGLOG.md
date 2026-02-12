# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# TODO:
- add lookup onjkects to DDA, DIA, and PRM readers for searching for ms1 or ms2 frames by id (similar to pymzml)
- implement PRM (ask sannie for sample data)

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
