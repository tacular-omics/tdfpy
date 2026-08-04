---
applyTo: "**"
---

# TDFpy - Bruker timsTOF Data Parser

TDFpy provides low-level and high-level APIs for reading Bruker `.tdf` and `.tdf_bin` files from timsTOF mass spectrometry instruments. It decodes the binary format directly in Python/NumPy, with no native library. 


## Updating README / Documentation / Changelog

When updating these files use neutral language. Avoid over the top adjectives, since most of the time the code is very mundane and not 'extraordinary'. Be straight to the point and factual. Documentation should be clear and concise, not flowery or embellished. Only explain further if necessary for clarity.

## Architecture Overview

### Three-Layer API Design
1. **Binary Layer** (`timsdata.py`): decodes `analysis.tdf_bin` frames (zstd + byte-plane de-interleave)
2. **Calibration Layer** (`calibration.py`): TOF index <-> m/z and scan <-> 1/K0 models from the SQLite tables
3. **High-Level Pythonic API** (`reader.py` / `elems.py`): acquisition-mode readers and frame elements

Key files:
- `src/tdfpy/timsdata.py`: pure-Python reader for `.d` folders
- `src/tdfpy/calibration.py`: calibration models (no I/O)
- `src/tdfpy/pandas_tdf.py`: DataFrame interface to SQLite metadata tables
- `src/tdfpy/spectra.py`: High-level MS1 spectrum extraction with centroiding
- `src/tdfpy/constants.py`: Database table names and physical constants

## Development Workflow

**Package Manager**: This project uses `uv` (not pip/poetry). See `Makefile` for all commands.

### Common Commands
```bash
make install-dev    # Install with dev dependencies
make test          # Run pytest tests
make lint          # Run mypy + pylint
make build         # Build package
make clean         # Remove build artifacts
```

### Testing
- Test data: `tests/data/200ngHeLaPASEF_1min.d/` (committed to repo)
- Run tests: `python -m unittest tests.test_<module> -v`
- **Important**: Never iterate over ALL MS1 spectra in tests - always LIMIT to 3-5 frames for performance
- `get_centroided_ms1_spectra()` returns a **generator**, not a list - convert with `list()` in tests

### Type Checking
- Project uses type hints throughout (Python 3.8+)
- `mypy` configured in `pyproject.toml` with relaxed settings for rapid development
- `timsdata.py` is ignored by pylint (auto-generated bindings)

## Critical Conventions

### Generator Pattern for Large Datasets
Functions that process multiple spectra/frames MUST return generators to avoid memory issues:
```python
def get_centroided_ms1_spectra(...) -> Generator[Ms1Spectrum, None, None]:
    for frame_id in frame_ids:
        yield get_centroided_ms1_spectrum(...)  # Yield, don't append
```

### NamedTuples for Data Structures
Prefer immutable NamedTuples over dataclasses for spectrum data:
```python
class Peak(NamedTuple):
    mz: float
    intensity: float
    ion_mobility: float  # 1/K0 in units

class Ms1Spectrum(NamedTuple):
    spectrum_index: int
    frame_id: int
    retention_time: float  # minutes
    peaks: List[Peak]
```

### Context Manager for Database Connections
Always use `timsdata_connect()` context manager to ensure proper cleanup:
```python
with timsdata_connect('path/to/data.d') as td:
    spectrum = get_centroided_ms1_spectrum(td, frame_id=1)
    # td.conn and td.handle auto-closed on exit
```

### Platform Support
Pure Python, so every platform is supported. Formats that have not been validated against Bruker's
library (legacy `TimsCompressionType` 1, unknown calibration `ModelType`s, pressure compensation)
raise rather than returning wrong numbers.

## Data Flow

1. **Bruker .d folder structure**:
   - `analysis.tdf`: SQLite database with metadata (frames, precursors, calibration)
   - `analysis.tdf_bin`: Binary blob with raw intensity/index arrays
   - frames are read from `analysis.tdf_bin` at the byte offset in `Frames.TimsId`

2. **Low-level access** (`TimsData`):
   - `readScans()`: Returns raw (index, intensity) arrays per scan
   - `indexToMz()`: Converts detector indices → m/z values using calibration
   - `scanNumToOneOverK0()`: Converts scan numbers → ion mobility (1/K0)

3. **High-level access** (`get_centroided_ms1_spectrum`):
   - Reads all scans in a frame
   - Converts indices to m/z
   - Applies peak merging with m/z and ion mobility tolerances
   - Returns clean `Ms1Spectrum` with `Peak` list

## Key Implementation Details

### Peak Merging Algorithm (spectra.py)
Greedy clustering that merges peaks within m/z AND ion mobility windows:
- Sort peaks by m/z for binary search
- Process in descending intensity order
- Use binary search to find m/z window, then filter by ion mobility
- Merge using intensity-weighted averages
- `min_peaks` parameter filters noise (default: 3 nearby peaks required)

### MS1 vs MS2 Frames
Query SQLite to distinguish frame types:
```sql
SELECT Id FROM Frames WHERE MsMsType = 0  -- MS1 frames
SELECT Id FROM Frames WHERE MsMsType != 0 -- MS2/PASEF frames
```

### Performance Patterns
- Pre-allocate NumPy arrays when total size known
- Batch operations: `td.indexToMz(frame_id, index_array)` not `[td.indexToMz(frame_id, i) for i in ...]`
- Use boolean masks for filtering, not list comprehensions
- Convert generator to list only when necessary

## Common Pitfalls

1. **Don't use `len()` on generators**: `get_centroided_ms1_spectra()` returns Generator, not List
2. **Always limit frame iteration in tests**: Full datasets can have 10K+ frames
3. **Check frame types**: Many queries require filtering `MsMsType = 0` for MS1-only
4. **Remember RT units**: Frames table stores seconds, API exposes minutes
5. **Ion mobility is 1/K0**: Not mobility itself - reciprocal of reduced mobility constant

## External Dependencies

- **NumPy**: All array operations, required for performance
- **Pandas**: DataFrame interface to SQLite tables (`PandasTdf` class)
- **SQLite3**: Standard library, reads `analysis.tdf` metadata
- **zstd**: `zstandard` on Python < 3.14, otherwise the stdlib `compression.zstd` module

## Version & Release

- Version in `src/tdfpy/__init__.py` (`__version__` attribute)
- Hatchling build system extracts version dynamically
- GitHub Actions: `python-package.yml` tests on Ubuntu/Windows × Python 3.8-3.11
- Publish: `make publish` (requires PyPI credentials)
