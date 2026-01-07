```
 ███████████ ██████████   ███████████                     
░█░░░███░░░█░░███░░░░███ ░░███░░░░░░█                     
░   ░███  ░  ░███   ░░███ ░███   █ ░  ████████  █████ ████
    ░███     ░███    ░███ ░███████   ░░███░░███░░███ ░███ 
    ░███     ░███    ░███ ░███░░░█    ░███ ░███ ░███ ░███ 
    ░███     ░███    ███  ░███  ░     ░███ ░███ ░███ ░███ 
    █████    ██████████   █████       ░███████  ░░███████ 
   ░░░░░    ░░░░░░░░░░   ░░░░░        ░███░░░    ░░░░░███ 
                                      ░███       ███ ░███ 
                                      █████     ░░██████  
                                     ░░░░░       ░░░░░░   
```


A Python package for parsing Bruker timsTOF data files (`.tdf` and `.tdf_bin`) with both low-level and high-level APIs.

TDFpy provides efficient access to mass spectrometry data from Bruker timsTOF instruments, including:
- High-level API for centroided MS1 spectra with peak merging
- Low-level ctypes bindings to Bruker's native libraries
- Pandas DataFrame interface to SQLite metadata tables

## Features

- **Three-layer architecture**: Native DLL → ctypes wrapper → Pythonic API
- **Memory-efficient**: Generator-based API for processing large datasets
- **Type-safe**: Full type hints throughout (Python 3.8+)
- **Cross-platform**: Supports Windows (`.dll`) and Linux (`.so`)
- **Peak centroiding**: 2D Peak merging with m/z and ion mobility tolerances (Rust-backend)

## Installation

### From PyPI
```bash
pip install tdfpy
```

### From source (using uv)
```bash
git clone https://github.com/pgarrett-scripps/tdfpy.git
cd tdfpy
uv pip install -e .
```

## Quick Start Examples

### High-Level API (Recommended for MS1 Spectra)

The high-level API provides centroided MS1 spectra with automatic peak merging:

```python
from tdfpy import timsdata_connect, get_centroided_ms1_spectrum, get_centroided_ms1_spectra

# Read a single MS1 spectrum with centroiding
with timsdata_connect('path/to/data.d') as td:
    # Get first MS1 frame ID
    cursor = td.conn.cursor()
    cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1")
    frame_id = cursor.fetchone()[0]
    
    spectrum = get_centroided_ms1_spectrum(td, frame_id)
    print(f"Frame {spectrum.frame_id}: {spectrum.num_peaks} peaks at RT={spectrum.retention_time:.2f} min")

    # Access individual peaks
    for peak in spectrum.peaks[:5]:
        print(f"  m/z: {peak.mz:.4f}, intensity: {peak.intensity:.0f}, mobility: {peak.ion_mobility:.4f}")

# Read all MS1 spectra (returns generator for memory efficiency)
with timsdata_connect('path/to/data.d') as td:
    spectra_generator = get_centroided_ms1_spectra(td)
    
    # Process spectra one at a time
    for spectrum in spectra_generator:
        print(f"Spectrum {spectrum.spectrum_index}: {spectrum.num_peaks} peaks")
    
    # Or convert to list (caution: may use significant memory)
    # spectra_list = list(get_centroided_ms1_spectra(td, frame_ids=[1, 2, 3]))
```

### Customizing Peak Merging

Control peak centroiding behavior with tolerance parameters:

```python
from tdfpy import timsdata_connect, get_centroided_ms1_spectrum

with timsdata_connect('path/to/data.d') as td:
    # Custom centroiding tolerances
    spectrum = get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        mz_tolerance=15,              # m/z tolerance in ppm
        mz_tolerance_type="ppm",      # or "da" for daltons
        im_tolerance=0.05,            # ion mobility tolerance (relative)
        im_tolerance_type="relative", # or "absolute"
        min_peaks=3                   # minimum nearby peaks to keep
    )
    
    # With noise filtering
    spectrum = get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        noise_filter="mad"            # or "percentile", "histogram", "baseline", "iterative_median"
    )
    
    # Get spectrum with CCS values instead of 1/K0
    spectrum = get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        ion_mobility_type="ccs"       # returns CCS in Ų, default is "ook0" (1/K0)
    )
```

### Noise Filtering

Apply statistical noise filtering before centroiding:

```python
from tdfpy import timsdata_connect, get_centroided_ms1_spectrum, estimate_noise_level
import numpy as np

with timsdata_connect('path/to/data.d') as td:
    # Use built-in noise filtering methods
    spectrum = get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        noise_filter="mad"  # Median Absolute Deviation (recommended)
    )
    
    # Available methods:
    # - "mad": Median Absolute Deviation (robust to outliers)
    # - "percentile": 75th percentile threshold
    # - "histogram": Histogram mode-based estimation
    # - "baseline": Bottom quartile statistics
    # - "iterative_median": Iterative median filtering
    
    # Use custom threshold value
    spectrum = get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        noise_filter=1000.0  # Remove peaks below intensity 1000
    )
    
    # Estimate noise independently
    intensity_array = np.array([100, 150, 200, 1000, 5000])
    threshold = estimate_noise_level(intensity_array, method="mad")
    print(f"Estimated noise threshold: {threshold:.2f}")
```

### Low-Level Database Access

Access SQLite metadata tables as Pandas DataFrames:

```python
from tdfpy import PandasTdf

pd_tdf = PandasTdf('path/to/data.d/analysis.tdf')

# Access various metadata tables
frames_df = pd_tdf.frames          # Frame metadata (RT, scan range, etc.)
precursors_df = pd_tdf.precursors  # Precursor information for MS2
properties_df = pd_tdf.properties  # Global metadata properties

# Query specific frames
ms1_frames = frames_df[frames_df['MsMsType'] == 0]
print(f"Found {len(ms1_frames)} MS1 frames")
```

### Low-Level Binary Data Access

Direct access to Bruker's native library functions:

```python
from tdfpy import timsdata_connect

with timsdata_connect('path/to/data.d') as td:
    # Read raw scan data for a frame
    frame_id = 1
    scan_data = td.readScans(frame_id, 0, 1000)  # (frame_id, start_scan, end_scan)
    
    # Convert detector indices to m/z values
    mz_values = td.indexToMz(frame_id, scan_data[0])  # scan_data[0] = indices
    
    # Convert scan numbers to ion mobility
    mobility = td.scanNumToOneOverK0(frame_id, 500)  # 1/K0 for scan 500
    
    # Read PASEF MS/MS data
    msms_data = td.readPasefMsMsForFrame(frame_id)
```

## Data Structures

### Peak (NamedTuple)
```python
Peak(
    mz: float,              # m/z value
    intensity: float,       # Peak intensity
    ion_mobility: float     # Ion mobility (1/K0 or CCS depending on extraction parameters)
)
```

### Ms1Spectrum (NamedTuple)
```python
Ms1Spectrum(
    spectrum_index: int,     # Sequential spectrum number
    frame_id: int,           # Frame ID from database
    retention_time: float,   # Retention time in minutes
    peaks: List[Peak],       # List of Peak objects
    ion_mobility_type: str   # Type of ion mobility values ("ook0" or "ccs")
)
```

## Development

This project uses `uv` for package management. See the `Makefile` for common commands:

```bash
make install-dev    # Install with dev dependencies
make test          # Run pytest tests
make lint          # Run mypy + pylint
make build         # Build package
make clean         # Remove build artifacts
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_spectra.py -v

# Run with coverage
make coverage
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- SQLite3 (standard library)
- Bruker's native timsTOF library (included in package)

## Architecture

TDFpy uses a three-layer architecture:

1. **Native DLL Layer**: Bruker's proprietary `timsdata.dll` (Windows) or `libtimsdata.so` (Linux)
2. **Low-Level Wrapper**: `timsdata.py` provides ctypes bindings to DLL functions
3. **High-Level API**: `spectra.py` provides Pythonic interface with NamedTuples and generators, `noise.py` provides noise estimation functions

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass (`make test`)
- Code passes linting (`make lint`)
- Type hints are included
- Docstrings follow Google style

## Citation

If you use TDFpy in your research, please cite the repository:
```
@software{tdfpy,
  author = {Garrett, Patrick},
  title = {TDFpy: Python parser for Bruker timsTOF data},
  url = {https://github.com/pgarrett-scripps/tdfpy},
  version = {0.2.0}
}
```


