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
- **MS1 Peak centroiding**: 2D Peak merging with m/z and ion mobility tolerances (Rust-backend)

## Installation

### From PyPI
```bash
pip install tdfpy
```


## Quick Start

### Read Your First Spectrum

```python
import tdfpy

# Open a .d folder and read a spectrum
with tdfpy.timsdata_connect('data.d') as td:
    spectrum = tdfpy.get_centroided_ms1_spectrum(td, frame_id=1)
    
print(f"{spectrum.num_peaks} peaks at RT {spectrum.retention_time:.2f} min")
# Output: 1523 peaks at RT 0.05 min

# Access peaks
for peak in spectrum.peaks[:3]:
    print(f"m/z {peak.mz:.2f}, intensity {peak.intensity:.0f}")
# Output:
# m/z 301.14, intensity 15234.0
# m/z 524.26, intensity 8901.0
# m/z 785.42, intensity 12456.0
```

### Process All MS1 Spectra

```python
import tdfpy

with tdfpy.timsdata_connect('data.d') as td:
    for spectrum in tdfpy.get_centroided_ms1_spectra(td):
        print(f"Frame {spectrum.frame_id}: {spectrum.num_peaks} peaks")
```

### Read Specific Frames

```python
import tdfpy

with tdfpy.timsdata_connect('data.d') as td:
    # Process frames 10-20
    spectra = tdfpy.get_centroided_ms1_spectra(td, frame_ids=range(10, 21))
    
    for spectrum in spectra:
        total_intensity = sum(peak.intensity for peak in spectrum.peaks)
        print(f"Frame {spectrum.frame_id}: Total intensity {total_intensity:.2e}")
```

## Advanced Examples

### Customize Peak Centroiding

```python
import tdfpy

with tdfpy.timsdata_connect('data.d') as td:
    spectrum = tdfpy.get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        mz_tolerance=15,              # PPM tolerance (default: 8)
        im_tolerance=0.05,            # Ion mobility tolerance (default: 0.05)
        min_peaks=5,                  # Min peaks to merge (default: 3)
        noise_filter="mad"            # Remove noise peaks
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
### Noise Filtering

```python
import tdfpy

with tdfpy.timsdata_connect('data.d') as td:
    # Automatic noise removal
    spectrum = tdfpy.get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
        noise_filter="mad"  # Recommended: Median Absolute Deviation
    )
    # Other methods: "percentile", "histogram", "baseline", "iterative_median"
    
    # Manual threshold
    spectrum = tdfpy.get_centroided_ms1_spectrum(
        td, 
        frame_id=1,
### Database Metadata (Pandas)

```python
import tdfpy

pd_tdf = tdfpy.PandasTdf('data.d/analysis.tdf')

# Access metadata tables
frames_df = pd_tdf.frames          # Frame info (RT, scans, etc.)
precursors_df = pd_tdf.precursors  # MS2 precursor info
properties_df = pd_tdf.properties  # Instrument settings

# Filter MS1 frames
ms1_frames = frames_df[frames_df['MsMsType'] == 0]
print(f"Found {len(ms1_frames)} MS1 frames")
print(f"RT range: {ms1_frames['Time'].min():.2f} - {ms1_frames['Time'].max():.2f} sec")
```     print(f"m/z {peak.mz:.2f}, CCS {peak.ion_mobility:.2f} Ų")
```nt(f"Found {len(ms1_frames)} MS1 frames")
```
### Low-Level API (Advanced)

Direct access to Bruker's native library:

```python
import tdfpy

with tdfpy.timsdata_connect('data.d') as td:
    # Read raw profile data
    scans = td.readScans(frame_id=1, scan_begin=0, scan_end=1000)
    
    # Each scan returns (index_array, intensity_array)
    indices, intensities = scans[500]
    
    # Convert indices to m/z
    mz_values = td.indexToMz(frame_id=1, indices=indices)
    
    # Convert scan number to ion mobility
    mobility = td.scanNumToOneOverK0(frame_id=1, scan_nums=[500])
``` msms_data = td.readPasefMsMsForFrame(frame_id)
```

## API Reference

### Key Functions

- **`timsdata_connect(path)`** - Open a .d folder (context manager)
- **`get_centroided_ms1_spectrum(td, frame_id, ...)`** - Read single spectrum
- **`get_centroided_ms1_spectra(td, frame_ids=None, ...)`** - Read multiple spectra (generator)

### Data Types

**`Peak`** (NamedTuple):
- `mz: float` - Mass-to-charge ratio
- `intensity: float` - Peak intensity
- `ion_mobility: float` - Ion mobility (1/K0 or CCS)

**`Ms1Spectrum`** (NamedTuple):
- `spectrum_index: int` - Sequential index
- `frame_id: int` - Frame ID from TDF
- `retention_time: float` - RT in minutes
- `peaks: List[Peak]` - Peak list
- `ion_mobility_type: str` - "ook0" or "ccs"
- `num_peaks: int` - Number of peaks (property)

## Development
## Performance

TDFpy includes a Rust-accelerated backend for peak merging:
- **11x faster** than pure Python implementation
- Processes 1.7M peaks in ~250ms
- Automatic fallback to Python if Rust unavailable

## Development

Uses `uv` for package management:

```bash
make install-dev    # Install with dev dependencies
make test          # Run tests
make lint          # Run ty + ruff
make build         # Build package
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

## Architecture

Three-layer design:

1. **Native**: Bruker's `timsdata.dll`/`.so` (included)
2. **Low-level**: ctypes bindings (`timsdata.py`)
3. **High-level**: Pythonic API with NamedTuples (`spectra.py`, `noise.py`)
- All tests pass (`make test`)
- Code passes linting (`make lint`)
- Type hints are included
- Docstrings follow Google style
## Contributing

Contributions welcome! Requirements:
- Tests pass: `make test`
- Linting passes: `make lint`
- Type hints included
- Docstrings for public APIs
  title = {TDFpy: Python parser for Bruker timsTOF data},
  url = {https://github.com/pgarrett-scripps/tdfpy},
  version = {0.2.0}
}
```


