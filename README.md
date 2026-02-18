
<div align="center">
  <img src="logo.png" alt="TDFpy Logo" width="400" style="margin: 20px;"/>
  
    A Python package for extracting data from Bruker timsTOF data files (.tdf and .tdf_bin). Includes a Rust-backed centroiding algorithm for efficient extraction of ion mobility data.
  
  [![Python package](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml)
  [![codecov](https://codecov.io/github/tacular-omics/tdfpy/graph/badge.svg?token=1CTVZVFXF7)](https://codecov.io/github/tacular-omics/tdfpy)
  [![PyPI version](https://badge.fury.io/py/tdfpy.svg)](https://badge.fury.io/py/tdfpy)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
  
</div>

## TDFpy

TDFpy simplifies this process by providing an API that works with familiar objects:

- DDA: MS1 spectra and precursors (MS2 spectra)
- DIA: MS1 spectra and DIA windows

(No need to think about PASEF frames)

**MS1 Spectra**

- MS1 objects include a Rust-backed centroiding function that returns a 3D NumPy array containing m/z, intensity, and 1/k0 values.

**Precursors (DDA)**

- Precursors are already centroided using Bruker's built-in C extensions.

**Windows (DIA)**

- DIA windows also have access to the centroiding function. However, be careful not to confuse the window's ion mobility dimension with that of the fragment ions. The TIMS components within the instrument are positioned before the fragmentation cell, meaning the ion mobility reported in DIA frames actually corresponds to the precursor ions from the MS1 frame, not the fragment ions.

## Installation

```bash
pip install tdfpy
```

## Quick Start

### DIA Data

```python
from tdfpy import DIA

# Open a DIA .d folder
with DIA('data.d') as dia:
    # Iterate over MS1 frames
    for frame in dia.ms1_frames:
        print(f"Frame {frame.frame_id} at {frame.time}s")
        # Get centroided peaks (m/z, intensity, ion mobility)
        centroided_peaks = frame.centroid()
        print(f"Peaks shape: {centroided_peaks.shape}")

    # Iterate over DIA windows
    for window in dia.windows:
        print(f"Window {window.window_id}: {window.isolation_mz} m/z")
        peaks = window.centroid()
```

### DDA Data

```python
from tdfpy import DDA

# Open a DDA .d folder
with DDA('data.d') as dda:
    # Iterate over MS1 frames
    for frame in dda.ms1:
        print(f"Frame {frame.frame_id} at {frame.time}s")
        centroided_peaks = frame.centroid()

    # Iterate over precursors
    for precursor in dda.precursors:
        print(f"Precursor {precursor.precursor_id}: {precursor.largest_peak_mz} m/z")
        # Raw peaks for precursor (no centroiding needed)
        peaks = precursor.peaks
```

### Lookups and Queries

Access specific frames, precursors, or windows directly by ID, or query by properties.

```python
# Access by ID
frame = dda.ms1[1]          # Get MS1 frame 1
precursor = dda.precursors[123]    # Get precursor 123
windows = dia.windows[0]           # Get windows for group 0 (returns list)

# Query Precursors (DDA)
# Find precursors near a specific m/z and retention time
results = dda.precursors.query(
    mz=1292.63, 
    mz_tolerance=0.01, 
    rt=2400.0, 
    rt_tolerance=10.0
)
```

## Development

The project uses `uv` for dependency management.

```bash
just install

# QC
just lint
just format
just ty
just test

# or run all QC commands:
just check
```
