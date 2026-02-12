# TDFpy

A Python package for extracting data from Bruker timsTOF data files (`.tdf` and `.tdf_bin`).

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
    for frame in dda.ms1_frames:
        print(f"Frame {frame.frame_id} at {frame.time}s")
        centroided_peaks = frame.centroid()

    # Iterate over precursors
    for precursor in dda.precursors:
        print(f"Precursor {precursor.precursor_id}: {precursor.largest_peak_mz} m/z")
        # Raw peaks for precursor (no centroiding needed)
        peaks = precursor.peaks
```

## Features

- **Simple Context Managers**: `DIA` and `DDA` classes handle file connections safely.
- **Easy Iteration**: Generators for frames, windows, and precursors.
- **Centroiding**: Built-in `centroid()` method for frames and windows.
- **Metadata Access**: Via `metadata` and `calibration` properties.
- **Rust Backend**: Performance-critical operations are optimized with Rust.

## Development

The project uses `uv` for dependency management.

```bash
make install-dev    # Install dev dependencies
make test          # Run tests
make lint          # Run type checking and linting
make build         # Build package
```
