
<div align="center">
  <img src="https://raw.githubusercontent.com/tacular-omics/tdfpy/main/logo.png" alt="TDFpy Logo" width="400" style="margin: 20px;"/>

  A Python package for extracting data from Bruker timsTOF data files (.tdf and .tdf_bin). Includes a Numba-accelerated centroiding algorithm for efficient extraction of ion mobility data.

  [![Python package](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml)
  [![codecov](https://codecov.io/gh/tacular-omics/tdfpy/graph/badge.svg?token=RMUiW11IR2)](https://codecov.io/gh/tacular-omics/tdfpy)
  [![PyPI version](https://badge.fury.io/py/tdfpy.svg)](https://badge.fury.io/py/tdfpy)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview

tdfpy provides a high-level Python API for reading Bruker timsTOF `.d` folders. It handles DDA, DIA, and PRM acquisition modes and exposes familiar Python objects — no need to think about raw PASEF frames or SQLite queries.

- **DDA** — iterate MS1 frames and precursors (MS2 spectra)
- **DIA** — iterate MS1 frames and DIA isolation windows
- **Centroiding** — Numba-accelerated peak merging across the m/z and ion mobility dimensions, returning `(N, 3)` arrays of `[m/z, intensity, 1/K0]`
- **Lazy spectral access** — frame metadata is loaded upfront; raw peak data is only read when you call `.peaks` or `.centroid()`

## Installation

```bash
pip install tdfpy
```

Requires Python 3.12+. The Bruker `libtimsdata` native library is bundled in the wheel (Linux).

## Quick Start

```python
from tdfpy import DDA, DIA

# DDA acquisition
with DDA("sample.d") as dda:
    for frame in dda.ms1:
        peaks = frame.centroid()  # shape (N, 3): [m/z, intensity, 1/K0]

    for precursor in dda.precursors:
        print(precursor.largest_peak_mz, precursor.charge)
        peaks = precursor.peaks  # centroided MS2 via Bruker's algorithm

# DIA acquisition
with DIA("sample.d") as dia:
    for frame in dia.ms1:
        peaks = frame.centroid()

    for window in dia.windows:
        print(window.isolation_mz, window.isolation_width)
        peaks = window.centroid()
```

## Lookups and Queries

Frames, precursors, and windows can be accessed by ID or queried by m/z and retention time:

```python
with DDA("sample.d") as dda:
    frame = dda.ms1[1]           # by frame ID
    precursor = dda.precursors[1]  # by precursor ID

    # query by m/z and RT window
    hits = dda.precursors.query(
        mz=1292.63,
        mz_tolerance=20.0,   # ppm
        rt=2400.0,           # seconds
        rt_tolerance=30.0,
    )
```

## Centroiding Options

`frame.centroid()` and `window.centroid()` accept parameters to control the peak merging:

```python
peaks = frame.centroid(
    mz_tolerance=8,               # ppm (default)
    mz_tolerance_type="ppm",      # or "da"
    im_tolerance=0.05,            # relative (default)
    im_tolerance_type="relative", # or "absolute"
    min_peaks=3,                  # minimum raw peaks to form a centroid
    noise_filter="mad",           # optional: "mad", "percentile", "histogram", etc.
    ion_mobility_type="ook0",     # or "ccs" / "voltage"
)
```

You can also call `merge_peaks` directly on your own arrays:

```python
from tdfpy import merge_peaks
import numpy as np

peaks = merge_peaks(mz_array, intensity_array, ion_mobility_array, mz_tolerance=10)
```

## Documentation

Full documentation at [tacular-omics.github.io/tdfpy](https://tacular-omics.github.io/tdfpy/)
