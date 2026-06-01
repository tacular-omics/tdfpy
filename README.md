
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
- **PRM** — iterate targets and their transitions
- **Composable peak pipeline** — `read_spectrum` → optional region exclusion / smoothing / noise-filter chain → centroider. Returns `(N, 3)` `[m/z, intensity, 1/K0]` arrays
- **Two centroiders** — `MergePeaksCentroider` (default, Numba-JIT'd greedy merge in float m/z) and `WatershedCentroider` (intensity-ordered region growing in integer TOF-index space)
- **Composable noise filters** — chain `MadThreshold`, `VerticalNoiseFilter`, and others via `noise=[…]`. String shorthand (`noise="mad"`) preserved for terseness
- **Lazy spectral access** — frame metadata is loaded upfront; raw peak data is only read when you call `.peaks`, `.raw_peaks()`, or `.centroid()`

## Installation

```bash
pip install tdfpy
```

Requires Python 3.12+. The Bruker `libtimsdata` native library is bundled in the wheel (Linux).

## Quick Start

```python
from tdfpy import DDA, DIA, PRM

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

# PRM acquisition
with PRM("sample.d") as prm:
    for target in prm.targets:
        print(target.monoisotopic_mz, target.charge)

    for transition in prm.transitions:
        print(transition.isolation_mz, transition.collision_energy)
        peaks = transition.peaks  # shape (N, 2): [m/z, intensity]
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

## Peak extraction

Every frame-element method (`Frame.raw_peaks()`, `Frame.centroid()`, and the matching
methods on `DiaWindow` and `PrmTransition`) accepts the same composable arguments:

```python
from tdfpy import (
    ChargeStateRegion, MadThreshold, VerticalNoiseFilter,
    MergePeaksCentroider, WatershedCentroider,
)

peaks = frame.centroid(
    # Region exclusion: drop the singly-charged contamination band
    exclude=ChargeStateRegion(),

    # Noise filter pipeline (applied in order, pre-centroid)
    noise=[
        VerticalNoiseFilter(min_streak_scans=5, num_iterations=2),
        MadThreshold(k=3),
    ],

    # Centroider — swap algorithms without changing the surrounding code
    centroid=MergePeaksCentroider(mz_tolerance=8, mz_tolerance_type="ppm", min_peaks=3),
    # or:  WatershedCentroider(attach_scan_half_width=10, attach_mz_idx_half_width=3)
)
```

Terser shorthand still works for common cases:

```python
peaks = frame.centroid()                       # all defaults
peaks = frame.centroid(noise="mad")            # MAD-based intensity floor
peaks = frame.centroid(noise=500.0)            # absolute threshold
```

### Watershed centroider

The watershed centroider works in integer `(scan, TOF-index)` space — avoiding the
float-m/z binning that the greedy merger does. Useful when peaks are closely spaced
or when seed selection needs to be robust to noisy spikes:

```python
peaks = frame.centroid(
    centroid=WatershedCentroider(
        attach_scan_half_width=10, attach_mz_idx_half_width=3,
        min_seed_intensity=50.0,
        # Position-preserving intensity smoothing before seed selection (on by default)
        smooth_scan_half_width=5, smooth_mz_idx_half_width=3,
    ),
)
```

### Custom pipelines

For ordering or transformations beyond what the convenience methods cover, call the
[pipeline ops](https://tacular-omics.github.io/tdfpy/api/pipeline/) directly:

```python
from tdfpy import (
    read_spectrum, exclude_region, smooth, apply_noise, convert,
    ChargeStateRegion, MadThreshold,
)

s = read_spectrum(td, frame_id)
s = exclude_region(s, ChargeStateRegion(), td=td, frame_id=frame_id)
s = smooth(s, im_window=3)
s = apply_noise(s, (MadThreshold(k=3),), td=td, frame_id=frame_id)
peaks = convert(s, td, frame_id)
```

### Noise filtering vs `min_peaks`

Intensity-based noise filters (`MadThreshold`, `PercentileThreshold`, etc.) can be
too aggressive: they cannot distinguish low-abundance real signal from electronic
noise. Raising `min_peaks` on the centroider is often a more reliable filter, since
electronic noise typically manifests as singletons while real peaks appear across
multiple scans. The structural [`VerticalNoiseFilter`](https://tacular-omics.github.io/tdfpy/api/noise/)
extends this idea to the IM axis.

## Documentation

Full documentation at [tacular-omics.github.io/tdfpy](https://tacular-omics.github.io/tdfpy/)
