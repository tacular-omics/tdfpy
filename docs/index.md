# tdfpy

[![Python package](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/tacular-omics/tdfpy/graph/badge.svg?token=RMUiW11IR2)](https://codecov.io/gh/tacular-omics/tdfpy)
[![PyPI version](https://badge.fury.io/py/tdfpy.svg)](https://badge.fury.io/py/tdfpy)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

A Python package for extracting data from Bruker timsTOF data files (`.tdf` and `.tdf_bin`). Includes a Numba-accelerated centroiding algorithm for efficient extraction of ion mobility data.

## Overview

tdfpy provides an API that works with familiar objects — no need to think about PASEF frames.

- **DDA** — MS1 spectra and precursors (MS2 spectra)
- **DIA** — MS1 spectra and DIA windows
- **PRM** — MS1 spectra, targets, and transitions
- **MALDI** — Work in progress

**MS1 Spectra** — MS1 objects include a Numba-accelerated centroiding function that returns a 3D NumPy array containing m/z, intensity, and 1/K0 values.

**Precursors (DDA)** — `precursor.peaks` returns an MS2 peak list centroided by tdfpy itself: intensities are summed per TOF index across the precursor's scan range (collapsing ion mobility) and merged at 30 ppm.

**Windows (DIA)** — DIA windows also have access to the centroiding function. Note that the ion mobility dimension in DIA frames corresponds to precursor ions from the MS1 frame, not fragment ions (TIMS components are positioned before the fragmentation cell).

## Quick Example

```python
from tdfpy import DDA, DIA, PRM

# DDA acquisition
with DDA('data.d') as dda:
    for frame in dda.ms1:
        peaks = frame.centroid()  # shape (N, 3): m/z, intensity, 1/K0

    for precursor in dda.precursors:
        print(precursor.largest_peak_mz, precursor.peaks)

# DIA acquisition
with DIA('data.d') as dia:
    for frame in dia.ms1:
        peaks = frame.centroid()

    for window in dia.windows:
        peaks = window.centroid()

# PRM acquisition
with PRM('data.d') as prm:
    for target in prm.targets:
        print(target.monoisotopic_mz, target.charge)

    for transition in prm.transitions:
        peaks = transition.peaks  # shape (N, 2): m/z, intensity
```

## Installation

```bash
pip install tdfpy
```

See [Getting Started](getting-started.md) for a full walkthrough.
