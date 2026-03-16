# tdfpy

[![Python package](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/tacular-omics/tdfpy/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/github/tacular-omics/tdfpy/graph/badge.svg?token=1CTVZVFXF7)](https://codecov.io/github/tacular-omics/tdfpy)
[![PyPI version](https://badge.fury.io/py/tdfpy.svg)](https://badge.fury.io/py/tdfpy)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

A Python package for extracting data from Bruker timsTOF data files (`.tdf` and `.tdf_bin`). Includes a Rust-backed centroiding algorithm for efficient extraction of ion mobility data.

## Overview

tdfpy provides an API that works with familiar objects — no need to think about PASEF frames.

- **DDA** — MS1 spectra and precursors (MS2 spectra)
- **DIA** — MS1 spectra and DIA windows
- **PRM** — Work in progress
- **MALDI** — Work in progress

**MS1 Spectra** — MS1 objects include a Rust-backed centroiding function that returns a 3D NumPy array containing m/z, intensity, and 1/K0 values.

**Precursors (DDA)** — Precursors are already centroided using Bruker's built-in C extensions.

**Windows (DIA)** — DIA windows also have access to the centroiding function. Note that the ion mobility dimension in DIA frames corresponds to precursor ions from the MS1 frame, not fragment ions (TIMS components are positioned before the fragmentation cell).

## Quick Example

```python
from tdfpy import DDA, DIA

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
```

## Installation

```bash
pip install tdfpy
```

See [Getting Started](getting-started.md) for a full walkthrough.
