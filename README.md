
<div align="center">
  <img src="https://raw.githubusercontent.com/tacular-omics/tdfpy/main/logo.png" alt="TDFpy Logo" width="400" style="margin: 20px;"/>

  A Python package for extracting data from Bruker timsTOF data files (.tdf and .tdf_bin). Includes a Numba-accelerated centroiding algorithm for efficient extraction of ion mobility data.

  [![Python package](https://github.com/tacular-omics/tdfpy/actions/workflows/ci.yml/badge.svg)](https://github.com/tacular-omics/tdfpy/actions/workflows/ci.yml)
  [![codecov](https://codecov.io/gh/tacular-omics/tdfpy/graph/badge.svg?token=RMUiW11IR2)](https://codecov.io/gh/tacular-omics/tdfpy)
  [![PyPI version](https://badge.fury.io/py/tdfpy.svg)](https://badge.fury.io/py/tdfpy)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19100532.svg)](https://doi.org/10.5281/zenodo.19100532)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

</div>

tdfpy reads Bruker timsTOF `.d` acquisitions straight from `analysis.tdf` and `analysis.tdf_bin` — no Bruker native library required. It gives you familiar Python objects for DDA, DIA, and PRM runs, plus a tunable, Numba-accelerated centroiding pipeline for pulling clean, ion-mobility-resolved peaks out of raw PASEF frames.

It's for proteomics and mass spec developers who want to script against timsTOF data without hand-rolling SQLite queries or reverse-engineering the binary frame format.

## Why tdfpy?

- **Pure Python, no native dependency** — `analysis.tdf_bin` is decoded directly, so it runs on Linux, macOS, and Windows, x86-64 and ARM
- **One API for DDA, DIA, and PRM** — frames, precursors, isolation windows, targets, and transitions are all typed Python objects
- **Composable peak pipeline** — chain region exclusion, smoothing, and noise filters before centroiding, or use short-hand defaults
- **Two centroiders** — a Numba-JIT'd greedy merge in float m/z space, and a watershed region-grower in integer TOF-index space, swappable without touching surrounding code
- **Lazy spectral access** — frame metadata loads upfront; raw peak data is only decoded when you call `.peaks`, `.raw_peaks()`, or `.centroid()`
- **Query by m/z and RT**, not just row index

## Installation

```bash
pip install tdfpy
```

Requires Python 3.12+. On Python 3.12/3.13 the `zstandard` package is installed automatically; Python 3.14+ uses the standard library's zstd module.

Optional extras:

```bash
pip install "tdfpy[viz]"  # matplotlib-based plotting helpers
pip install "tdfpy[mcp]"  # MCP server for AI-agent access to acquisitions
```

## Quick Start

```python
from tdfpy import DDA

with DDA("sample.d") as dda:
    # Iterate over MS1 frames
    for frame in dda.ms1:
        print(f"Frame {frame.frame_id} at RT {frame.time:.1f}s")
        peaks = frame.centroid()  # shape (N, 3): [m/z, intensity, 1/K0]
        print(f"  {len(peaks)} centroided peaks")
        break

    # Iterate over precursors (MS2)
    for precursor in dda.precursors:
        print(f"Precursor {precursor.precursor_id}: {precursor.largest_peak_mz:.4f} m/z")
        peaks = precursor.peaks  # MS2 centroided by tdfpy (mobility collapse + merge)
        break
```

DIA and PRM acquisitions work the same way with `DIA(...)` and `PRM(...)`; see the [getting started guide](https://tacular-omics.github.io/tdfpy/getting-started/) for both.

## What else it can do

| Feature | Example |
| --- | --- |
| **Lookups & queries** | `dda.precursors.query(mz=1292.63, mz_tolerance=20.0, rt=2400.0, rt_tolerance=30.0)` — by ID or by m/z/RT window |
| **Custom peak pipelines** | `frame.centroid(exclude=ChargeStateRegion(), smooth=Smooth(...), noise=[MadThreshold(k=3), ...], centroid=WatershedCentroider(...))` |
| **Noise filter shorthand** | `frame.centroid(noise="mad")` or `frame.centroid(noise=500.0)` for common cases |
| **CLI validation** | `tdfpy validate sample.d --full` checks every binary frame without modifying the acquisition |
| **MCP server** | `tdfpy-mcp` exposes acquisition inspection and spectrum extraction as tools for AI agents |

Full pipeline options (region exclusion, smoothing, the two centroiders, and the noise-filter chain) are covered in the [analysis guide](https://tacular-omics.github.io/tdfpy/analysis/) and [API reference](https://tacular-omics.github.io/tdfpy/api/pipeline/).

## Related packages

tdfpy is the timsTOF reader in the [tacular-omics](https://github.com/tacular-omics) family. [mzmlpy](https://github.com/tacular-omics/mzmlpy) reads mzML files the same way, and [spxtacular](https://github.com/tacular-omics/spxtacular) builds spectrum-processing pipelines on top of either.

## Documentation

Full documentation: [tacular-omics.github.io/tdfpy](https://tacular-omics.github.io/tdfpy/)

- [Changelog](https://github.com/tacular-omics/tdfpy/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/tacular-omics/tdfpy/blob/main/CONTRIBUTING.md)

## Citation

If you use tdfpy in published work, please cite it — see [`CITATION.cff`](https://github.com/tacular-omics/tdfpy/blob/main/CITATION.cff) or the [DOI record](https://doi.org/10.5281/zenodo.19100532).

## License

[MIT](https://github.com/tacular-omics/tdfpy/blob/main/LICENSE)
