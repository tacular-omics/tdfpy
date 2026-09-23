# tdfpy

[![Python package](https://github.com/tacular-omics/tdfpy/actions/workflows/ci.yml/badge.svg)](https://github.com/tacular-omics/tdfpy/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tdfpy.svg)](https://pypi.org/project/tdfpy/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19100532.svg)](https://doi.org/10.5281/zenodo.19100532)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

tdfpy reads Bruker timsTOF `.d` folders (`analysis.tdf` and `analysis.tdf_bin`) in pure
Python, with no Bruker native library. DDA, DIA and PRM acquisitions (PASEF and diaPASEF)
come back as familiar objects: MS1 frames, precursors, isolation windows, PRM targets and
transitions.

Spectra are read lazily and centroided by a Numba-accelerated pipeline that keeps ion
mobility: region exclusion, smoothing, noise filters, and a choice of two centroiders.
Peaks are NumPy arrays of `[m/z, intensity, 1/K0]`.

```python
from tdfpy import DDA

with DDA("sample.d") as dda:
    for frame in dda.ms1:
        peaks = frame.centroid()  # shape (N, 3): m/z, intensity, 1/K0
```

## Installation

```bash
pip install tdfpy
```

Requires Python 3.12+. Extras: `tdfpy[viz]` (plots), `tdfpy[mcp]` (MCP server).

## Where next

- [Getting started](getting-started.md): DDA, DIA and PRM walkthroughs on real data.
- [Spectrum batches and file checks](analysis.md): batch window extraction and `.d` validation.
- [MCP interface](mcp.md): let an AI agent query and extract timsTOF data.
- [API reference](api/readers.md): every public class and function.

## Related packages

The tacular-omics mass spectrometry stack:

- **tdfpy** (this package) reads Bruker timsTOF `.d` data.
- [mzmlpy](https://tacular-omics.github.io/mzmlpy/) reads mzML files.
- [spxtacular](https://tacular-omics.github.io/spxtacular/) processes the spectra from both: centroiding, deconvolution, matching, scoring and plotting.
