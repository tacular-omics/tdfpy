---
title: "tdfpy: A Python package for parsing and centroiding Bruker timsTOF mass spectrometry data"
tags:
  - Python
  - Proteomics
  - Mass Spectrometry
  - Ion Mobility
  - timsTOF
  - TIMS
  - PASEF
  - Bioinformatics
authors:
  - name: Patrick T. Garrett
    orcid: 0000-0002-8434-9693
    affiliation: 1
  - name: John R. Yates III
    orcid: 0000-0001-5267-1672
    corresponding: true
    affiliation: 1
affiliations:
  - name: The Scripps Research Institute, United States
    index: 1
date: 30 May 2026
bibliography: paper.bib
---

# Summary

Bruker timsTOF mass spectrometers measure three properties for every detected
ion: mass-to-charge ratio (m/z), intensity, and ion mobility — reported as
reciprocal reduced mobility 1/K0 (V·s/cm²) or collision cross section
(CCS, Å²). The mobility dimension arises from Trapped Ion Mobility
Spectrometry (TIMS), which separates ions by size and charge before mass
analysis. Bruker stores each acquisition as a `.d` folder pairing a SQLite
metadata database (`analysis.tdf`) with a compressed binary of raw scans
(`analysis.tdf_bin`); that binary is undocumented, and reading it has
conventionally required Bruker's closed-source `libtimsdata` C library.

**tdfpy** decodes `analysis.tdf_bin` itself, in Python and NumPy, with no
native library and therefore no platform restriction. It exposes an
object-oriented API for the three principal acquisition modes —
Data-Dependent Acquisition (DDA), Data-Independent Acquisition (DIA), and
Parallel Reaction Monitoring (PRM) — covering the PASEF [@Meier2015PASEF] and
diaPASEF [@Meier2018diaPASEF] scan strategies, and reimplements Bruker's
calibration models so that raw indices become m/z and ion mobility without a
vendor call. It provides a composable peak-processing pipeline — region
exclusion, intensity smoothing, noise filtering, and centroiding — with two
interchangeable centroiders, treating ion mobility as a first-class clustering
dimension alongside m/z. Performance-critical kernels are JIT-compiled with
Numba [@Lam2015Numba], with pure-Python fallbacks. Centroided peaks are
returned as `(N, 3)` NumPy [@Harris2020NumPy] arrays of
`[m/z, intensity, ion mobility]`, ready for downstream analysis.

# Statement of Need

The timsTOF platform is widely used for high-throughput bottom-up proteomics,
where PASEF acquisition multiplies sequencing speed by synchronising TIMS
elution with MS2 fragmentation — routinely producing datasets with thousands
of frames and millions of raw ion measurements per run. Search engines such as
DIA-NN [@Demichev2020DIA-NN] and Spectronaut [@Bruderer2015Spectronaut] consume
processed peak lists, but researchers developing algorithms, extracting custom
features, or auditing data quality need direct access to the raw
three-dimensional ion clouds in a form that Python scientific libraries can
consume.

Bruker ships a reference Python SDK, but it exposes only a low-level procedural
interface over the closed-source library and is not an installable package.
tdfpy replaces that dependency with its own decoder and layers a full
proteomics pipeline on top: a pip-installable package (`pip install tdfpy`)
with no Rust or C toolchain requirement and no vendor binary to obtain,
acquisition-mode-aware readers that expose frames, precursors, and isolation
windows as typed dataclasses, and a peak-processing pipeline that clusters in
joint (m/z, ion mobility) space.

# State of the Field

**AlphaTims** [@Willems2021AlphaTims] is the closest neighbour, providing fast
indexed random access to individual ion measurements via Numba-accelerated flat
arrays; its focus is retrieval and visualisation rather than
acquisition-mode-aware iteration over frames, precursors, and DIA/PRM windows as
structured objects, and it does not centroid in joint (m/z, ion mobility) space.
**ionmob** [@Teschner2023ionmob] predicts CCS but does not parse raw spectra.
Vendor Python bindings require manual installation of Bruker's proprietary SDK
and offer no high-level abstractions. AlphaTims-based tools (AlphaPept, AlphaDIA)
wrap AlphaTims for feature finding but do not expose the raw three-dimensional
plane through a composable pipeline. tdfpy is distinguished by its installation
simplicity, typed object model, interchangeable centroiders, and first-class
treatment of ion mobility as a clustering dimension.

# Implementation

tdfpy is organised in three layers: a decoding layer that reads
`analysis.tdf_bin` in pure Python and NumPy (frame decompression and
de-interleaving, plus the TOF↔m/z, scan↔1/K0, and 1/K0↔CCS conversions); a
SQLite metadata layer that wraps `analysis.tdf` as pandas DataFrames
[@McKinney2010Pandas]; and a reader layer that materialises the metadata as
typed dataclasses (`DDAMs1Frame`, `Precursor`, `DiaWindow`, `PrmTransition`).
Metadata is loaded eagerly while spectral binary data is read lazily — only
when `.peaks`, `.raw_peaks()`, or `.centroid()` is called — and context
managers guarantee the open binary file is closed.

The conversion models are reimplemented from the run's own `MzCalibration` and
`TimsCalibration` tables rather than approximated from summary metadata, and
both decoder and models are pinned by regression tests against golden values
captured from Bruker's library while it was still vendored: frame decoding is
bit-exact over all 1,710 frames and 29,399,513 peaks of the three bundled
example acquisitions, and the calibration reproduces the vendor conversions to
~1e-10 relative in m/z and ~1e-15 in mobility. Variants that were never
validated this way — legacy per-scan compression, unknown calibration model
types, recalibrated state — raise rather than return plausible-looking numbers,
since a silently wrong calibration is undetectable downstream.

MS2 peak lists for DDA precursors and DIA windows are likewise produced
in-house, replacing the proprietary peak picker: intensities are summed per TOF
index over the relevant scan ranges (collapsing ion mobility) and merged at
30 ppm. Measured against the vendor output over 10 precursors and 12 DIA
windows, strong peaks agree to 0.0–1.9 ppm, total ion current to within 4%, and
peak counts run 0.95–1.09× — a divergence consistent with smoothing inside the
closed picker, and enforced as bounds by the test suite.

Peak processing is a chain of composable ops that work on raw integer
`(scan, TOF index)` data and defer unit conversion to a final step:
`read_spectrum → subset_scans → exclude_region → smooth → apply_noise →`
centroider. Convenience methods orchestrate this chain; power users can compose
the ops directly. Region exclusion (`ChargeStateRegion`) drops uninteresting
bands such as the singly-charged contamination line. Noise filtering offers
intensity-threshold estimators and two structural filters that exploit signal
geometry — `VerticalNoiseFilter`, which keeps the vertical ion-mobility streaks
of genuine ions, and `HorizontalHaloFilter`, which clears the diffuse halo
flanking bright peaks along the m/z axis (leaving the mobility streak intact).
Two centroiders are interchangeable per call site: a greedy
m/z-tolerance merger (default) and a watershed region-grower
[@Beucher1979Watershed] operating in integer index space. Full algorithm
descriptions and tunable parameters are documented online.
\autoref{fig:pipeline} shows the pipeline applied to one MS1 frame.

![`tdfpy` applied to one MS1 frame from the bundled example DDA acquisition
(zoomed to m/z 400–1200 and 1/K0 0.6–1.4), generated reproducibly by
`scripts/make_paper_figure.py` with the `VerticalNoiseFilter` →
`HorizontalHaloFilter` chain. Top-left: all raw (m/z, 1/K0) peaks, with those
rejected as noise drawn in grey behind the retained peaks (coloured by
log-intensity), so signal sits in front of the noise; top-right: the resulting
intensity-weighted centroids; bottom-left: the centroided m/z spectrum (stem
lines) over the retained raw signal; bottom-right: the peaks rejected as
noise.\label{fig:pipeline}](pipeline.png)

tdfpy is a pure-Python wheel (`py3-none-any`) with no compile step at install
and no bundled vendor binary; the Numba JIT replaced an earlier Rust extension
that produced platform-specific wheels, and dropping the native library lifted
the last platform restriction, so the package now installs on macOS and ARM as
well as Linux and Windows x86-64. Continuous integration fails the build if a
wheel contains a native binary. A PEP 561 `py.typed` marker ships the
package's type annotations. tdfpy has been on PyPI since 2022 and is
used at The Scripps Research Institute for in-house timsTOF DDA/DIA/PRM feature
detection and retention-time calibration [@tdfpy_zenodo].

# Example Usage

```python
from tdfpy import DDA, DIA

with DDA("experiment.d") as dda:
    for frame in dda.ms1:
        peaks = frame.centroid()      # (N, 3): [m/z, intensity, 1/K0]
    for precursor in dda.precursors:
        ms2, ccs = precursor.peaks, precursor.ccs

with DIA("experiment.d") as dia:
    for window in dia.windows:
        peaks = window.centroid()     # scoped to the isolation window
```

Region exclusion, smoothing, a chain of noise filters, and the choice of
centroider are all passed as keyword arguments to `.centroid()`; see the
documentation for the composable-pipeline and raw-op APIs.

# AI Usage Disclosure

Generative AI models (Claude, Cursor, and GitHub Copilot) were used to assist
in code development, test authoring, and manuscript drafting. All
AI-generated content was reviewed and verified against the source code by
the authors.

# Availability

tdfpy is distributed through PyPI (https://pypi.org/project/tdfpy/) and
available as open-source software on GitHub
(https://github.com/tacular-omics/tdfpy). Documentation is hosted at
https://tacular-omics.github.io/tdfpy. The software is released under the
MIT license.

# Acknowledgements

The authors thank Bruker Daltonics for making the `libtimsdata` shared
library available for use in open-source software development; it served as
the reference implementation against which tdfpy's decoder and calibration
models were validated.

# References
