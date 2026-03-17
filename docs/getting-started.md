# Getting Started

## Installation

```bash
pip install tdfpy
```

Requires Python 3.12+.

## Detecting Acquisition Type

Before loading data, you can inspect the acquisition type of a `.d` folder:

```python
from tdfpy import get_acquisition_type

acq_type = get_acquisition_type(D_PATH)
# Returns one of: "DDA", "DIA", "PRM", "Unknown"
print(acq_type)
```

## DDA Acquisitions

```python
from tdfpy import DDA

with DDA(D_PATH) as dda:
    # Iterate over MS1 frames
    for frame in dda.ms1:
        print(f"Frame {frame.frame_id} at RT {frame.time:.1f}s")
        # Centroid the frame — returns shape (N, 3): [m/z, intensity, 1/K0]
        peaks = frame.centroid()
        print(f"  {len(peaks)} centroided peaks")

    # Iterate over precursors (MS2)
    for precursor in dda.precursors:
        print(f"Precursor {precursor.precursor_id}: {precursor.largest_peak_mz:.4f} m/z")
        # Raw centroided peaks from Bruker's algorithm
        peaks = precursor.peaks
```

## DIA Acquisitions

```python
from tdfpy import DIA

with DIA(D_PATH) as dia:
    # MS1 frames
    for frame in dia.ms1:
        peaks = frame.centroid()

    # DIA windows
    for window in dia.windows:
        print(f"Window group {window.window_group}: isolation {window.isolation_mz} m/z")
        peaks = window.centroid()
```

## Lookups and Queries

Access frames, precursors, or windows directly by ID or query by properties.

```python
from tdfpy import DDA

with DDA(D_PATH) as dda:
    # Access by ID
    frame = dda.ms1[1]
    precursor = dda.precursors[1]

    # Query precursors by m/z and retention time
    results = dda.precursors.query(
        mz=1292.63,
        mz_tolerance=20.0,       # ppm by default
        rt=2400.0,               # seconds
        rt_tolerance=30.0,       # seconds
    )
    for p in results:
        print(p.precursor_id, p.largest_peak_mz)
```

```python
from tdfpy import DIA

with DIA(D_PATH) as dia:
    # Get all windows in a window group
    group_windows = dia.windows[0]  # returns a list

    # Query windows by retention time
    results = dia.windows.query(rt=1200.0, rt_tolerance=60.0)
    for w in results:
        print(w.window_group, w.isolation_mz)
```

## How Data Access Works

A `.d` folder contains two files: `analysis.tdf` (a SQLite database with metadata) and
`analysis.tdf_bin` (a binary file with the raw spectral data).

When you open a `DDA` or `DIA` reader, it immediately:

1. Opens a connection to the binary file
2. Reads all frame and precursor metadata from the SQLite database into memory

The objects you get back — `Frame`, `Precursor`, `DiaWindow`, etc. — all hold a reference
to that open connection. Their fields (`frame_id`, `rt`, `monoisotopic_mz`, etc.) are
available immediately. **Spectral data is fetched lazily**: calling `.peaks` or `.centroid()`
reads from the binary file at that moment.

This means objects cannot be used after the reader closes:

```python
from tdfpy import DDA

with DDA(D_PATH) as dda:
    frame = dda.ms1[1]
    peaks = frame.centroid()  # fine — connection is open

# peaks = frame.centroid()  # RuntimeError: connection is closed
```

## Development

The project uses `uv` for dependency management and `just` as a task runner.

```bash
just install-dev     # install with dev dependencies
just test            # run tests
just lint            # ruff linter
just check           # lint + test + type check
```

To serve the docs locally:

```bash
uv run --group docs mkdocs serve
```
