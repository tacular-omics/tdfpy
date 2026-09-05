Spectrum batches and file checks

These helpers support reading and centroiding timsTOF data. Existing extraction
methods continue to return arrays. Processing diagnostics are an internal
development tool, and chromatogram extraction is deferred.

To process DIA or PRM windows with bounded reuse, pass windows in their existing
order to `iter_window_spectra`. Adjacent windows of the same frame share one
decode. Each result is a `(window, peaks)` pair. The numerical array matches
`window.centroid()` with the same settings. No diagnostic accounting runs as
part of batch extraction.

```python
from itertools import islice
from tdfpy import DIA, MergePeaksCentroider, iter_window_spectra

with DIA(D_PATH) as reader:
    for window, peaks in iter_window_spectra(
        islice(reader.windows, 4),
        centroid=MergePeaksCentroider(max_peaks=10),
    ):
        assert peaks.shape[1] == 3
        assert window.frame_id > 0
```

For PRM, pass `reader.transitions`. Unsorted inputs retain caller order and may
decode a frame again. Consume the iterator inside the reader's context. It
retains the current frame and uses no global spectrum cache or worker pool.
Returned numerical arrays remain usable after the reader closes.

For acquisition checks, run `tdfpy validate sample.d` or
`python -m tdfpy validate sample.d`. Add `--full` to decode every binary frame.
The command writes JSON and exits with status 0 on success or 1 for a failed
check. The Python function returns a structured report:

```python
from tdfpy import validate_acquisition

report = validate_acquisition(D_PATH)
assert report.valid
assert report.frames_checked > 0
```

Metadata mode checks supported metadata and calibration references. It does
not validate compressed payloads. Full mode additionally runs the decoder's
integrity checks for every frame, collecting frame-specific failures. Neither
mode repairs data or proves numerical equivalence to vendor software.

The built-in extraction and gate paths can share an open reader across worker
threads. Metadata needed by those paths is snapshotted when the reader opens.
Direct access to `td.conn` retains SQLite's thread rules. User-written filters
must be thread-safe themselves. Wait for workers before closing the reader.

All `ion_mobility_type="ccs"` raw-spectrum conversions assume charge +1.
Raw peaks do not identify charge states. `Precursor.ccs` uses a known precursor
charge when present, falling back to +1 when it is absent. Precursor scan
coordinates retain the original fractional metadata value.

For AI agents, an [optional MCP server](mcp.md) exposes acquisition queries,
spectrum extraction, conversions, and file checks without changing the core
Python installation.

::: tdfpy.iter_window_spectra

::: tdfpy.validate_acquisition
