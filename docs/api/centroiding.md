# Centroiding

timsTOF raw data is profile-like: the binary file stores one intensity value per scan per
m/z index, spread across hundreds of mobility bins. Centroiding collapses that cloud of raw
measurements into a compact list of peaks — each with a single m/z, intensity, and ion
mobility value.

tdfpy provides two centroiding functions:

- **`get_centroided_spectrum`** — high-level: reads a full frame from disk, applies optional
  noise filtering, and returns centroided peaks in one call.
- **`merge_peaks`** — low-level: centroids pre-assembled NumPy arrays of m/z, intensity, and
  ion mobility values. Use this when you already have the raw arrays or need fine-grained
  control.

In practice, most workflows should call `.centroid()` directly on a `Frame`, `DiaWindow`, or
`PrmTransition` object — that method delegates to `get_centroided_spectrum` internally.

## Numba JIT backend

When [Numba](https://numba.pydata.org/) is installed (it is included in the default
`tdfpy` dependencies), the core clustering loop runs as a JIT-compiled native function
(`_merge_peaks_numba_kernel`). This is typically 5–20× faster than the pure-Python
fallback for large frames. The backend is selected automatically:

```python
# Numba used if available (default)
peaks = merge_peaks(mz, intensity, im)

# Force the Python fallback (useful for debugging or environments without Numba)
peaks = merge_peaks(mz, intensity, im, use_numba=False)
```

The first call after import triggers Numba's JIT compilation — expect a few seconds of
overhead. Subsequent calls use the cached compiled kernel.

---

## `get_centroided_spectrum`

Reads frame `frame_id` from the open `TimsData` connection, converts m/z indices to
m/z values, assembles the raw peak arrays, optionally filters noise, and runs centroiding.

```python
from tdfpy import timsdata_connect, get_centroided_spectrum

with timsdata_connect("experiment.d") as td:
    # Default: 1/K0 ion mobility, 8 ppm m/z tolerance
    peaks = get_centroided_spectrum(td, frame_id=1)
    print(peaks.shape)   # (N, 3) — columns: [m/z, intensity, 1/K0]

    # Tighter tolerances, CCS instead of 1/K0
    peaks = get_centroided_spectrum(
        td,
        frame_id=1,
        ion_mobility_type="ccs",
        mz_tolerance=5.0,
        im_tolerance=0.03,
    )

    # Noise filtering before centroiding
    peaks = get_centroided_spectrum(
        td,
        frame_id=1,
        noise_filter="mad",   # median absolute deviation
    )

    # Hard intensity threshold
    peaks = get_centroided_spectrum(td, frame_id=1, noise_filter=500.0)
```

Available `noise_filter` options: `"mad"`, `"percentile"`, `"histogram"`, `"baseline"`,
`"iterative_median"`, or any `float`/`int` as a direct threshold. Pass `None` to skip
filtering (the default).

::: tdfpy.get_centroided_spectrum

---

## `merge_peaks`

Centroids pre-assembled arrays. The algorithm is a greedy intensity-ordered scan: starting
from the highest-intensity raw peak, every neighbouring peak within the m/z and ion mobility
tolerances is merged into a single centroid via intensity-weighted averaging. Merged peaks
are marked as used and skipped in subsequent iterations.

| Parameter | Default | Notes |
|---|---|---|
| `mz_tolerance` | `8.0` | Width of the m/z matching window |
| `mz_tolerance_type` | `"ppm"` | `"ppm"` or `"da"` |
| `im_tolerance` | `0.1` | Width of the ion mobility window |
| `im_tolerance_type` | `"relative"` | `"relative"` (fraction of 1/K0) or `"absolute"` |
| `min_peaks` | `3` | Raw peaks required to form a centroid; set to `0` or `1` to keep all |
| `max_peaks` | `None` | Cap on output peaks (highest-intensity first) |
| `use_numba` | `True` | Set to `False` to force the Python fallback |

```python
import numpy as np
from tdfpy import merge_peaks

mz  = np.array([500.001, 500.002, 700.005, 700.006, 700.007])
inten = np.array([8000.0,  4000.0,  6000.0,  5000.0,  3000.0])
im  = np.array([0.85,     0.85,    0.92,    0.92,    0.92])

peaks = merge_peaks(mz, inten, im, mz_tolerance=10.0, min_peaks=2)
print(peaks)
# shape (2, 3): two centroided peaks, columns [m/z, intensity, 1/K0]
```

### Noise filtering vs `min_peaks`

The `noise_filter` parameter (available on both `get_centroided_spectrum` and `.centroid()`)
estimates a threshold from the intensity distribution and discards centroids below it.
While convenient, **intensity-based noise estimation has a fundamental limitation**: it cannot
distinguish low-abundance real signal from electronic noise, and will discard both equally.
Methods such as `"mad"` are anchored to the median of all centroid intensities — if your
sample has sparse signal, the threshold can rise above legitimate low-abundance peaks.

A more reliable strategy is to increase `min_peaks` instead:

```python
# Prefer: raise min_peaks to filter noise without discarding low-abundance signal
peaks = merge_peaks(mz, intensity, im, min_peaks=5)

# Noise arises from single scans; real peaks appear across multiple scans.
# min_peaks=5 means a centroid must be supported by at least 5 raw measurements.
```

Because electronic noise typically manifests as a singleton in a single scan, requiring
several supporting raw peaks is a structural filter — it targets the *origin* of noise
rather than its intensity. Low-abundance real peaks that appear consistently across scans
will survive, whereas noise will not.

Use `noise_filter` only if you have a calibrated threshold or a specific method validated
for your acquisition settings, and always verify against a `noise_filter=None` baseline first.

::: tdfpy.merge_peaks
