# Pipeline

The pipeline module exposes the composable ops behind `get_raw_peaks` and
`get_centroided_spectrum`. Each op takes (and most return) a
[`RawSpectrum`](#tdfpy.RawSpectrum) — raw peaks in their native
``(scan_number, TOF_index, intensity)`` integer form.

Use the convenience entry points for common workflows; reach into the ops
when you need a custom ordering, want to plug in a transformation, or
want to skip a step.

```python
from tdfpy import (
    read_spectrum, subset_scans, exclude_region,
    apply_noise, convert, centroid_peaks,
    ChargeStateRegion, MadThreshold, WatershedCentroider,
)

with tdfpy.timsdata_connect("data.d") as td:
    s = read_spectrum(td, frame_id=1)
    s = subset_scans(s, scan_num_begin=0, scan_num_end=400)
    s = exclude_region(s, ChargeStateRegion(), td=td, frame_id=1)
    s = apply_noise(s, (MadThreshold(k=3),), td=td, frame_id=1)
    centroids = WatershedCentroider(
        attach_scan_half_width=10, attach_mz_idx_half_width=3
    )(s, td, 1)
```

`WatershedCentroider` accepts an optional per-group "leash" via
`max_scan_from_seed` and `max_mz_idx_from_seed` — bounds on how far any
group member can be from its seed. Useful for stopping chain-grown
groups from wandering across the data. `max_mz_idx_from_seed` defaults
to `10`; `max_scan_from_seed` defaults to `None` (no bound on that axis).

```python
# Cap group span at ±20 TOF indices from the seed
WatershedCentroider(
    attach_scan_half_width=10, attach_mz_idx_half_width=3,
    max_mz_idx_from_seed=20,
)
```

Position-preserving intensity smoothing runs before seed selection by
default, controlled by `WatershedCentroider`'s `smooth_scan_half_width` /
`smooth_mz_idx_half_width` fields (defaults `5` and `3`). Set either to
`0` to disable. There is no standalone smoothing pipeline op — the
convolution-style `smooth()` was removed because it expanded the point
set, often dramatically. The watershed's box-mean smoothing rewrites
intensities in place and is the canonical smoothing path.

---

## Data carrier

::: tdfpy.RawSpectrum

---

## Reading

::: tdfpy.read_spectrum

---

## Scoping

::: tdfpy.subset_scans

::: tdfpy.exclude_region

---

## Noise filtering

::: tdfpy.apply_noise

---

## Conversion

::: tdfpy.convert

---

## Centroiders

The two centroiders share an [`Centroider`](#tdfpy.Centroider) ABC.
`MergePeaksCentroider` (default) operates on float m/z values via a greedy
tolerance-based merge; `WatershedCentroider` works in integer index space
via intensity-ordered region growing.

::: tdfpy.Centroider

::: tdfpy.MergePeaksCentroider

::: tdfpy.WatershedCentroider

::: tdfpy.centroid_peaks
