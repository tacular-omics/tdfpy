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

The standalone [`smooth`](#tdfpy.smooth) op (and the lower-level
[`box_smooth`](#tdfpy.box_smooth) array helper) rewrite intensities in
place — a box **sum** or **mean** over a `(±scan_half_width,
±mz_idx_half_width)` window — without expanding the point set. Summing
(the default) amplifies genuine ion-mobility streaks ahead of noise
filtering; the mean variant backs `WatershedCentroider`'s seed-stabilising
smoother, which runs before seed selection by default via the
`smooth_scan_half_width` / `smooth_mz_idx_half_width` fields (defaults `5`
and `3`; set either to `0` to disable).

```python
from tdfpy import read_spectrum, smooth, apply_noise, VerticalNoiseFilter

s = read_spectrum(td, frame_id=1)
s = smooth(s, scan_half_width=5, mz_idx_half_width=2)   # box sum, amplify streaks
s = apply_noise(s, (VerticalNoiseFilter(),), td=td, frame_id=1)
```

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

## Smoothing

The convenience entry points (`get_raw_peaks`, `get_centroided_spectrum`,
`Frame.centroid()`, …) accept smoothing as a single `smooth=Smooth(...)`
argument; `smooth` / `box_smooth` are the underlying composable ops.

::: tdfpy.Smooth

::: tdfpy.smooth

::: tdfpy.box_smooth

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
