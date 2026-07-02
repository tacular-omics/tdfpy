# Noise filters

Composable noise filters live in `tdfpy.noise`. A pipeline of filters is
applied in order; each takes raw `(scan_indices, mz_indices, intensities)`
and returns a boolean keep-mask. Frozen dataclasses make them hashable
(suitable for caching) and `dataclasses.replace`-tweakable.

```python
from tdfpy import MadThreshold, VerticalNoiseFilter, get_raw_peaks

peaks = get_raw_peaks(
    td, frame_id,
    noise=[
        VerticalNoiseFilter(min_streak_scans=5, num_iterations=2),
        MadThreshold(k=3),
    ],
)
```

User-facing APIs (`get_raw_peaks`, `get_centroided_spectrum`,
`Frame.raw_peaks`, etc.) also accept the string shorthand for terseness:
`noise="mad"`, `noise="iterative_median"`, `noise=500.0`, etc. See
[`coerce_filters`](#tdfpy.coerce_filters) for the accepted forms.

---

## Base class & coercion

::: tdfpy.NoiseFilter

::: tdfpy.coerce_filters

---

## Intensity-threshold filters

Each subclass exposes the knobs of its estimator as dataclass fields.

::: tdfpy.IntensityThreshold

::: tdfpy.AbsoluteThreshold

::: tdfpy.MadThreshold

::: tdfpy.PercentileThreshold

::: tdfpy.HistogramThreshold

::: tdfpy.BaselineThreshold

::: tdfpy.IterativeMedianThreshold

---

## Structural filters

::: tdfpy.VerticalNoiseFilter
    options:
      members:
        - keep_mask
        - run

::: tdfpy.noise.VerticalNoiseDiagnostics

::: tdfpy.HorizontalHaloFilter
    options:
      members:
        - keep_mask

## Precursor-space gates

Acquisition-aware **MS1-only** gates that drop signal the instrument never
schedules for fragmentation (so it can never become an identification). Each
reads the relevant region from `analysis.tdf`, converts it once to per-scan
integer TOF-index intervals via the run calibration, and tests membership with a
vectorised binary search. Both are no-ops (keep everything) when the run carries
no region, so they are safe to include unconditionally.

```python
from tdfpy import SelectionPolygonGate, DiaMs1WindowGate, MadThreshold, get_raw_peaks

# ddaPASEF: keep only MS1 inside the PASEF selection polygon, then denoise.
peaks = get_raw_peaks(td, frame_id, noise=[SelectionPolygonGate(), MadThreshold(k=3)])

# diaPASEF: keep only MS1 inside the union of isolation windows.
peaks = get_raw_peaks(td, frame_id, noise=[DiaMs1WindowGate()])
```

::: tdfpy.SelectionPolygonGate
    options:
      members:
        - keep_mask

::: tdfpy.DiaMs1WindowGate
    options:
      members:
        - keep_mask
