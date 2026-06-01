# Region exclusion

A region is a known area of the (m/z, 1/K0) plane that you want to drop
wholesale — typically based on physical knowledge of the acquisition
rather than from estimating noise. The canonical example is the
singly-charged / polymer contamination band in timsTOF MS1.

Conceptually distinct from [noise filters](noise.md): region exclusion
answers *"which part of the data plane are we even interested in?"*,
while noise filtering answers *"of what's left, what's real signal?"*.

```python
from tdfpy import ChargeStateRegion, get_raw_peaks

# Drop the typical singly-charged region
peaks = get_raw_peaks(td, frame_id, exclude=ChargeStateRegion())

# Custom line + cap
peaks = get_raw_peaks(
    td, frame_id,
    exclude=ChargeStateRegion(
        line=((400.0, 0.75), (1200.0, 1.5)),
        cap_at_upper_endpoint=True,
    ),
)
```

The line is converted to a per-scan TOF-index cutoff once per frame, so
exclusion happens via a single vectorized integer comparison.

---

::: tdfpy.ChargeStateRegion
