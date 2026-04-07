# Lookups

Lookup classes provide iteration, index access, and query methods over collections
of frames, precursors, DIA windows, or PRM targets and transitions. All lookup objects
support:

- **Iteration** — `for item in lookup:`
- **Index access** — `lookup[id]`
- **Length** — `len(lookup)`
- **`.get(id)`** — returns `None` (or a default) instead of raising on a missing ID
- **`.query()`** — filter by m/z, retention time, or ion mobility with tolerances

---

## MS1 Frame Lookup

::: tdfpy.Ms1FrameLookup

---

## Precursor Lookup

::: tdfpy.PrecursorLookup

---

## DIA Window Lookup

`DiaWindowLookup` groups windows by `window_group`. Because a single window group
definition repeats across many frames, indexing by `window_group_id` returns a **list**
of `DiaWindow` objects — one per frame that used that group.

```python
from tdfpy import DIA

with DIA("experiment.d") as dia:
    # Iterate over all windows across all frames
    for window in dia.windows:
        print(window.frame_id, window.isolation_mz, window.rt)

    # All windows belonging to window group 3 (one per frame)
    group3 = dia.windows[3]

    # Query by retention time (±30 s default)
    for window in dia.windows.query(rt=600.0, rt_tolerance=15.0):
        print(window.window_group, window.isolation_mz)

    # Query by window group AND retention time
    for window in dia.windows.query(window_group_index=5, rt=600.0, rt_tolerance=10.0):
        peaks = window.centroid()
```

::: tdfpy.DiaWindowLookup

---

## PRM Lookups

In a PRM experiment the instrument cycles through a list of **targets** (predefined
precursor ions) and collects MS2 spectra for each. The two lookup classes below reflect
that structure:

- `PrmTargetLookup` — the list of analytes being monitored (one entry per analyte)
- `PrmTransitionLookup` — the individual MS2 acquisitions (many per target, spread across
  the chromatographic run)

### PRM Target Lookup

`PrmTargetLookup` gives direct access to `PrmTarget` objects by their integer `target_id`.
Use `.query()` to filter targets by m/z, expected retention time, or ion mobility (1/K0).

```python
from tdfpy import PRM

with PRM("experiment.d") as prm:
    # Iterate over all targets
    for target in prm.targets:
        print(target.target_id, target.monoisotopic_mz, target.charge)

    # Access a specific target by ID
    t = prm.targets[1]
    print(t.description, t.time, t.one_over_k0)

    # Query by m/z (20 ppm window)
    for target in prm.targets.query(mz=565.3189, mz_tolerance=20.0):
        print(target.target_id, target.monoisotopic_mz)

    # Query by m/z and expected RT (±30 s)
    for target in prm.targets.query(mz=565.3189, rt=480.0, rt_tolerance=30.0):
        print(target.target_id, target.description)

    # Query by 1/K0 (ion mobility)
    for target in prm.targets.query(ook0=0.92, ook0_tolerance=0.05):
        print(target.target_id, target.one_over_k0)
```

::: tdfpy.PrmTargetLookup

### PRM Transition Lookup

`PrmTransitionLookup` gives access to `PrmTransition` objects — the individual MS2
acquisitions captured during the run. Indexing by `target_id` returns a **list** of all
transitions collected for that target across the chromatographic run.

```python
from tdfpy import PRM

with PRM("experiment.d") as prm:
    # All transitions for target 1 (list — one per MS2 frame)
    transitions = prm.transitions[1]
    for t in transitions:
        print(t.frame_id, t.rt, t.collision_energy)
        peaks = t.peaks  # list of (mz, intensity) arrays per mobility scan

    # Query transitions for a specific target near a retention time
    for tr in prm.transitions.query(target=1, rt=480.0, rt_tolerance=30.0):
        centroided = tr.centroid()  # shape (N, 3): [m/z, intensity, 1/K0]

    # Query using a PrmTarget object directly
    target = prm.targets[1]
    for tr in prm.transitions.query(target=target, rt=target.time, rt_tolerance=20.0):
        print(tr.frame_id, tr.isolation_mz)
```

::: tdfpy.PrmTransitionLookup
