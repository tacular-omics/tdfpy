# PRM Data Elements

Parallel Reaction Monitoring (PRM) experiments select a predefined list of precursor ions
and collect high-resolution MS2 spectra for each across the chromatographic run.
The two classes on this page represent those two levels of structure.

## PrmTarget

A `PrmTarget` represents one entry in the instrument's target list — a single analyte
defined by its m/z, charge state, expected retention time, and expected ion mobility.
The instrument uses these values to schedule isolation windows and select the correct
mobility range during data collection.

Each target accumulates back-references to all `PrmTransition` objects collected for it
via the `transitions` field.

```python
from tdfpy import PRM

with PRM("experiment.d") as prm:
    for target in prm.targets:
        print(
            f"Target {target.target_id}: "
            f"{target.monoisotopic_mz:.4f} m/z, "
            f"charge {target.charge}, "
            f"RT {target.time:.1f} s, "
            f"1/K0 {target.one_over_k0:.3f}"
        )
        # All transitions collected for this target
        for tr in target.transitions:
            print(f"  Frame {tr.frame_id}, RT {tr.rt:.1f} s")
```

::: tdfpy.PrmTarget

---

## PrmTransition

A `PrmTransition` represents a single MS2 acquisition event for a PRM target — one
isolation window applied to a specific frame and mobility scan range. Multiple transitions
are collected for each target as the analyte elutes across time.

`PrmTransition` provides `.peaks` for raw scan data and `.centroid()` for processed spectra,
consistent with the `DiaWindow` and `PasefFrameMsmsInfo` APIs.

```python
from tdfpy import PRM

with PRM("experiment.d") as prm:
    for transition in prm.transitions:
        print(
            f"Frame {transition.frame_id}, "
            f"target {transition.target.target_id}, "
            f"isolation {transition.isolation_mz:.3f} m/z, "
            f"CE {transition.collision_energy:.1f} eV"
        )
        # Centroided MS2 spectrum — shape (N, 3): [m/z, intensity, 1/K0]
        peaks = transition.centroid()
        break
```

::: tdfpy.PrmTransition
