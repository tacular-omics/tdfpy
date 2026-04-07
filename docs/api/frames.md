# Frames

`Frame` is the base class for all MS1 frames. `DDAMs1Frame`, `DIAMs1Frame`, and `PRMMs1Frame`
inherit every field and method listed under `Frame` — only their additional fields are shown
below each subclass.

::: tdfpy.Frame

::: tdfpy.DDAMs1Frame
    options:
      inherited_members: false
      members: [precursors]

::: tdfpy.DIAMs1Frame
    options:
      inherited_members: false
      members: [dia_windows]

## PRM MS1 Frame

In a PRM acquisition, each MS1 frame carries references to the `PrmTransition` objects
that were being collected in nearby MS2 frames. This lets you correlate survey scans with
the targeted transitions acquired in the same run.

::: tdfpy.PRMMs1Frame
    options:
      inherited_members: false
      members: [prm_transitions]
