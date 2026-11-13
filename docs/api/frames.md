# Frames

`Frame` is the base class for all MS1 frames. `DDAMs1Frame`, `DIAMs1Frame`, and `PRMMs1Frame`
inherit every field and method listed under `Frame` — only their additional fields are shown
below each subclass.

::: tdfpy.Frame

::: tdfpy.DDAMs1Frame
    options:
      inherited_members: false
      members: [precursors]

## DIA and PRM MS1 frames

`DIAMs1Frame` and `PRMMs1Frame` add no fields to `Frame`; they exist so code can tell the
acquisition type from the frame's type. Find the windows or transitions of a run with
`DIA.windows` and `PRM.transitions` (both lookups with `query()` / `query_range()`).

::: tdfpy.DIAMs1Frame
    options:
      inherited_members: false
      members: []

::: tdfpy.PRMMs1Frame
    options:
      inherited_members: false
      members: []

## Frame MS/MS type

Every frame carries the raw `MsMsType` value of its row in the `Frames` table:
`MS1` on an MS1 frame, and `DDA_MS2`, `DIA_MS2` or `PRM_MS2` on an MS/MS frame.

::: tdfpy.MsMsType
    options:
      show_if_no_docstring: true
