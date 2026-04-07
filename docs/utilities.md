# Utilities

## `slice_d_folder` — Extracting a time range from a `.d` folder

`slice_d_folder` creates a smaller, self-contained `.d` folder from an existing one by keeping
only a contiguous range of frames. The output is a fully valid Bruker `.d` folder: both the
SQLite metadata (`analysis.tdf`) and the binary scan data (`analysis.tdf_bin`) are rebuilt so
that downstream tools — including tdfpy's own readers — can open the result directly.

This is useful for:

- Creating small test datasets from a large acquisition
- Isolating a chromatographic peak or retention time window for focused analysis
- Reducing file size before sharing or archiving

### What gets filtered

The slicer keeps all frames whose `Id` falls within `[frame_start, frame_end]` (inclusive,
1-based) and removes everything else:

| Table | Behaviour |
|---|---|
| `Frames` | Rows outside the range are deleted |
| `PasefFrameMsMsInfo` | Rows referencing deleted frames are deleted |
| `DiaFrameMsMsInfo` | Rows referencing deleted frames are deleted |
| `PrmFrameMsMsInfo` | Rows referencing deleted frames are deleted |
| `Precursors` | Orphaned rows (parent frame deleted) are removed |
| `DiaFrameMsMsWindows` | Orphaned window groups are removed |
| `analysis.tdf_bin` | Rebuilt from scratch — only kept frames' blobs are written |

The `TimsId` offsets in the `Frames` table are updated to point to the correct positions in
the new binary file, so the output can be opened immediately with `DDA`, `DIA`, `PRM`, or
any Bruker-compatible tool.

!!! note "Frame IDs vs retention time"
    `frame_start` and `frame_end` are raw frame IDs (the `Id` column in the `Frames` table),
    not retention times. If you need to slice by time, open the `.d` folder first and look up
    frame IDs using `dda.ms1` or `dia.ms1`.

### Basic usage

```python
from tdfpy import slice_d_folder

out = slice_d_folder(
    source_dir="experiment.d",
    dest_dir="experiment_slice.d",
    frame_start=100,
    frame_end=300,
)
print(out)  # PosixPath('experiment_slice.d')
```

The destination directory is created automatically. If it already exists it is overwritten.

### Slicing by retention time

Open the source file first to map retention time to frame IDs:

```python
from tdfpy import DDA, slice_d_folder

with DDA("experiment.d") as dda:
    # Find frames within a retention time window (seconds)
    rt_min, rt_max = 600.0, 900.0  # 10 – 15 min
    frame_ids = [
        frame.frame_id
        for frame in dda.ms1
        if rt_min <= frame.time <= rt_max
    ]

first_frame = min(frame_ids)
last_frame = max(frame_ids)

slice_d_folder(
    source_dir="experiment.d",
    dest_dir="experiment_10to15min.d",
    frame_start=first_frame,
    frame_end=last_frame,
)
```

### Opening the result

The sliced folder can be opened with any tdfpy reader exactly like the original:

```python
from tdfpy import DDA

with DDA("experiment_slice.d") as dda:
    for frame in dda.ms1:
        peaks = frame.centroid()
        print(frame.frame_id, len(peaks))
```

::: tdfpy.slice_d_folder
    options:
      docstring_style: numpy
