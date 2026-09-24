# Migrating to 5.0

tdfpy 5.0 cleans up the public API in one release. None of the renamed names keep an
alias, and everything deprecated in 4.x is gone. The table below lists every rename;
the sections after it cover changes of type or behaviour.

## Renamed

| 4.x | 5.0 |
|---|---|
| `Frame.time`, `PrmTarget.time` | `.rt` |
| `PrmTarget.one_over_k0` | `PrmTarget.ook0` |
| `Frame.mz_calibration` | `Frame.mz_calibration_id` |
| `Frame.tims_calibration` | `Frame.tims_calibration_id` |
| `Frame.property_group` | `Frame.property_group_id` |
| `Precursor.parent_frame` | `Precursor.parent_frame_id` |
| `PasefFrameMsmsInfo.precursor` | `PasefFrameMsmsInfo.precursor_id` |
| `DiaWindowGroup.window_group`, `DiaWindow.window_group` | `.window_group_id` |
| `Frame.peaks`, `DiaWindow.peaks`, `PrmTransition.peaks` (property) | `.scan_peaks()` (method) |
| `Precursor.peaks`, `PasefFrameMsmsInfo.peaks` (property) | `.merged_peaks()` (method) |
| `Precursor.pasef_peaks` (property) | `Precursor.pasef_merged_peaks()` (method) |
| `FrameMetadata.time` | `FrameMetadata.rt` |
| `FrameMetadata.mz_calibration`, `tims_calibration`, `property_group` | `*_id` |
| `DiaWindowLookup.query(window_group_index=...)`, `.query_range(window_group_index=...)` | `window_group=` (an id or a `DiaWindowGroup`) |
| `MetaData.one_over_k0_acq_range`, `_lower`, `_upper` | `MetaData.ook0_acq_range`, `_lower`, `_upper` |
| `TimsData.indexToMz`, `mzToIndex` | `index_to_mz`, `mz_to_index` |
| `TimsData.scanNumToOneOverK0`, `oneOverK0ToScanNum` | `scan_num_to_ook0`, `ook0_to_scan_num` |
| `TimsData.scanNumToVoltage`, `voltageToScanNum` | `scan_num_to_voltage`, `voltage_to_scan_num` |
| `TimsData.readScans` | `TimsData.read_scans` |
| `tdfpy.timsdata.oneOverK0ToCCSforMz` | `tdfpy.ook0_to_ccs` |
| `tdfpy.timsdata.ccsToOneOverK0forMz`, `ccsToOneOverK0ToCCSforMz` | `tdfpy.ccs_to_ook0` |
| `tdfpy.calibration.one_over_k0_to_ccs`, `ccs_to_one_over_k0` | `ook0_to_ccs`, `ccs_to_ook0` |
| `TimsCalibration.scan_to_one_over_k0`, `one_over_k0_to_scan` | `scan_to_ook0`, `ook0_to_scan` |
| MCP tool `query_dia_windows(window_group=...)` | `window_group_id=` |
| MCP tools `query_precursors`, `query_dia_windows`, `query_prm_targets`, `query_prm_transitions`: `rt=` | `rt_range=` |
| MCP `query_precursors` / `query_prm_targets`: `mz=` | `precursor_mz_range=` |
| MCP `query_dia_windows` / `query_prm_transitions`: `mz=` | `isolation_mz_range=` (matched on the isolation center) |
| `PasefFrameMsmsInfo.mz_range`, `DiaWindow.mz_range`, `PrmTransition.mz_range`, `Precursor.mz_range` | `.isolation_mz_range` |
| `.mz_begin`, `.mz_end` (isolation windows) | removed: `lo, hi = w.isolation_mz_range` |
| `PrmTarget.monoisotopic_mz` | `PrmTarget.precursor_mz` |
| `Frame.summed_intensities` | `Frame.total_ion_current` |
| `Frame.max_intensity` | `Frame.base_peak_intensity` |
| `PrecursorLookup.query_range(mz_range=...)`, `PrmTargetLookup.query_range(mz_range=...)` | `precursor_mz_range=` |
| `PrecursorLookup.query(mz=...)`, `PrmTargetLookup.query(mz=...)` | `precursor_mz=` |
| `Polarity.POSITIVE`, `Polarity.NEGATIVE` | `"positive"`, `"negative"` |

These names match mzmlpy, so code can read both formats with one vocabulary.

Every spectral accessor is now a method, so each call visibly decodes the frame:

| method | returns | cost |
|---|---|---|
| `scan_peaks()` | a list of `(N_i, 2)` `[m/z, intensity]` arrays, one per scan | decode only |
| `raw_peaks()` | one `(N, 3)` `[m/z, intensity, 1/K0]` array | decode only |
| `centroid()` | one `(N, 3)` array | decode + 2D centroiding |
| `merged_peaks()` (`Precursor`, `PasefFrameMsmsInfo`) | one `(N, 2)` array, mobility collapsed | decode + greedy m/z merge; the slowest, call once |

`PasefFrameMsmsInfo` now has `scan_peaks()`, `raw_peaks()` and `centroid()` too.

## Added

- `ms_level` on frames (1 or 2, from `msms_type`) and on `PasefFrameMsmsInfo`,
  `DiaWindow` and `PrmTransition` (always 2). Those window classes carry `msms_type`
  as a class constant.
- `Precursor.precursor_mz`: `monoisotopic_mz` when known, otherwise
  `largest_peak_mz`. `PrecursorLookup` matches on it.
- `reader.ms1.query_range(rt_range=(lo, hi))` and `reader.ms1.query(rt=, rt_tolerance=)`.
- `tdfpy.iter_precursor_spectra(dda.precursors)` yields `(precursor, peaks)` with
  `peaks == precursor.merged_peaks()`, decoding each PASEF frame once. Use it instead
  of looping over `merged_peaks()` when you need many precursors.

## Removed

- `plot_centroiding(mz_tolerance=, mz_tolerance_type=, im_tolerance=, im_tolerance_type=, min_peaks=, max_peaks=)`,
  deprecated in 4.1. Pass `centroid=MergePeaksCentroider(...)`.
- `tdfpy.centroiding.batch_iterator`, `calculate_nmass`, `get_tdf_df` and `Peak`.
  They were unused helpers, not part of the documented API.
- `DIAMs1Frame.dia_windows` and `PRMMs1Frame.prm_transitions`. They were always
  empty. Use `DIA.windows` and `PRM.transitions`.

## Changed types and behaviour

- **Errors.** Every tdfpy error subclasses `TdfpyError`, which subclasses
  `ValueError`. A missing id or key raises `TdfpyKeyError` (also a `KeyError`);
  spectral access after the reader closes raises `ReaderClosedError` (also a
  `RuntimeError`). `UnsupportedTdfError` and `UnsupportedCalibrationError` still
  subclass `NotImplementedError`. SQLite errors from `PandasTdf`,
  `convert_table_to_df` and `slice_d_folder` are raised as `TdfpyError` (was
  `RuntimeError` or a raw `sqlite3.Error`). Handlers for the old built-in types keep
  working, except `except RuntimeError` around `convert_table_to_df`.
- **`get_acquisition_type`** returns `AcquisitionType`, a `StrEnum`, so
  `get_acquisition_type(p) == "DDA"` still holds.
- **`Frame.msms_type`** is a `MsMsType` (an `IntEnum`), not a plain `int`.
  `MsMsType` and `Polarity` are exported from `tdfpy`.
- **`Polarity`** is `Literal["positive", "negative"]` (was a `StrEnum` with the same
  values). Every `polarity` field is `Polarity | None`: a `Polarity` column value
  other than `+` / `-` gives `None` and one warning per file, and
  `Precursor.polarity` is `None` with a warning when its PASEF frames disagree.
  Compare with strings: `frame.polarity == "positive"`. `FrameMetadata.polarity`
  is still the raw `"+"` / `"-"`.
- **Lookup keywords.** A `(low, high)` tuple is always `*_range` and goes to
  `query_range`; a point is `rt=` / `precursor_mz=` / `ook0=` plus a `*_tolerance` and goes
  to `query`, as in mzmlpy 0.10. `query(mz=...)` is now `query(precursor_mz=...)`
  and the range is `precursor_mz_range=`; `mz_tolerance` / `mz_tolerance_type` are unchanged.
- **MS1 centroids.** `MergePeaksCentroider` now sorts on the integer TOF index
  before converting to m/z, and `merge_peaks` orders its input by m/z, then
  descending 1/K0 (TOF, then scan), then intensity, and seeds equal-intensity
  peaks in that order with a stable sort, so the result does not depend on input
  order even when many points share one m/z. About 1.2-1.4% of centroids
  on dense MS1 frames differ from 4.x (1.43% diaPASEF, 1.16% DDA, 15-min runs);
  summed intensity moves by under 0.02%. `merged_peaks()` is unchanged.
- **Elements are frozen, slotted and keyword-only.** Assigning to a field raises
  `FrozenInstanceError`; constructing one needs keyword arguments.
  `PrmTarget` equality and hashing ignore `transitions`.
- **Lookups.** `dia.windows[group_id]` and `prm.transitions[target_id]` return a
  tuple (was a list). `DIA.window_groups` is a tuple (was a generator). Every lookup
  has `ids()`, supports `id in lookup`, and `get(id, default)`. The arguments of
  `query()` and `query_range()` are keyword-only.
- **`MetaData` and `Calibration`** are read-only `Mapping`s of key to value. The
  `df` pandas Series field is replaced by `table`, a plain mapping. Use
  `dict(metadata)` or `pd.Series(metadata)` if you need a copy.
- **`plot_centroiding`** takes every argument after `frame_id` by keyword, including
  `ion_mobility_type`.
- **Mobility ranges are `(low, high)`.** `ook0_range`, `ccs_range` and
  `voltage_range` (and their `*_begin` / `*_end`) on `PasefFrameMsmsInfo`,
  `Precursor`, `DiaWindow` and `PrmTransition` used to follow scan order, which is
  high-to-low 1/K0. They now match `isolation_mz_range`, `ook0_acq_range` and
  `query_range(ook0_range=)`. Code that swapped them by hand must stop.
- **Wrong reader for the file.** `DDA` on a DIA run (or any other mismatch) raises
  `AcquisitionTypeError`. Check first with `tdfpy.get_acquisition_type(path)`.
- **Closed readers.** After the `with` block, `reader.timsdata`, `metadata`,
  `calibration` and `pandas_tdf` raise `ReaderClosedError`, like the spectral
  accessors. Read what you need inside the block.
- **`AcquisitionType.UNKNOWN`** is `"unknown"` (was `"Unknown"`).
- **`TimsData`** takes its options by keyword: `TimsData(path, use_recalibrated_state=True)`.
  `FrameMetadata` is exported from `tdfpy` and its fields follow `Frame` (`rt`, `*_id`).
- **`TimsData.read_scans` and `read_frame_arrays`** raise `TdfpyError` for a
  non-integer bound or a range outside `0 <= scan_begin <= scan_end <= num_scans`.
  `read_scans` padded with empty arrays and `read_frame_arrays` clamped; an empty
  range (`begin == end`) is valid and reads nothing.
- **Window accessors** (`scan_peaks`, `raw_peaks`, `centroid`, `merged_peaks`) raise
  `TdfpyError` when a window's scan range does not fit its frame, instead of clamping.

## Downstream code

```python
# 4.x
frame.time, target.one_over_k0, metadata.one_over_k0_acq_range
td.scanNumToOneOverK0(frame_id, scans)
from tdfpy.timsdata import oneOverK0ToCCSforMz
ms2 = precursor.peaks
lo, hi = window.mz_begin, window.mz_end
dda.precursors.query_range(mz_range=(500, 600))
frame.summed_intensities, target.monoisotopic_mz
frame.polarity == Polarity.POSITIVE

# 5.0
frame.rt, target.ook0, metadata.ook0_acq_range
td.scan_num_to_ook0(frame_id, scans)
from tdfpy import ook0_to_ccs
ms2 = precursor.merged_peaks()
lo, hi = window.isolation_mz_range
dda.precursors.query_range(precursor_mz_range=(500, 600))
frame.total_ion_current, target.precursor_mz
frame.polarity == "positive"
```
