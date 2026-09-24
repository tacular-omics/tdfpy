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

`Precursor.peaks` and `PasefFrameMsmsInfo.peaks` stay: they are the single merged
`(N, 2)` spectrum. `.scan_peaks()` is the per-scan list of `(N, 2)` arrays, and
`PasefFrameMsmsInfo` now has `scan_peaks()`, `raw_peaks()` and `centroid()` too.

## Removed

- `plot_centroiding(mz_tolerance=, mz_tolerance_type=, im_tolerance=, im_tolerance_type=, min_peaks=, max_peaks=)`,
  deprecated in 4.1. Pass `centroid=MergePeaksCentroider(...)`.
- `tdfpy.centroiding.batch_iterator`, `calculate_nmass`, `get_tdf_df` and `Peak`.
  They were unused helpers, not part of the documented API.

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

## Downstream code

```python
# 4.x
frame.time, target.one_over_k0, metadata.one_over_k0_acq_range
td.scanNumToOneOverK0(frame_id, scans)
from tdfpy.timsdata import oneOverK0ToCCSforMz

# 5.0
frame.rt, target.ook0, metadata.ook0_acq_range
td.scan_num_to_ook0(frame_id, scans)
from tdfpy import ook0_to_ccs
```
