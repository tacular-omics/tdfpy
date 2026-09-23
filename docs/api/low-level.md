# Low-level access

Direct access to the two files inside a `.d` folder: `PandasTdf` reads the `analysis.tdf`
SQLite metadata into pandas DataFrames, and `TimsData` decodes frames from
`analysis.tdf_bin` in pure Python (no Bruker native library). Prefer the high-level
`DDA`/`DIA`/`PRM` readers unless you need raw frame or scan data.

::: tdfpy.PandasTdf

::: tdfpy.TimsData

::: tdfpy.timsdata_connect

::: tdfpy.FrameMetadata

## Errors

Unsupported or unvalidated formats raise instead of returning approximate values.

::: tdfpy.UnsupportedTdfError

::: tdfpy.UnsupportedCalibrationError
