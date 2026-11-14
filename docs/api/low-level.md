# Low-level access

Direct access to the two files inside a `.d` folder: `PandasTdf` reads the `analysis.tdf`
SQLite metadata into pandas DataFrames, and `TimsData` decodes frames from
`analysis.tdf_bin` in pure Python (no Bruker native library). Prefer the high-level
`DDA`/`DIA`/`PRM` readers unless you need raw frame or scan data.

::: tdfpy.PandasTdf

::: tdfpy.TimsData

::: tdfpy.timsdata_connect

::: tdfpy.FrameMetadata

## Collision cross section conversions

Convert inverse reduced mobility (1/K0, V·s/cm²) to collision cross section (Å²)
and back with the Mason-Schamp equation. These helpers use Bruker's ion-mass
convention, `mz * abs(charge)`, without subtracting proton masses.

::: tdfpy.ook0_to_ccs

::: tdfpy.ccs_to_ook0

## Errors

Every tdfpy error subclasses `TdfpyError` (a `ValueError`). Unsupported or unvalidated
formats raise instead of returning approximate values.

::: tdfpy.TdfpyError

::: tdfpy.TdfpyKeyError

::: tdfpy.ReaderClosedError

::: tdfpy.AcquisitionTypeError

::: tdfpy.UnsupportedTdfError

::: tdfpy.UnsupportedCalibrationError
