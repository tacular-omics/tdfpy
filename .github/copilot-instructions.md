---
applyTo: "**"
---

# tdfpy — Copilot instructions

See [AGENTS.md](../AGENTS.md) — the single source of truth for agent
instructions in this repo. It covers the repository layout, commands, testing,
code style, and the full list of design decisions. Read it before changing
anything; the rules below are the ones most often violated, copied verbatim.

- **No native library.** `analysis.tdf_bin` is decoded directly in
  Python/NumPy (v3.0.0); Bruker's `libtimsdata` is gone. Do not reintroduce
  it or any ctypes binding. Build backend is hatchling; wheels are
  `py3-none-any` and CI fails if one contains a `.so`/`.dll`.
- **Unvalidated formats must raise, never approximate.** A wrong calibration
  produces plausible numbers that nothing downstream can detect. Legacy
  `TimsCompressionType` 1, unknown calibration `ModelType`s, recalibrated
  state and pressure compensation all raise. If you add support for one,
  validate it against Bruker's library first and extend
  `tests/test_calibration_golden.py`.
- **Numba is a hard dependency.** Every JIT-compiled kernel has a
  pure-Python NumPy fallback gated on `_HAS_NUMBA`. When adding a new
  kernel, write both paths and add a Numba/Python equivalence test.
- **Composable pipeline.** Every centroiding entry point orchestrates the
  same ordered ops: `read_spectrum → subset_scans → exclude_region →
  apply_noise → centroider`. Each op consumes and produces a `RawSpectrum`
  in integer-index space; conversion to float (m/z, 1/K0) happens once at
  the end via `convert`. Power users compose the ops directly.

## Updating README / Documentation / Changelog

When updating these files use neutral language. Avoid over the top adjectives,
since most of the time the code is very mundane and not 'extraordinary'. Be
straight to the point and factual. Documentation should be clear and concise,
not flowery or embellished. Only explain further if necessary for clarity.
