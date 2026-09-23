---
applyTo: "**"
---

# tdfpy — Copilot instructions

Read [CLAUDE.md](../CLAUDE.md) first: it is the single source of truth for
commands, layout, public API, conventions and design rules. The rules most
often broken:

- **No native library, no build step.** `analysis.tdf_bin` is decoded in
  Python/NumPy. Do not reintroduce Bruker's `libtimsdata`, ctypes, Rust or any
  compiled extension; the wheel must stay `py3-none-any`.
- **Unvalidated formats must raise, never approximate.** Do not loosen
  compression-type or calibration `ModelType` guards, and never regenerate
  `tests/data/*_golden.json` to make a test pass.
- **Every Numba kernel has a pure-Python fallback** gated on `_HAS_NUMBA`; add
  both paths and an equivalence test. Import numba only inside that try/except.
- **Fixed pipeline order:** `read_spectrum → subset_scans → exclude_region →
  smooth → apply_noise → centroider`, in integer index space, `convert` once at
  the end. Noise filters are pre-centroid only.
- **Lazy access:** spectral data after the reader's `with` block must raise
  `RuntimeError`. Test against `tests/data/example_*.d`, do not mock `TimsData`.

Run `just check` before pushing. Only the tacular-omics overseer bumps versions
or publishes.

When editing README, docs or the changelog, use neutral, factual language
without superlatives.
