"""Build llms-full.txt: a self-contained tdfpy guide for LLM agents.

The output is an orientation section written for agents (``HEADER`` below),
followed by the user-facing docs pages with every mkdocstrings ``::: tdfpy.X``
directive expanded to the object's real signature and docstring summary,
introspected from the installed package. Public names that no docs page covers
are appended at the end.

Writes both copies, which must stay identical:

- ``llms-full.txt`` at the repo root (read by agents working from a clone), and
- ``docs/llms-full.txt`` (shipped at /llms-full.txt on the docs site).

Usage (from anywhere; imports the tdfpy in the current environment)::

    uv run python scripts/build_llms_full.py
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import tdfpy

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUTPUTS = [REPO_ROOT / "llms-full.txt", DOCS_DIR / "llms-full.txt"]

# User-facing pages, in reading order, relative to docs/. maintenance.md is for
# maintainers and is left out.
DOCS_ORDER = [
    "index.md",
    "getting-started.md",
    "utilities.md",
    "analysis.md",
    "mcp.md",
    "api/readers.md",
    "api/frames.md",
    "api/precursor.md",
    "api/windows.md",
    "api/prm.md",
    "api/metadata.md",
    "api/lookup.md",
    "api/centroiding.md",
    "api/pipeline.md",
    "api/noise.md",
    "api/regions.md",
    "api/viz.md",
    "api/low-level.md",
    "citation.md",
]

DIRECTIVE = re.compile(r"^::: (\S+)\s*$")
RULE = "=" * 78
# Dunder methods worth listing; every other underscore name is private.
SHOWN_DUNDERS = ("__getitem__", "__iter__", "__len__", "__call__", "__contains__")
NOISE_SPEC_EXPANDED = "NoiseFilter | str | float | int | list['NoiseSpec'] | tuple['NoiseSpec', ...] | None"

HEADER = """\
# tdfpy — full usage guide for LLMs

> Read Bruker timsTOF data (`.d` folders: `analysis.tdf` SQLite + `analysis.tdf_bin`)
> in pure Python/NumPy. Mode-aware readers for DDA, DIA and PRM (PASEF / diaPASEF),
> lazy spectral access, and a composable Numba-accelerated centroiding pipeline.

This file is self-contained: an orientation section written for agents, then the
user-facing tdfpy documentation pages (docs/*.md) with every API reference
directive expanded to the real signature and one-line docstring.
Index: https://tacular-omics.github.io/tdfpy/llms.txt

## Install

```bash
pip install tdfpy                 # numpy, pandas, numba; zstandard on Python < 3.14
pip install "tdfpy[viz]"          # + matplotlib, for plot_centroiding
pip install "tdfpy[mcp]"          # + MCP SDK, for the tdfpy-mcp agent server
```

Python 3.12+. Pure-Python wheel (`py3-none-any`): no Bruker SDK, no native
library, works on Linux, macOS and Windows.

## Orientation

```python
import tdfpy
from tdfpy import DDA, DIA, PRM, get_acquisition_type, ChargeStateRegion, WatershedCentroider

get_acquisition_type("run.d")          # "DDA" | "DIA" | "PRM" | "unknown"

with DDA("run.d") as dda:              # always use the context manager
    frame = dda.ms1[1]                 # Ms1FrameLookup: by frame id, or iterate
    raw = frame.raw_peaks()            # (N, 3) float: m/z, intensity, 1/K0 (no centroiding)
    cen = frame.centroid()             # (N, 3) float: m/z, intensity, 1/K0
    cen = frame.centroid(noise="mad", exclude=ChargeStateRegion())
    cen = frame.centroid(centroid=WatershedCentroider())
    for p in dda.precursors.query(precursor_mz=652.3, rt=1200.0):   # PrecursorLookup
        p.merged_peaks()               # (N, 2): MS2 m/z, intensity (mobility collapsed, 30 ppm merge)

with DIA("run.d") as dia:
    for w in dia.windows:              # DiaWindowLookup; also dia.window_groups
        w.centroid()                   # (N, 3); the 1/K0 axis is the PRECURSOR mobility

with PRM("run.d") as prm:
    for tr in prm.transitions:         # PrmTransitionLookup; also prm.targets
        tr.centroid()                  # (N, 3)
        tr.scan_peaks()                # list of (N, 2) arrays, one per mobility scan
```

Return types at a glance:

| call | returns |
|---|---|
| `Frame.raw_peaks()`, `Frame.centroid()`, `DiaWindow.centroid()`, `PrmTransition.centroid()` | `np.ndarray` shape `(N, 3)`: m/z, intensity, ion mobility (`ion_mobility_type="ook0"` default; also `"ccs"`, `"voltage"`) |
| `Precursor.merged_peaks()` | `np.ndarray` `(N, 2)`: m/z, intensity |
| `Frame.scan_peaks()`, `DiaWindow.scan_peaks()`, `PrmTransition.scan_peaks()` | `list[np.ndarray]`, one `(N, 2)` array per mobility scan |
| `get_acquisition_type(path)` | `AcquisitionType` (a `StrEnum`: `"DDA"`, `"DIA"`, `"PRM"` or `"unknown"`) |
| `validate_acquisition(path, full=False)` | `ValidationReport` (`.valid`, `.issues`) |

Keyword arguments shared by `raw_peaks()` / `centroid()` and the functional
`get_raw_peaks` / `get_centroided_spectrum`:

- `exclude=ChargeStateRegion(...)`: drop the singly-charged band before anything else.
- `smooth=Smooth(...)`: optional smoothing in (scan, TOF-index) space.
- `noise=`: `None`, a `NoiseFilter`, a string (`"mad"`, `"percentile"`, `"histogram"`,
  `"baseline"`, `"iterative_median"`), a number (absolute intensity threshold), or a
  list of any of these applied in order.
- `centroid=` (centroid only): `MergePeaksCentroider()` (default) or `WatershedCentroider()`.

Pipeline order is fixed: `read_spectrum → subset_scans → exclude_region → smooth →
apply_noise → centroider`, all in integer (scan, TOF-index) space, then `convert` to
m/z and 1/K0 once at the end. Compose the ops yourself for custom processing.

## Command line

```bash
tdfpy validate run.d            # JSON report; exit 0 if valid, 1 if not
tdfpy validate run.d --full     # also decode and check every frame
tdfpy-mcp --data-root DIR --output-dir OUT   # MCP stdio server, needs tdfpy[mcp]
python -m tdfpy.mcp --data-root DIR --output-dir OUT
```

## Gotchas

- Spectral data is lazy. Frames, precursors, windows and transitions hold the
  reader's open connection; using them after the `with` block raises
  `ReaderClosedError` (subclasses `RuntimeError`). Extract arrays inside the block.
- Unsupported formats raise instead of guessing: legacy compression type 1,
  `use_recalibrated_state=True` and pressure compensation raise
  `UnsupportedTdfError`; unknown m/z or mobility calibration model types raise
  `UnsupportedCalibrationError`. Do not catch-and-ignore these.
- `get_acquisition_type` returns `"unknown"` for an unrecognised mode but raises
  `FileNotFoundError` if `analysis.tdf` is missing.
- DIA and PRM MS2 ion mobility is the precursor's mobility (the TIMS cell sits
  before fragmentation), not a fragment property.
- The first centroiding call JIT-compiles Numba kernels (a few seconds); later
  calls are fast. `use_numba=False` forces the pure-Python fallback.
- Noise filtering happens before centroiding. There are no post-centroid filters;
  to suppress isolated noise after merging, raise `min_peaks` on the centroider.
- Algorithm settings (`MergePeaksCentroider`, `WatershedCentroider`, noise filters,
  `ChargeStateRegion`, `Smooth`) are frozen dataclasses: hashable, safe as cache keys.
- `scan_peaks()` returns a list per mobility scan, not one array; use `.centroid()`
  for a single spectrum.
- Every error tdfpy raises subclasses `TdfpyError` (a `ValueError`); a missing id
  raises `TdfpyKeyError` (also a `KeyError`). Frame elements are frozen and
  keyword-only; lookup `[]` for DIA windows and PRM transitions returns a tuple.

"""


def clean_type(text: str) -> str:
    """Shorten fully-qualified names in a rendered signature or annotation."""
    text = re.sub(r"numpy\.ndarray\[tuple\[typing\.Any, \.\.\.\], numpy\.dtype\[([\w.]+)\]\]", r"NDArray[\1]", text)
    text = re.sub(r"pathlib(\._local)?\.Path", "Path", text)
    text = text.replace("numpy.ndarray", "np.ndarray")
    text = re.sub(r"tdfpy\.[a-z_]+(\.[a-z_]+)?\.", "", text)
    text = text.replace("collections.abc.", "")
    return text.replace(NOISE_SPEC_EXPANDED, "NoiseSpec")


def signature(obj: Any) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""
    return clean_type(sig).replace("(self, ", "(").replace("(self)", "()")


def annotation(ann: Any) -> str:
    if ann is inspect.Signature.empty:
        return ""
    if isinstance(ann, type):
        return ann.__name__
    return clean_type(str(ann))


def summary(obj: Any) -> str:
    """First paragraph of the docstring, on one line."""
    doc = inspect.getdoc(obj) or ""
    return doc.strip().split("\n\n")[0].replace("\n", " ").strip()


def resolve(path: str) -> Any:
    """Import ``a.b.C.d`` by importing the longest module prefix, then getattr."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(path)


def _describe_class(cls: type, name: str) -> list[str]:
    is_dc = dataclasses.is_dataclass(cls)
    is_exc = issubclass(cls, BaseException)
    kind = "exception" if is_exc else ("dataclass" if is_dc else "class")
    bases = [b.__name__ for b in cls.__bases__ if b is not object and not b.__name__.startswith("_") and b.__name__ != "Generic"]
    head = f"`{name}`" if is_dc or is_exc else f"`{name}{signature(cls)}`"
    out = [f"**{kind} {head}**" + (f" (subclass of {', '.join(bases)})" if bases else "")]
    if text := summary(cls):
        out.append(f"  {text}")

    # A class with a public tdfpy base only lists what it adds; the base is
    # documented on its own.
    own_only = any(b.__module__.startswith("tdfpy") and not b.__name__.startswith("_") for b in cls.__mro__[1:])
    field_names: set[str] = set()
    if dataclasses.is_dataclass(cls):
        all_fields = dataclasses.fields(cls)
        field_names = {f.name for f in all_fields}
        shown = [f for f in all_fields if not f.name.startswith("_")]
        if own_only:
            inherited = {f.name for b in cls.__mro__[1:] if dataclasses.is_dataclass(b) for f in dataclasses.fields(b)}
            shown = [f for f in shown if f.name not in inherited]
        if shown:
            out.append("  - fields: " + ", ".join(_field(f) for f in shown))
        if "_timsdata" in field_names:
            out.append("  - returned by the readers; do not construct directly")

    for mname in sorted(dir(cls)):
        if mname.startswith("_") and mname not in SHOWN_DUNDERS:
            continue
        if mname in field_names or mname == "timsdata":
            continue
        owner = next((b for b in cls.__mro__ if mname in b.__dict__), None)
        if owner is None or not owner.__module__.startswith("tdfpy"):
            continue
        if own_only and owner is not cls and not owner.__name__.startswith("_"):
            continue
        if mname.startswith("__") and owner.__name__.startswith("_"):
            continue
        raw = inspect.getattr_static(cls, mname)
        if isinstance(raw, property) or type(raw).__name__ == "cached_property":
            fget = raw.fget if isinstance(raw, property) else raw.func
            ret = annotation(inspect.signature(fget).return_annotation)
            desc = summary(fget)
            out.append(f"  - `.{mname}`" + (f" -> `{ret}`" if ret else "") + (f": {desc}" if desc else ""))
        elif callable(raw) or isinstance(raw, (staticmethod, classmethod)):
            member = getattr(cls, mname)
            out.append(f"  - `.{mname}{signature(member)}`: {summary(member)}")
    if own_only and bases:
        out.append(f"  - inherits all fields and methods of {', '.join(bases)}")
    return out


def _field(f: dataclasses.Field) -> str:
    ftype = f.type if isinstance(f.type, str) else (f.type.__name__ if isinstance(f.type, type) else str(f.type))
    text = f"`{f.name}: {clean_type(str(ftype))}`"
    if f.default is not dataclasses.MISSING:
        text += f" = {f.default!r}"
    return text


def describe(path: str) -> str:
    """Render one ``::: path`` directive as Markdown."""
    obj = resolve(path)
    name = path.split(".")[-1]
    if inspect.isclass(obj):
        return "\n".join(_describe_class(obj, name))
    if callable(obj):
        out = [f"**`{name}{signature(obj)}`**"]
        if text := summary(obj):
            out.append(f"  {text}")
        return "\n".join(out)
    return f"**`{name}`** = `{obj!r}`"


def expand_page(text: str, seen: set[str]) -> str:
    """Replace ``::: target`` directives (and their indented options) in a page."""
    out: list[str] = []
    in_options = False
    for line in text.split("\n"):
        if m := DIRECTIVE.match(line):
            seen.add(m.group(1).split(".")[-1])
            out.append(describe(m.group(1)))
            in_options = True
            continue
        if in_options and line.startswith("    "):
            continue  # mkdocstrings options block under the directive
        in_options = False
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def build() -> str:
    parts = [HEADER]
    seen: set[str] = set()
    for rel in DOCS_ORDER:
        page = (DOCS_DIR / rel).read_text(encoding="utf-8")
        parts.append(f"\n{RULE}\n# docs/{rel}\n{RULE}\n\n")
        parts.append(expand_page(page, seen))

    missing = [n for n in tdfpy.__all__ if n not in seen]
    if missing:
        parts.append(f"\n{RULE}\n# Other public names (exported from tdfpy, not on a docs page)\n{RULE}\n\n")
        parts.extend(describe(f"tdfpy.{n}") + "\n\n" for n in missing)
    return re.sub(r"\n{3,}", "\n\n", "".join(parts))


def main() -> None:
    text = build()
    for out_path in OUTPUTS:
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({len(text.encode()) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
