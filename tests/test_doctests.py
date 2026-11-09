"""Execute every docstring example in the package.

The suite already runs the fenced code blocks in ``docs/getting-started.md`` and
``docs/analysis.md`` through pytest-examples (:mod:`tests.test_docs`). That covers a
different surface: this module executes the ``>>>`` examples inside ``src/tdfpy``
docstrings, which nothing ran before.
"""

import doctest
import importlib
import pkgutil

import tdfpy

# Measured 2026-11-08: the walk yields 28 names under `tdfpy`, one of which is excluded
# below, so 28 modules are imported and tested (`tdfpy` itself plus 27), and they hold 0
# docstring examples. Raise these when you add modules or examples; never lower them
# without saying why in the commit message.
MIN_MODULES = 28
MIN_EXAMPLES = 0

# Excluded by name because importing it *runs* the MCP server's command line: `__main__.py`
# has no `if __name__ == "__main__"` guard (it cannot, that is its whole purpose), so its
# module body calls `main()`, argparse rejects the pytest argv and the process exits 2.
# `tdfpy/__main__.py` is guarded and is walked normally. Nothing else is excluded: an
# import failure must fail this test rather than quietly drop a module from the count.
SKIP_MODULES = frozenset({"tdfpy.mcp.__main__"})


def _raise(name: str) -> None:
    """Re-raise the import error ``pkgutil.walk_packages`` would otherwise swallow."""
    raise


def _all_modules() -> list[object]:
    """Import every module in the package and return them, ``tdfpy`` first."""
    mods: list[object] = [tdfpy]
    for _finder, name, _ispkg in pkgutil.walk_packages(tdfpy.__path__, prefix=f"{tdfpy.__name__}.", onerror=_raise):
        if name in SKIP_MODULES:
            continue
        mods.append(importlib.import_module(name))
    return mods


def test_docstring_examples() -> None:
    """Every ``>>>`` example in the package runs, and the counts never silently fall."""
    mods = _all_modules()
    assert len(mods) >= MIN_MODULES, f"module walk found only {len(mods)}"
    attempted = failed = 0
    for mod in mods:
        result = doctest.testmod(mod, verbose=False)
        attempted += result.attempted
        failed += result.failed
    assert failed == 0, f"{failed} of {attempted} docstring examples failed"
    assert attempted >= MIN_EXAMPLES, f"only {attempted} examples ran, expected >= {MIN_EXAMPLES}"


def test_skipped_modules_still_exist() -> None:
    """A name in :data:`SKIP_MODULES` that the walk no longer yields is a stale exclusion."""
    walked = {name for _finder, name, _ispkg in pkgutil.walk_packages(tdfpy.__path__, prefix=f"{tdfpy.__name__}.", onerror=_raise)}
    assert SKIP_MODULES <= walked, f"stale exclusions: {sorted(SKIP_MODULES - walked)}"
