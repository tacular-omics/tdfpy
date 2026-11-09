"""Execute every docstring example in the package.

The suite already runs the fenced code blocks in ``docs/getting-started.md`` and
``docs/analysis.md`` through pytest-examples (:mod:`tests.test_docs`). That covers a
different surface: this module executes the ``>>>`` examples inside ``src/tdfpy``
docstrings, which nothing ran before.

It also has to run in the install the rest of the suite is written for. ``tests/test_mcp.py``
and ``tests/test_mcp_vocabulary.py`` open with ``pytest.importorskip("mcp")``, so a lean
install (no ``mcp`` extra) is a supported environment, and the ``tdfpy.mcp`` subtree is the
one part of the package that cannot be imported there. It is gated below, and
:func:`test_mcp_gate_matches_the_environment` is what stops the gate going wrong quietly.
"""

import doctest
import importlib
import importlib.util
import pkgutil

import pytest

import tdfpy

# Measured 2026-11-08, each count printed beside the run that produced it, in real
# environments rather than simulated ones. Against the built wheel, `pytest -q
# tests/test_doctests.py` with no extras installed walks 28 module names under `tdfpy` and
# imports 24 of them; with `tdfpy[viz,mcp]` it walks the same 28 and imports 28 (`tdfpy`
# itself plus 27 names, one being excluded below). The four gated `tdfpy.mcp` modules are
# the whole difference. Zero docstring examples either way.
# MIN_MODULES is the lean floor, the one that holds in every supported install.
# Raise these when you add modules or examples; never lower them without saying why in the
# commit message.
MIN_MODULES = 24
MIN_EXAMPLES = 0

# Walked names under `tdfpy.mcp`, inclusive of the package itself: `tdfpy.mcp`, `__main__`,
# `models`, `server`, `service`. Asserted only where the extra is installed, which is where
# a module vanishing from `mcp/` would otherwise be invisible: lean, it is skipped anyway.
MCP_SUBTREE_NAMES = 5

# The top-level modules `tdfpy.mcp` imports at module scope: `mcp` (server.py) and `pydantic`
# (models.py, server.py, service.py). Located with `find_spec`, never by catching the
# ImportError from the import itself, because catching it would also swallow a genuinely
# broken module in an install that does have the extra.
#
# `tdfpy/mcp/__init__.py` also names `mcp_types` in its ModuleNotFoundError handler. Nothing
# in the subtree imports it, so it is not part of this key: the gate asks what the subtree
# needs, not what the entry point apologises for.
MCP_EXTRA_IMPORTS = ("mcp", "pydantic")

MCP_PREFIX = "tdfpy.mcp"

# The gated modules, each of which imports something from MCP_EXTRA_IMPORTS at module scope.
MCP_GATED_MODULES = ("tdfpy.mcp.models", "tdfpy.mcp.server", "tdfpy.mcp.service")

# Excluded by name in every environment, for a reason that has nothing to do with the extra:
# importing it *runs* the MCP server's command line. `__main__.py` has no
# `if __name__ == "__main__"` guard (it cannot, that is its whole purpose), so its module
# body calls `main()`, argparse rejects the pytest argv and the process exits 2.
# `tdfpy/__main__.py` is guarded and is walked normally. Nothing else is excluded
# unconditionally: an import failure must fail this test rather than quietly drop a module
# from the count.
SKIP_MODULES = frozenset({"tdfpy.mcp.__main__"})


def _module_present(name: str) -> bool:
    """Report whether ``name`` can be located without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _mcp_extra_installed() -> bool:
    """Report whether every top-level module the ``mcp`` extra provides is installed."""
    return all(_module_present(name) for name in MCP_EXTRA_IMPORTS)


def _in_mcp_subtree(name: str) -> bool:
    """Report whether ``name`` is ``tdfpy.mcp`` or a module inside it."""
    return name == MCP_PREFIX or name.startswith(f"{MCP_PREFIX}.")


def _raise(name: str) -> None:
    """Re-raise the import error ``pkgutil.walk_packages`` would otherwise swallow."""
    raise


def _walk_names() -> list[str]:
    """Return every module name under ``tdfpy``, in walk order."""
    return [name for _finder, name, _ispkg in pkgutil.walk_packages(tdfpy.__path__, prefix=f"{tdfpy.__name__}.", onerror=_raise)]


def _all_modules(names: list[str], *, with_mcp: bool) -> list[object]:
    """Import every walked module this environment can import, ``tdfpy`` first."""
    mods: list[object] = [tdfpy]
    for name in names:
        if name in SKIP_MODULES:
            continue
        if not with_mcp and _in_mcp_subtree(name):
            continue
        mods.append(importlib.import_module(name))
    return mods


def test_mcp_gate_matches_the_environment() -> None:
    """The gate's verdict has to match what importing the gated modules actually does.

    Without this the gate is self-certifying: a key naming something no install provides
    reads as "extra absent" everywhere, the subtree is skipped in every environment, and
    the lean floor is met because that is exactly the count a correct skip produces. Then
    the two subtree floors never run, because they sit behind the same verdict. So the
    verdict is checked against the import, which is the one thing the key cannot influence.
    """
    if _mcp_extra_installed():
        for name in MCP_GATED_MODULES:
            importlib.import_module(name)
        return
    for name in MCP_GATED_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            assert missing in MCP_EXTRA_IMPORTS, (
                f"{name} is missing {missing!r}, which the gate's key {MCP_EXTRA_IMPORTS} does not name, so this is a broken module and not an absent extra"
            )
        else:
            pytest.fail(f"the gate skipped {MCP_PREFIX} but {name} imports fine: the key {MCP_EXTRA_IMPORTS} does not name what the subtree needs")


def test_docstring_examples() -> None:
    """Every ``>>>`` example in the package runs, and the counts never silently fall."""
    with_mcp = _mcp_extra_installed()
    names = _walk_names()
    if with_mcp:
        walked_mcp = [name for name in names if _in_mcp_subtree(name)]
        assert len(walked_mcp) >= MCP_SUBTREE_NAMES, (
            f"the mcp extra is installed but the walk found only {len(walked_mcp)} names under {MCP_PREFIX}: {walked_mcp}"
        )
    mods = _all_modules(names, with_mcp=with_mcp)
    assert len(mods) >= MIN_MODULES, f"module walk imported only {len(mods)} (mcp extra installed: {with_mcp})"
    if with_mcp:
        tested_mcp = [mod.__name__ for mod in mods if _in_mcp_subtree(mod.__name__)]
        assert len(tested_mcp) >= MCP_SUBTREE_NAMES - len(SKIP_MODULES), (
            f"the mcp extra is installed but only {len(tested_mcp)} of its modules were tested: {tested_mcp}"
        )
    attempted = failed = 0
    for mod in mods:
        result = doctest.testmod(mod, verbose=False)
        attempted += result.attempted
        failed += result.failed
    assert failed == 0, f"{failed} of {attempted} docstring examples failed"
    assert attempted >= MIN_EXAMPLES, f"only {attempted} examples ran, expected >= {MIN_EXAMPLES}"


def test_skipped_modules_still_exist() -> None:
    """A name in :data:`SKIP_MODULES` that the walk no longer yields is a stale exclusion."""
    walked = set(_walk_names())
    assert SKIP_MODULES <= walked, f"stale exclusions: {sorted(SKIP_MODULES - walked)}"
