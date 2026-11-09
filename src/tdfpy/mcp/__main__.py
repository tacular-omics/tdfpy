"""``python -m tdfpy.mcp``: the optional MCP interface's command line.

Guarded like :mod:`tdfpy.__main__`, so importing this module does not run the
server's argument parser. Without the guard, any import of it (a module walk, a
docstring-example runner, ``pydoc``) hits ``argparse`` on the importer's argv and
exits the process.
"""

from . import main

if __name__ == "__main__":  # pragma: no cover - the entry point, exercised by `python -m tdfpy.mcp`
    main()
