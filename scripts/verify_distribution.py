"""Inspect distributions and exercise an installed wheel outside the checkout.

Usage: uv run python scripts/verify_distribution.py dist
Requires uv and access to runtime dependency wheels, either cached or online.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument(
        "--with-mcp",
        action="store_true",
        help="Also install the optional extra and exercise its stdio entry point",
    )
    args = parser.parse_args()
    wheels = list(args.dist.glob("*.whl"))
    sources = list(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("Expected exactly one wheel and one source archive.")
    wheel = wheels[0].resolve()
    assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "tdfpy/py.typed" in names
        assert not any(n.endswith((".so", ".dll", ".dylib", ".pyd")) for n in names)
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        assert not any(".d/" in n or n.endswith("analysis.tdf_bin") for n in names)
        assert any(n.endswith("/pyproject.toml") for n in names)
        assert any(n.endswith("/src/tdfpy/__init__.py") for n in names)

    fixtures = Path(__file__).resolve().parent.parent / "tests" / "data"
    with tempfile.TemporaryDirectory(prefix="tdfpy-wheel-") as tmp:
        root = Path(tmp)
        venv = root / "env"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(venv)], check=True
        )
        python = venv / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)], check=True
        )
        smoke = root / "smoke.py"
        smoke.write_text(
            """from pathlib import Path
import sys
from importlib.util import find_spec
import tdfpy

assert find_spec("mcp") is None
assert "pydantic" not in sys.modules
assert Path(tdfpy.__file__).is_relative_to(Path(sys.prefix))
fixtures = Path(sys.argv[1])
for cls, name in ((tdfpy.DDA, "dda"), (tdfpy.DIA, "dia"), (tdfpy.PRM, "prm")):
    with cls(fixtures / f"example_{name}.d") as reader:
        frame = next(iter(reader.ms1))
        raw = frame.raw_peaks()
        assert raw.shape == (frame.num_peaks, 3)
        peaks = frame.centroid(centroid=tdfpy.MergePeaksCentroider(max_peaks=5))
        assert peaks.shape[1] == 3
        assert tdfpy.validate_acquisition(reader.analysis_path).valid
print("Installed-wheel checks passed for DDA, DIA, and PRM")
""",
            encoding="utf-8",
        )
        subprocess.run([str(python), str(smoke), str(fixtures)], cwd=root, check=True)
        if args.with_mcp:
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python), f"{wheel}[mcp]"],
                check=True,
            )
            server_command = venv / (
                "Scripts/tdfpy-mcp.exe" if sys.platform == "win32" else "bin/tdfpy-mcp"
            )
            smoke.write_text(
                """import asyncio
from pathlib import Path
import sys
from mcp import Client, StdioServerParameters
import tdfpy

assert Path(tdfpy.__file__).is_relative_to(Path(sys.prefix))
async def main():
    params = StdioServerParameters(command=sys.argv[2], args=[
        "--data-root", sys.argv[1], "--output-dir", str(Path.cwd() / "exports")])
    async with Client(params, read_timeout_seconds=30) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 18
        result = await client.call_tool("preview_spectrum", {
            "acquisition": "example_dia.d", "selection": {"kind": "frame", "id": 1},
            "processing": {"mode": "raw"}, "preview_limit": 2})
        assert not result.is_error
        assert result.structured_content["peak_count"] > 2
        assert len(result.structured_content["preview"]) == 2
asyncio.run(main())
print("Installed optional MCP console and stdio extraction passed")
""",
                encoding="utf-8",
            )
            subprocess.run(
                [str(python), str(smoke), str(fixtures), str(server_command)],
                cwd=root,
                check=True,
            )
    print("Distribution verification passed")


if __name__ == "__main__":
    main()
