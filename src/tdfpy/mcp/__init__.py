"""Optional local MCP interface. Importing tdfpy does not import the SDK."""


def main() -> None:
    """Start the optional stdio server."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Local timsTOF tools for MCP clients")
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frame-peaks", type=int, default=5_000_000)
    args = parser.parse_args()
    try:
        from .server import create_server
    except ModuleNotFoundError as exc:
        if exc.name in {"mcp", "pydantic", "mcp_types"}:
            parser.exit(
                2, "Install the optional interface with pip install 'tdfpy[mcp]'.\n"
            )
        raise
    server = create_server(args.data_root, args.output_dir, args.max_frame_peaks)
    server.run(transport="stdio")
