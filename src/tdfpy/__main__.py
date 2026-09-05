"""Command-line acquisition validation."""

import argparse
from dataclasses import asdict
import json

from .validation import validate_acquisition


def main() -> int:
    """Validate a run, print JSON, and return 0 on success or 1 on failure."""
    parser = argparse.ArgumentParser(prog="tdfpy")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Check an acquisition without changing it"
    )
    validate.add_argument("analysis_directory")
    validate.add_argument(
        "--full", action="store_true", help="Decode and check every frame"
    )
    args = parser.parse_args()
    report = validate_acquisition(args.analysis_directory, full=args.full)
    print(json.dumps({"valid": report.valid, **asdict(report)}, indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
