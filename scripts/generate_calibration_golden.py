"""Guard the independent calibration references against self-generation.

Reference capture must use an independently verified external implementation.
The former generator imported tdfpy itself after removal of the native reader.
It must not overwrite vendor reference values with values under test.
"""


def main() -> None:
    raise SystemExit(
        "Calibration reference generation is disabled. Capture values with an "
        "independent external reference implementation and record its version, "
        "fixture checksums, and calibration settings in a separate output file. "
        "Do not overwrite tests/data/calibration_golden.json with tdfpy output."
    )


if __name__ == "__main__":
    main()
