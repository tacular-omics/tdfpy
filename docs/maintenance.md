# Maintenance and release verification

## Checks

Run `just check` for lint, formatting verification, type checking, and tests.
Run `uv run --group docs mkdocs build --strict` for documentation validation.
The test suite executes examples from the getting-started and analysis pages.

## Distributions and CI

Build distributions with `uv build --out-dir dist`, then run
`uv run python scripts/verify_distribution.py dist`. Use a clean output
directory containing one wheel and one source archive. Verification checks
the pure Python wheel and its typing marker, checks source archive exclusions,
and installs the wheel into a temporary environment outside the checkout.
It exercises DDA, DIA, and PRM extraction with the bundled fixtures. Runtime
dependencies must be available in the uv cache or from the package index.

CI tests Linux on Python 3.12, 3.13, and 3.14, plus macOS and Windows on 3.13,
a lowest-dependency job, a strict docs build, and the MCP extra on all three
platforms. A separate job uploads coverage and JUnit output. The release
workflow checks the tagged revision and its installed artifacts before upload.
Numba's compiled code is not fully represented by ordinary coverage tracing.

## Reference data

Independent reference values must remain independent. The calibration generator
is disabled because its former import path now resolves to tdfpy's own reader.
Do not replace `tests/data/calibration_golden.json` or `peaks_golden.json` with
values produced by the implementation being tested. Any future external capture
must record the reference implementation version, acquisition checksums,
calibration and pressure settings, and a separate output path. Review new
captures before extending the committed references.

An independent raw-decoding reference set and broader instrument fixtures remain
external validation work. Prioritize calibration changes, negative polarity,
empty frames, and acquisition boundaries. Cross-version agreement between two
tdfpy builds is useful regression evidence, but is not vendor validation.

## Publishing

Releases publish to PyPI with Trusted Publishing (OIDC, no API token). Publishing a
GitHub release runs `.github/workflows/publish.yml`, which checks that the tag, version
and changelog agree, runs the tests, builds and verifies the distributions, and uploads
them from the `pypi` environment. See the
[PyPI trusted publisher documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Benchmarks

Benchmark a fixed acquisition with
`uv run python scripts/benchmark_reader.py tests/data/example_dia.d`.
The JSON output records acquisition checksums, versions, opening time, raw
decoding throughput, first and warm centroid calls, repeated-window timings,
and process peak RSS where supported. The first centroid call only measures
fresh compilation if it runs in a fresh process with an empty `NUMBA_CACHE_DIR`.
RSS includes imports and all benchmark stages. Timings are measurements, not
portable CI pass/fail thresholds.

## Compatibility

Changes that affect downstream callers (types, corrected values, newly rejected
parameters) are listed per release in the
[changelog](https://github.com/tacular-omics/tdfpy/blob/main/CHANGELOG.md).
Local checks and benchmark commands do not publish releases.
