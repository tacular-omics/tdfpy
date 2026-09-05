Maintenance and release verification

Run `just check` for lint, formatting verification, type checking, and tests.
Run `uv run --group docs mkdocs build --strict` for documentation validation.
The test suite executes examples from the getting-started and analysis pages.

Build distributions with `uv build --out-dir dist`, then run
`uv run python scripts/verify_distribution.py dist`. Use a clean output
directory containing one wheel and one source archive. Verification checks
the pure Python wheel and its typing marker, checks source archive exclusions,
and installs the wheel into a temporary environment outside the checkout.
It exercises DDA, DIA, and PRM extraction with the bundled fixtures. Runtime
dependencies must be available in the uv cache or from the package index.

CI tests Linux and Windows on Python 3.12, 3.13, and 3.14, plus a representative
macOS job. Coverage and JUnit output come from one pytest invocation. The release
workflow checks the tagged revision and its installed artifacts before upload.
Numba's compiled code is not fully represented by ordinary coverage tracing.

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

The publishing workflow supports PyPI Trusted Publishing behind the repository
variable `PYPI_TRUSTED_PUBLISHING=true`. First configure the existing PyPI
project's GitHub publisher for owner `tacular-omics`, repository `tdfpy`, and
workflow `python-publish.yml`. This workflow does not specify a GitHub
environment. Verify the publisher configuration, then enable the variable.
The OIDC path needs no API token. Until enabled, the existing token-based step
continues to run. After successful migration, revoke and remove the old token.
See the [PyPI setup documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

Benchmark a fixed acquisition with
`uv run python scripts/benchmark_reader.py tests/data/example_dia.d`.
The JSON output records acquisition checksums, versions, opening time, raw
decoding throughput, first and warm centroid calls, repeated-window timings,
and process peak RSS where supported. The first centroid call only measures
fresh compilation if it runs in a fresh process with an empty `NUMBA_CACHE_DIR`.
RSS includes imports and all benchmark stages. Timings are measurements, not
portable CI pass/fail thresholds.

Version 4.0.0 preserves array-returning APIs. The public
`Precursor.scan_number` now contains a float. Downstream callers that index a
scan array with that metadata value must choose an explicit rounding rule.
Corrected mobility and CCS values can differ from earlier releases. Invalid
parameters that were silently accepted now raise. Metadata database connections
are read-only. These compatibility changes are documented in the 4.0.0 release
notes. Local checks and benchmark commands do not publish releases.
