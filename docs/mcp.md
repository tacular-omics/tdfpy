Optional MCP interface

The MCP server lets an AI agent inspect and extract timsTOF data through the
same reader and centroiding code used by Python callers. It is a local stdio
server. An MCP client launches it as a subprocess. Source acquisitions remain
read-only, and complete spectrum exports go into a separate output directory.

The server is available starting in tdfpy 4.0.0. Install it with
`pip install 'tdfpy[mcp]'`. From a checkout, install and launch it with:

```bash
uv sync --extra mcp
uv run --extra mcp tdfpy-mcp --data-root /data/timstof --output-dir /data/tdfpy-results
```

The launch command is `tdfpy-mcp`, or `python -m tdfpy.mcp`. Ordinary `pip install tdfpy` does not install the MCP
SDK, and importing `tdfpy` does not import MCP or Pydantic.

Repeat `--data-root` to expose several input directories. With one root, tools
accept acquisition paths relative to it. With multiple roots, use absolute
paths returned by discovery. The output directory must be outside acquisition
folders. Paths in the launch configuration refer to the machine running the
server, which is also where the data must be accessible.

An MCP client that accepts an `mcpServers` configuration can use this entry for
an installed environment. Replace every placeholder with an absolute path:

```json
{
  "mcpServers": {
    "tdfpy": {
      "command": "/absolute/path/to/environment/bin/tdfpy-mcp",
      "args": [
        "--data-root", "/absolute/path/to/acquisitions",
        "--output-dir", "/absolute/path/to/tdfpy-results"
      ]
    }
  }
}
```

On Windows, use the environment's `Scripts/tdfpy-mcp.exe`. For development, the
client can instead launch the absolute path to `uv` with arguments
`run --directory /absolute/path/to/tdfpy --extra mcp tdfpy-mcp`, followed by the
same data and output arguments. The client may require a different surrounding
configuration format, but the command and arguments are the same. This package
does not change client settings automatically.

What the agent can do

| Tools | Purpose |
| --- | --- |
| `server_info`, `discover_acquisitions` | Find configured roots, limits, and available acquisitions |
| `inspect_acquisition` | Summarize acquisition mode, frame types, retention-time coverage, and source files |
| `list_metadata_tables`, `read_metadata_table` | Inspect actual SQLite schema and page selected columns with typed filters |
| `query_frames` | Find frames by RT, polarity, and MS/MS type |
| `query_precursors` | Find DDA precursors by RT and precursor m/z |
| `query_dia_windows` | Find DIA windows by RT, isolation-center m/z, and window group |
| `query_prm_targets`, `query_prm_transitions` | Inspect PRM targets and find their measured transitions |
| `get_processing_options` | Discover both centroiders, ten noise filters and gates, smoothing, and region exclusion |
| `preview_spectrum` | Extract a spectrum with full-result statistics and a bounded strongest-peak preview |
| `export_spectrum` | Save a complete raw or centroided spectrum with extraction settings |
| `export_window_batch` | Save several DIA or PRM spectra while reusing adjacent windows' decoded frames |
| `read_artifact` | Inspect an export manifest or page through a numerical array |
| `convert_coordinates` | Convert TOF, m/z, scan, inverse mobility, voltage, and charge-aware CCS coordinates |
| `check_acquisition`, `check_frames` | Check supported metadata and page through binary integrity checks |

The `tdfpy://guide` resource explains units and tool sequencing. The
`tdfpy://processing` resource supplies configuration schemas, and
`tdfpy://artifacts/{artifact_id}` returns an export manifest. Two optional MCP
prompts, `inspect_timstof` and `extract_timstof`, guide common workflows. Clients
that do not expose resources or prompts can use the equivalent tools.

A typical extraction

1. Discover the acquisition and inspect its mode.
2. Query frames, precursors, or windows. Keep the returned selection object.
3. Read the processing options before setting non-default parameters.
4. Preview a small selection to check IDs, units, and processing settings.
5. Export the complete result. For adjacent DIA or PRM windows, use a batch.
6. Read the manifest or load the NPZ in Python for downstream analysis.

For example, `query_dia_windows` accepts:

```json
{
  "acquisition": "sample.d",
  "rt": {"lower": 300.0, "upper": 330.0},
  "mz": {"lower": 600.0, "upper": 650.0},
  "limit": 10
}
```

The `mz` condition matches isolation centers. It does not select every window
whose isolation band overlaps the interval. A result contains a selection such
as `{"kind": "dia_window", "id": 42}`. This ID is the window's zero-based
position in the full acquisition lookup, not a window group or frame ID. Query
pagination does not renumber it. PRM transitions use the same index convention.
Frames and DDA precursors use their actual stored IDs.

Pass the selection to `preview_spectrum` or `export_spectrum`. A processing
configuration can use the existing Python algorithm names:

```json
{
  "mode": "centroid",
  "centroider": {
    "name": "MergePeaksCentroider",
    "parameters": {"mz_tolerance": 8.0, "min_peaks": 3}
  },
  "noise": [
    {"name": "MadThreshold", "parameters": {"k": 3.0}}
  ],
  "ion_mobility_type": "ook0"
}
```

Parameters are validated against the actual algorithm definitions. Unknown
names and fields raise errors. All filters run before centroiding. The server
imposes no automatic centroid peak cap. An explicitly requested `max_peaks`
retains the underlying algorithm's seed-traversal meaning. The preview limit
only limits what appears in the tool response.

Frame selections can also specify `scan_begin` and `scan_end`. All selections
can use an optional `mz_range`, and frame or window selections can use
`mobility_range`. Each range has `lower` and `upper` fields. These physical
ranges select the output after processing. They do not change the ions seen by
the centroider. Mobility bounds use the requested output units. For changes to
pre-centroid processing, use the existing exclusion, smoothing, and noise options.

Numerical contracts

- RT is in seconds. Selection intervals and scan bounds are half-open.
- Raw mode returns digitizer peaks normalized to a 100 ms accumulation window.
  Smoothing can change intensities. Filtering and centroid thresholds can remove
  intensity. A reported intensity sum describes the selected processed result.
- Frame and window arrays have columns `[mz, intensity, ook0]` or
  `[mz, intensity, voltage]`, as stated in the export metadata.
- DDA precursor arrays have columns `[mz, intensity]`. They use the existing
  mobility-collapsed precursor picker. Processing overrides are rejected for
  these selections. Query `PasefFrameMsMsInfo` by `Precursor` to obtain individual
  frame and scan selections when custom processing is needed.
- CCS conversion requires an explicit positive charge magnitude and m/z. The
  extraction tools do not assume that raw ions have charge +1.
- Empty valid selections return empty arrays. Read failures return tool errors.
  A preview contains the strongest peaks in descending intensity order, and
  its truncation flag makes omissions explicit. Exports retain complete arrays
  in the underlying Python API's order.

Exports contain named numerical arrays and a `metadata` Unicode array containing
JSON. Load with `numpy.load(path, allow_pickle=False)`, then parse the manifest
with `json.loads(str(data["metadata"]))`. The manifest records selections,
processing requests, columns, package version, and source file identity. Source
identity uses file sizes and modification times, not acquisition content hashes.
The returned SHA256 identifies the exported artifact itself.

Limits and operation

Metadata pages contain at most 200 rows, previews at most 100 peaks, and window
batches at most 32 windows. Discovery examines at most 20,000 directories and
does not follow directory symlinks. `check_frames` processes at most 128 frames
per request. Continue using `next_offset` until it is null. A single page's
success does not certify the rest of the acquisition. Metadata-check responses
show at most 200 issues with the total issue count and an explicit truncation flag.

The default frame limit is five million stored peaks. Raise it at startup with
`--max-frame-peaks` when a known dataset requires more. A spectrum selection may
reference at most 128 frames. Each export is limited to 512 MiB of uncompressed
numerical arrays. These are workload guards, not a guarantee against malformed
binary data exhausting memory. Initial Numba compilation can take longer than
subsequent requests, so give the client an appropriate tool timeout.

Each request opens and closes its readers. There are no persistent acquisition
handles for an agent to leak. Keep input datasets unchanged during processing.
The server permits no arbitrary Python execution, arbitrary SQL, source edits,
network transport, or automatic format repair. File checks exercise tdfpy's
supported-format guards and decoder. They do not replace independently captured
vendor references or establish the scientific quality of an experiment.

Maintenance

Run `just test-mcp` for direct numerical comparisons and MCP protocol tests,
including a real stdio subprocess. The normal installed-wheel check verifies
that the core package works without MCP. The optional CI jobs exercise MCP on
Linux, Windows, and macOS. Run `uv run python scripts/verify_distribution.py dist --with-mcp` to check
both the core wheel and the optional install with its console entry point.

The implementation uses the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
