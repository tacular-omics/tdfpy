"""Drift guard for the MCP vocabulary: tool schemas and records use tdfpy 5.0 names.

Walks every tool's input and output schema, bans the pre-5.0 and non-shared names,
checks the shared ``Polarity`` / ``ToleranceUnit`` values, requires unknown arguments to
be rejected, and compares returned record keys with the library's attribute names.
"""

import asyncio
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args

import pytest

pytest.importorskip("mcp")

from mcp import Client

from tdfpy import DDA, Frame, Polarity, Precursor, ToleranceUnit
from tdfpy.elems import DiaWindow
from tdfpy.mcp.server import create_server

DATA = Path(__file__).parent / "data"

# Old or non-shared spellings. The shared vocabulary is rt / rt_range, ook0, *_tolerance_unit,
# precursor_mz / isolation_mz, window_group_id, mobility_type.
BANNED = re.compile(
    r"^(unit|tolerance_type|.*_tolerance_type|retention_time.*|inverse_reduced.*|ion_mobility_.*|target_mz|scan_start_time|"
    r"ce|tic|TIC|time|Time|one_over_k0.*|mz_begin|mz_end|window_group|monoisotopic_mz|"
    r"SummedIntensities|MaxIntensity|MzCalibration|TimsCalibration|PropertyGroup|Id)$"
)
# Names that match BANNED but are correct here. Empty today; add with a reason.
ALLOWED: set[str] = set()
# Keys the MCP adds to library records: the spectrum selection object, and the length of a
# tuple field (``pasef_frame_msms_infos`` is summarised as a count rather than inlined).
MCP_ONLY_KEYS = {"selection"}

# Minimal valid arguments for every tool. A new tool must be added here.
MINIMAL_ARGS: dict[str, dict[str, Any]] = {
    "server_info": {},
    "discover_acquisitions": {},
    "inspect_acquisition": {"acquisition": "example_dia.d"},
    "list_metadata_tables": {"acquisition": "example_dia.d"},
    "read_metadata_table": {"acquisition": "example_dia.d", "table": "Frames"},
    "query_frames": {"acquisition": "example_dia.d"},
    "query_precursors": {"acquisition": "example_dda.d"},
    "query_dia_windows": {"acquisition": "example_dia.d"},
    "query_prm_targets": {"acquisition": "example_prm.d"},
    "query_prm_transitions": {"acquisition": "example_prm.d"},
    "get_processing_options": {},
    "preview_spectrum": {"acquisition": "example_dia.d", "selection": {"kind": "frame", "id": 1}},
    "export_spectrum": {"acquisition": "example_dia.d", "selection": {"kind": "frame", "id": 1}},
    "export_window_batch": {"acquisition": "example_dia.d", "indices": [0]},
    "read_artifact": {"artifact_id": "0" * 32},
    "convert_coordinates": {"acquisition": "example_dia.d", "frame_id": 1, "conversion": "tof_to_mz", "values": [100.0]},
    "check_acquisition": {"acquisition": "example_dia.d"},
    "check_frames": {"acquisition": "example_dia.d"},
}


def _walk(schema: Any, path: str, out: list[tuple[str, str, dict]], defs: dict, seen: set[str]) -> None:
    """Collect ``(path, name, subschema)`` for every property, following ``$ref`` into ``$defs`` once."""
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        if name not in seen:
            seen.add(name)
            _walk(defs.get(name, {}), f"{path}<{name}>", out, defs, seen)
    for key, sub in (schema.get("properties") or {}).items():
        out.append((f"{path}.{key}", key, sub))
        _walk(sub, f"{path}.{key}", out, defs, seen)
    for key in ("items", "anyOf", "oneOf", "allOf", "additionalProperties"):
        value = schema.get(key)
        for sub in value if isinstance(value, list) else [value]:
            _walk(sub, path, out, defs, seen)


def _enum(sub: dict) -> set:
    """Union of ``enum`` / ``const`` values, looking through ``anyOf`` (Optional)."""
    values: set = set()
    for option in [sub, *sub.get("anyOf", [])]:
        values |= set(option.get("enum", []))
        if "const" in option:
            values.add(option["const"])
    return values


def _run(server, coroutine_function):
    async def main():
        async with Client(server, read_timeout_seconds=15) as client:
            return await coroutine_function(client)

    return asyncio.run(main())


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    return create_server([DATA], tmp_path_factory.mktemp("mcp") / "output")


@pytest.fixture(scope="module")
def properties(server) -> list[tuple[str, str, dict]]:
    async def collect(client):
        found: list[tuple[str, str, dict]] = []
        for tool in (await client.list_tools()).tools:
            for kind, schema in (("in", tool.input_schema), ("out", tool.output_schema or {})):
                _walk(schema, f"{tool.name}:{kind}", found, schema.get("$defs", {}), set())
        return found

    return _run(server, collect)


def test_no_banned_names(properties):
    bad = [path for path, name, _ in properties if BANNED.match(name) and name not in ALLOWED]
    assert not bad, f"old or non-shared names in MCP schemas: {bad}"


def test_tolerance_switches_and_units(properties):
    for path, name, sub in properties:
        if name.endswith("unit"):
            assert name == "tolerance_unit" or name.endswith("_tolerance_unit"), path
            assert _enum(sub) <= set(get_args(ToleranceUnit)), path


def test_shared_enums(properties):
    polarity = [(path, _enum(sub)) for path, name, sub in properties if name == "polarity"]
    assert polarity, "query_frames should take polarity"
    for path, values in polarity:
        assert values == set(get_args(Polarity)), path
    mobility = [(path, _enum(sub)) for path, name, sub in properties if name == "mobility_type"]
    assert mobility, "Processing should take mobility_type"
    for path, values in mobility:
        # Same parameter name as mzmlpy's MCP. "drift_time" is not a TIMS quantity, and CCS
        # needs a charge that raw spectra do not have, so the MCP offers 1/K0 and voltage.
        assert values == {"ook0", "voltage"}, path


def test_every_tool_has_minimal_args(server):
    async def names(client):
        return {tool.name for tool in (await client.list_tools()).tools}

    assert _run(server, names) == set(MINIMAL_ARGS)


def test_unknown_argument_rejected(server):
    async def call_all(client):
        results = {}
        for name, args in MINIMAL_ARGS.items():
            results[name] = await client.call_tool(name, {**args, "__bogus__": 1})
        # The 4.x name of the query_frames range must not silently return unfiltered rows.
        results["query_frames(rt=)"] = await client.call_tool("query_frames", {"acquisition": "example_dia.d", "rt": {"lower": 0, "upper": 1}})
        return results

    for name, result in _run(server, call_all).items():
        assert result.is_error, name
        text = result.content[0].text
        assert "Extra inputs are not permitted" in text, (name, text)


def _library_keys(cls: type) -> set[str]:
    names = {name for name in dir(cls) if not name.startswith("_")}
    if is_dataclass(cls):
        # A tuple field is summarised as ``<field>_count``.
        names |= {f"{f.name}_count" for f in fields(cls) if not f.name.startswith("_")}
    return names


def test_query_frames_records_use_frame_names(server):
    async def query(client):
        return await client.call_tool("query_frames", {"acquisition": "example_dda.d", "msms_type": 0, "limit": 1})

    result = _run(server, query)
    assert not result.is_error
    row = result.structured_content["items"][0]
    assert set(row) <= _library_keys(Frame) | MCP_ONLY_KEYS
    with DDA(DATA / "example_dda.d") as reader:
        frame = reader.ms1[row["frame_id"]]
        for name, value in row.items():
            if name != "selection":
                assert value == getattr(frame, name), name
    assert row["polarity"] in get_args(Polarity)
    assert row["selection"] == {"kind": "frame", "id": row["frame_id"]}


@pytest.mark.parametrize(
    ("tool", "acquisition", "cls"),
    [("query_precursors", "example_dda.d", Precursor), ("query_dia_windows", "example_dia.d", DiaWindow)],
)
def test_entity_records_use_library_names(server, tool, acquisition, cls):
    async def query(client):
        return await client.call_tool(tool, {"acquisition": acquisition, "limit": 2})

    result = _run(server, query)
    assert not result.is_error
    for row in result.structured_content["items"]:
        assert set(row) <= _library_keys(cls) | MCP_ONLY_KEYS


def test_coordinate_frame_polarity_is_shared_vocabulary(server):
    async def convert(client):
        return await client.call_tool("convert_coordinates", MINIMAL_ARGS["convert_coordinates"])

    result = _run(server, convert)
    assert not result.is_error
    assert result.structured_content["frame"]["polarity"] in get_args(Polarity)


def test_server_info_units_use_rt(server):
    async def info(client):
        return await client.call_tool("server_info", {})

    assert set(_run(server, info).structured_content["units"]) == {"rt", "mobility", "intensity"}


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("query_precursors", {"acquisition": "example_dia.d"}, "require DDA"),
        ("preview_spectrum", {"acquisition": "example_dia.d", "selection": {"kind": "frame", "id": 999999}}, "not found"),
        ("inspect_acquisition", {"acquisition": "missing.d"}, "missing.d"),
        ("read_artifact", {"artifact_id": "../analysis.tdf"}, "artifact_id"),
    ],
)
def test_library_errors_reach_model(server, tool, args, message):
    """Library errors are sent as ToolError text, not the SDK's bare "Error executing tool"."""

    async def call(client):
        return await client.call_tool(tool, args)

    result = _run(server, call)
    assert result.is_error
    text = result.content[0].text
    assert message in text, text
    assert text != f"Error executing tool {tool}"
