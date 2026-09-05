"""Optional MCP contracts tested against the real acquisition fixtures."""

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters
from pydantic import ValidationError

from tdfpy import DDA, DIA, PRM, TimsData, MergePeaksCentroider
from tdfpy.mcp.models import (
    Interval,
    Operation,
    Predicate,
    Processing,
    SpectrumSelection,
)
from tdfpy.mcp.server import create_server
from tdfpy.mcp.service import AcquisitionService, _options, processing_options

DATA = Path(__file__).parent / "data"


@pytest.fixture
def service(tmp_path):
    return AcquisitionService([DATA], tmp_path / "output")


@pytest.fixture
def config():
    return Processing(
        centroider=Operation(name="MergePeaksCentroider", parameters={"max_peaks": 10})
    )


def test_core_import_keeps_optional_features_private():
    code = """import sys
import tdfpy
assert 'mcp' not in sys.modules
assert 'pydantic' not in sys.modules
for name in ('process_frame', 'extract_chromatogram', 'ProcessedSpectrum', 'WindowSpectrum'):
    assert not hasattr(tdfpy, name)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_discovery_schema_metadata_and_half_open_frame_queries(service):
    first = service.discover(1, 0, 2)
    second = service.discover(1, first["next_offset"], 2)
    assert first["total"] == 3
    assert len(first["items"]) + len(second["items"]) == 3
    info = service.inspect("example_dda.d")
    assert info["acquisition_type"] == "DDA"
    assert info["summary"]["frames"] == 710
    tables = service.tables("example_dda.d")
    assert "MzCalibration" in {t["name"] for t in tables["tables"]}
    page = service.read_table("example_dda.d", "Frames", ["Id", "Time"], [], 0, 2)
    a, b = page["rows"]
    selected = service.read_table(
        "example_dda.d",
        "Frames",
        ["Id"],
        [
            Predicate(column="Time", operator="ge", value=a["Time"]),
            Predicate(column="Time", operator="lt", value=b["Time"]),
        ],
        0,
        10,
    )
    assert selected["rows"] == [{"Id": a["Id"]}]
    assert not selected["next_offset"]
    with pytest.raises(ValueError, match="Unknown table"):
        service.read_table("example_dda.d", 'Frames" UNION SELECT 1', None, [], 0, 1)
    with pytest.raises(ValueError, match="Unknown column"):
        service.read_table(
            "example_dda.d", "Frames", None, [Predicate(column="bad")], 0, 1
        )


@pytest.mark.parametrize(
    "mode,kind",
    [
        ("dda", "precursor"),
        ("dia", "dia_window"),
        ("prm", "prm_target"),
        ("prm", "prm_transition"),
    ],
)
def test_entity_queries_page_and_select_actual_ids(service, mode, kind):
    path = f"example_{mode}.d"
    page = service.entities(path, kind, None, None, None, 0, 1)
    row = page["items"][0]
    next_page = service.entities(path, kind, None, None, None, 1, 1)
    assert next_page["items"][0] != row
    rt = row.get("rt", row.get("time"))
    selected = service.entities(
        path, kind, Interval(lower=rt, upper=rt), None, None, 0, 1
    )
    assert selected["total"] == 0
    if kind != "prm_target":
        assert SpectrumSelection.model_validate(row["selection"]).kind == kind
    with pytest.raises(ValueError, match="queries require"):
        service.entities(
            "example_dia.d" if mode != "dia" else "example_dda.d",
            kind,
            None,
            None,
            None,
            0,
            1,
        )


@pytest.mark.parametrize("mode,reader_cls", [("dda", DDA), ("dia", DIA), ("prm", PRM)])
@pytest.mark.parametrize("raw", [True, False])
def test_spectrum_exports_match_python_api(service, config, mode, reader_cls, raw):
    config = Processing(mode="raw") if raw else config
    path = f"example_{mode}.d"
    with reader_cls(DATA / path) as reader:
        frame = next(iter(reader.ms1))
        expected = (
            frame.raw_peaks()
            if raw
            else frame.centroid(centroid=MergePeaksCentroider(max_peaks=10))
        )
        selection = SpectrumSelection(kind="frame", id=frame.frame_id)
    peaks, metadata = service.spectrum(path, selection, config)
    np.testing.assert_array_equal(peaks, expected)
    preview = service.preview(peaks, metadata, 2)
    assert len(preview["preview"]) <= 2
    assert preview["peak_count"] == len(expected)
    assert preview["intensity_sum"] == expected[:, 1].sum()
    artifact = service.write_artifact(iter([("peaks", peaks, metadata)]))
    with np.load(artifact["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["peaks"], expected)
    with open(artifact["path"], "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == artifact["sha256"]
    manifest = service.artifact(artifact["artifact_id"], None, 0, 2)
    assert manifest["spectra"][0]["shape"] == list(expected.shape)
    rows = service.artifact(artifact["artifact_id"], "peaks", 0, 2)
    assert rows["rows"] == expected[:2].tolist()


@pytest.mark.parametrize(
    "mode,reader_cls,kind,attribute",
    [
        ("dia", DIA, "dia_window", "windows"),
        ("prm", PRM, "prm_transition", "transitions"),
    ],
)
def test_window_extraction_and_batch_export(
    service, config, mode, reader_cls, kind, attribute
):
    path = f"example_{mode}.d"
    indices = [1, 0, 2]
    artifact = service.export_batch(path, indices, config)
    with (
        reader_cls(DATA / path) as reader,
        np.load(artifact["path"], allow_pickle=False) as archive,
    ):
        windows = list(getattr(reader, attribute))
        manifest = json.loads(str(archive["metadata"]))
        for number, index in enumerate(indices):
            expected = windows[index].centroid(
                centroid=MergePeaksCentroider(max_peaks=10)
            )
            np.testing.assert_array_equal(archive[f"spectrum_{number:05d}"], expected)
            assert manifest["spectra"][number]["metadata"]["selection"] == {
                "kind": kind,
                "id": index,
            }
    peaks, _ = service.spectrum(path, SpectrumSelection(kind=kind, id=0), config)
    with reader_cls(DATA / path) as reader:
        np.testing.assert_array_equal(
            peaks,
            next(iter(getattr(reader, attribute))).centroid(
                centroid=MergePeaksCentroider(max_peaks=10)
            ),
        )


def test_precursor_uses_existing_picker_and_rejects_ignored_options(service, config):
    row = service.entities("example_dda.d", "precursor", None, None, None, 0, 1)[
        "items"
    ][0]
    selection = SpectrumSelection.model_validate(row["selection"])
    peaks, metadata = service.spectrum("example_dda.d", selection, Processing())
    with DDA(DATA / "example_dda.d") as reader:
        np.testing.assert_array_equal(peaks, reader.precursors[selection.id].peaks)
    assert metadata["columns"] == ["mz", "intensity"]
    with pytest.raises(ValueError, match="Omit processing overrides"):
        service.spectrum("example_dda.d", selection, config)


def test_physical_output_selection_is_explicit_and_empty_is_valid(service):
    selection = SpectrumSelection(
        kind="frame",
        id=1,
        mz_range=Interval(lower=500, upper=600),
        mobility_range=Interval(lower=0.8, upper=1.2),
    )
    actual, _ = service.spectrum("example_dia.d", selection, Processing(mode="raw"))
    full, _ = service.spectrum(
        "example_dia.d", SpectrumSelection(kind="frame", id=1), Processing(mode="raw")
    )
    mask = (
        (full[:, 0] >= 500)
        & (full[:, 0] < 600)
        & (full[:, 2] >= 0.8)
        & (full[:, 2] < 1.2)
    )
    np.testing.assert_array_equal(actual, full[mask])
    empty, metadata = service.spectrum(
        "example_dia.d",
        SpectrumSelection(kind="frame", id=1, scan_begin=0, scan_end=0),
        Processing(mode="raw"),
    )
    result = service.preview(empty, metadata, 5)
    assert result["peak_count"] == 0 and result["mz_range"] is None
    with pytest.raises(ValueError, match="not found"):
        service.spectrum(
            "example_dia.d",
            SpectrumSelection(kind="frame", id=999999),
            Processing(mode="raw"),
        )


def test_calibration_conversions_match_frame_and_roundtrip_ccs(service):
    with TimsData(DATA / "example_dia.d") as td:
        expected = td.indexToMz(1, [100, 1000])
    result = service.convert("example_dia.d", 1, "tof_to_mz", [100, 1000], None, None)
    np.testing.assert_array_equal(result["values"], expected)
    inverse = service.convert(
        "example_dia.d", 1, "mz_to_tof", result["values"], None, None
    )
    np.testing.assert_allclose(inverse["values"], [100, 1000])
    ccs = service.convert("example_dia.d", 1, "ook0_to_ccs", [1.0], 500, 2)
    roundtrip = service.convert(
        "example_dia.d", 1, "ccs_to_ook0", ccs["values"], 500, 2
    )
    assert roundtrip["values"] == pytest.approx([1.0])
    with pytest.raises(ValueError, match="explicit positive charge"):
        service.convert("example_dia.d", 1, "ook0_to_ccs", [1.0], 500, None)


def test_operations_reject_unknowns_and_preserve_core_defaults():
    catalog = processing_options()
    for group, schemas in catalog["groups"].items():
        assert schemas
        for name in schemas:
            field = {
                "centroiders": "centroider",
                "noise": "noise",
                "smoothing": "smoothing",
                "exclusion": "exclusion",
            }[group]
            spec = Operation(
                name=name,
                parameters={},
            )
            config = Processing(**{field: [spec] if field == "noise" else spec})
            _options(config)
    for operation in [
        Operation(name="eval"),
        Operation(name="MergePeaksCentroider", parameters={"typo": 1}),
        Operation(name="MergePeaksCentroider", parameters={"max_peaks": 1.5}),
    ]:
        with pytest.raises(ValueError):
            _options(Processing(centroider=operation))
    with pytest.raises(ValidationError):
        Processing(mode="raw", centroider=Operation(name="MergePeaksCentroider"))


def test_paths_and_artifact_failures_do_not_modify_sources(
    service, tmp_path, monkeypatch
):
    from tdfpy.mcp import service as module

    with pytest.raises(ValueError, match="outside"):
        service.acquisition(str(tmp_path))
    with pytest.raises(ValueError, match="artifact_id"):
        service.artifact("../analysis.tdf", None, 0, 1)
    with pytest.raises(ValueError, match="outside acquisition"):
        AcquisitionService([DATA], tmp_path / "source.d" / "output")
    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="Export exceeds"):
        service.write_artifact(iter([("peaks", np.ones((1, 3)), {})]))
    assert not list(service.output_dir.iterdir())
    with pytest.raises(ValueError, match="max-frame-peaks"):
        limited = AcquisitionService([DATA], tmp_path / "limited", 1)
        limited.spectrum(
            "example_dia.d",
            SpectrumSelection(kind="frame", id=1),
            Processing(mode="raw"),
        )


def test_symlink_acquisition_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "escape.d").symlink_to(DATA / "example_dia.d", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is unavailable")
    service = AcquisitionService([root], tmp_path / "output")
    with pytest.raises(ValueError, match="outside"):
        service.acquisition("escape.d")
    assert service.discover(2, 0, 10)["items"] == []


def test_validation_pages_and_corruption(service, tmp_path):
    from tdfpy import slice_d_folder

    assert service.check("example_dia.d")["valid"]
    result = service.check_frames("example_dia.d", 0, 2)
    assert result["passed"] and result["frames_checked"] == [1, 2]
    assert result["next_offset"] == 2
    source = slice_d_folder(DATA / "example_dda.d", tmp_path / "bad.d", 1, 1)
    binary = source / "analysis.tdf_bin"
    binary.write_bytes(binary.read_bytes()[:8])
    scoped = AcquisitionService([tmp_path], tmp_path / "output")
    report = scoped.check_frames(str(source), 0, 2)
    assert not report["passed"]
    assert report["issues"][0]["frame_id"] == 1


def test_protocol_tool_schemas_errors_resources_and_prompts(tmp_path):
    async def run():
        server = create_server([DATA], tmp_path / "output")
        async with Client(server, read_timeout_seconds=15) as client:
            listed = await client.list_tools()
            tools = {t.name: t for t in listed.tools}
            assert len(tools) == 18
            assert tools["preview_spectrum"].annotations.read_only_hint
            assert not tools["export_spectrum"].annotations.read_only_hint
            assert not tools["export_spectrum"].annotations.destructive_hint
            result = await client.call_tool("server_info", {})
            assert not result.is_error
            assert result.structured_content["transport"] == "stdio"
            bad = await client.call_tool(
                "query_frames", {"acquisition": "example_dia.d", "limit": 10000}
            )
            assert bad.is_error
            resources = await client.list_resources()
            assert "tdfpy://guide" in {str(r.uri) for r in resources.resources}
            guide = await client.read_resource("tdfpy://guide")
            assert "half-open" in guide.contents[0].text
            prompts = await client.list_prompts()
            assert len(prompts.prompts) == 2
            prompt = await client.get_prompt(
                "inspect_timstof", {"acquisition": "example_dia.d"}
            )
            assert prompt.messages

    asyncio.run(run())


def test_stdio_server_launch_and_real_extraction(tmp_path):
    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "tdfpy.mcp",
                "--data-root",
                str(DATA),
                "--output-dir",
                str(tmp_path / "output"),
            ],
        )
        async with Client(params, read_timeout_seconds=20) as client:
            query = await client.call_tool(
                "query_dia_windows", {"acquisition": "example_dia.d", "limit": 1}
            )
            assert not query.is_error
            selection = query.structured_content["items"][0]["selection"]
            result = await client.call_tool(
                "preview_spectrum",
                {
                    "acquisition": "example_dia.d",
                    "selection": selection,
                    "processing": {"mode": "raw"},
                    "preview_limit": 3,
                },
            )
            assert not result.is_error
            assert result.structured_content["peak_count"] > 3
            assert len(result.structured_content["preview"]) == 3

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["raw", "centroid"])
def test_filter_chain_and_watershed_match_core(service, mode):
    from tdfpy import AbsoluteThreshold, DiaMs1WindowGate, Smooth, WatershedCentroider

    config = Processing(
        mode=mode,
        centroider=Operation(name="WatershedCentroider")
        if mode == "centroid"
        else None,
        noise=[
            Operation(name="DiaMs1WindowGate"),
            Operation(name="AbsoluteThreshold", parameters={"value": 100.0}),
        ],
        smoothing=Operation(
            name="Smooth", parameters={"scan_half_width": 0, "mz_idx_half_width": 1}
        ),
    )
    peaks, _ = service.spectrum(
        "example_dia.d", SpectrumSelection(kind="frame", id=1), config
    )
    with DIA(DATA / "example_dia.d") as reader:
        options = {
            "noise": [DiaMs1WindowGate(), AbsoluteThreshold(100)],
            "smooth": Smooth(0, 1),
        }
        expected = (
            reader.ms1[1].raw_peaks(**options)
            if mode == "raw"
            else reader.ms1[1].centroid(centroid=WatershedCentroider(), **options)
        )
    np.testing.assert_array_equal(peaks, expected)


def test_duplicate_artifact_id_never_removes_an_existing_file(service, monkeypatch):
    from tdfpy.mcp import service as module

    token = module.uuid4()
    monkeypatch.setattr(module, "uuid4", lambda: token)
    artifact = service.write_artifact(iter([("peaks", np.ones((1, 3)), {})]))
    before = Path(artifact["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        service.write_artifact(iter([("peaks", np.zeros((1, 3)), {})]))
    assert Path(artifact["path"]).read_bytes() == before
