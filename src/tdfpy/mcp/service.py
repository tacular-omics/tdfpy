"""Local acquisition operations behind the optional MCP transport."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np
from pydantic import TypeAdapter

from tdfpy import (
    DDA,
    DIA,
    PRM,
    TimsData,
    __version__,
    get_acquisition_type,
    get_raw_peaks,
    get_centroided_spectrum,
    iter_window_spectra,
    validate_acquisition,
)
from tdfpy import noise as noise_module
from tdfpy.calibration import one_over_k0_to_ccs, ccs_to_one_over_k0
from tdfpy.pipeline import MergePeaksCentroider, WatershedCentroider, Smooth
from tdfpy.regions import ChargeStateRegion

from .models import Interval, Operation, Predicate, Processing, SpectrumSelection

CENTROIDERS = {c.__name__: c for c in (MergePeaksCentroider, WatershedCentroider)}
FILTERS = {
    name: getattr(noise_module, name)
    for name in (
        "AbsoluteThreshold",
        "MadThreshold",
        "PercentileThreshold",
        "HistogramThreshold",
        "BaselineThreshold",
        "IterativeMedianThreshold",
        "VerticalNoiseFilter",
        "HorizontalHaloFilter",
        "SelectionPolygonGate",
        "DiaMs1WindowGate",
    )
}
READERS = {"DDA": DDA, "DIA": DIA, "PRM": PRM}
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def processing_options() -> dict:
    """Describe supported configurations using their actual Python definitions."""
    groups = {
        "centroiders": CENTROIDERS,
        "noise": FILTERS,
        "smoothing": {"Smooth": Smooth},
        "exclusion": {"ChargeStateRegion": ChargeStateRegion},
    }
    return {
        "groups": {
            group: {
                name: TypeAdapter(cls).json_schema() for name, cls in registry.items()
            }
            for group, registry in groups.items()
        },
        "order": [
            "read",
            "subset_scans",
            "exclude_region",
            "smooth",
            "noise",
            "centroid",
        ],
        "defaults": "Defaults match the Python API. No output peak cap is imposed by the server.",
        "max_peaks": "MergePeaksCentroider.max_peaks limits seed traversal, not final intensity ranking.",
        "precursors": "DDA precursor extraction uses the existing mobility-collapsed picker and its defaults. Processing overrides are not supported for precursors.",
    }


def _operation(spec: Operation | None, registry: dict) -> Any:
    if spec is None:
        return None
    if spec.name not in registry:
        raise ValueError(
            f"Unknown operation {spec.name!r}. Choose from {sorted(registry)}"
        )
    cls = registry[spec.name]
    unknown = set(spec.parameters) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown parameters for {spec.name}: {sorted(unknown)}")
    # JSON strict mode accepts JSON arrays for tuples, while rejecting string
    # numbers and booleans where the algorithm expects a numeric parameter.
    return TypeAdapter(cls).validate_json(
        json.dumps(spec.parameters, allow_nan=False), strict=True
    )


def _options(config: Processing) -> dict:
    options = {
        "exclude": _operation(
            config.exclusion, {"ChargeStateRegion": ChargeStateRegion}
        ),
        "smooth": _operation(config.smoothing, {"Smooth": Smooth}),
        "noise": tuple(_operation(spec, FILTERS) for spec in config.noise),
        "ion_mobility_type": config.ion_mobility_type,
    }
    if config.mode == "centroid":
        options["centroid"] = _operation(config.centroider, CENTROIDERS)
    return options


def _scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return _scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, bytes):
        return {"blob_bytes": len(value), "omitted": True}
    if isinstance(value, str) and len(value) > 1000:
        return {"prefix": value[:1000], "characters": len(value), "truncated": True}
    return value


def _metadata(obj: Any) -> dict:
    result = {}
    for field in fields(obj):
        if field.name.startswith("_"):
            continue
        value = getattr(obj, field.name)
        if isinstance(value, tuple):
            result[field.name + "_count"] = len(value)
        elif is_dataclass(value):
            if hasattr(value, "target_id"):
                result["target_id"] = value.target_id
        else:
            result[field.name] = _scalar(value)
    return result


def _page(items: list, offset: int, limit: int) -> dict:
    return {
        "items": items[offset : offset + limit],
        "total": len(items),
        "offset": offset,
        "next_offset": offset + limit if offset + limit < len(items) else None,
    }


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class AcquisitionService:
    """Each operation owns and closes its readers. No global open-file cache."""

    def __init__(
        self, roots: list[Path], output_dir: Path, max_frame_peaks: int = 5_000_000
    ):
        if not roots or max_frame_peaks < 1:
            raise ValueError("Provide data roots and a positive max_frame_peaks")
        self.roots = tuple(root.expanduser().resolve(strict=True) for root in roots)
        if not all(root.is_dir() for root in self.roots):
            raise ValueError("Every data root must be a directory")
        output = output_dir.expanduser().resolve()
        if any(p.suffix.lower() == ".d" for p in (output, *output.parents)):
            raise ValueError("The output directory must be outside acquisition folders")
        output.mkdir(parents=True, exist_ok=True)
        self.output_dir = output.resolve(strict=True)
        self.max_frame_peaks = max_frame_peaks

    def acquisition(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            if len(self.roots) != 1:
                raise ValueError(
                    "Use an absolute acquisition path when multiple roots are configured"
                )
            candidate = self.roots[0] / candidate
        candidate = candidate.resolve(strict=True)
        if not any(candidate.is_relative_to(root) for root in self.roots):
            raise ValueError("Acquisition is outside the configured data roots")
        if not candidate.is_dir():
            raise ValueError("Expected an acquisition directory")
        for name in ("analysis.tdf", "analysis.tdf_bin"):
            source = (candidate / name).resolve(strict=True)
            if not source.is_file() or not source.is_relative_to(candidate):
                raise ValueError(
                    f"{name} must be a file within the acquisition directory"
                )
        return candidate

    @contextmanager
    def connection(self, path: Path) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            (path / "analysis.tdf").as_uri() + "?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            yield connection
        finally:
            connection.close()

    def info(self) -> dict:
        return {
            "package_version": __version__,
            "transport": "stdio",
            "data_roots": [str(root) for root in self.roots],
            "output_dir": str(self.output_dir),
            "max_frame_peaks": self.max_frame_peaks,
            "limits": {
                "page_rows": 200,
                "preview_peaks": 100,
                "batch_windows": 32,
                "frame_checks_per_call": 128,
                "artifact_bytes": MAX_ARTIFACT_BYTES,
            },
            "units": {
                "retention_time": "seconds",
                "mobility": "1/K0 in V s/cm^2",
                "intensity": "raw intensities normalized to a 100 ms accumulation window",
            },
            "intervals": "Selection intervals and scan bounds are half-open",
            "sources": "All acquisitions are opened read-only. Tools write only new output artifacts.",
        }

    def discover(self, depth: int, offset: int, limit: int) -> dict:
        found = []
        pending = [(root, 0) for root in reversed(self.roots)]
        visited = 0
        seen = set()
        while pending:
            path, level = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            visited += 1
            if visited > 20_000:
                raise ValueError(
                    "Discovery exceeded 20,000 directories. Configure a narrower data root"
                )
            if (path / "analysis.tdf").is_file():
                found.append(
                    {
                        "path": str(path),
                        "has_binary": (path / "analysis.tdf_bin").is_file(),
                    }
                )
                continue
            if level < depth:
                pending.extend(
                    (p, level + 1)
                    for p in sorted(path.iterdir(), reverse=True)
                    if p.is_dir() and not p.is_symlink() and p != self.output_dir
                )
        return _page(found, offset, limit)

    def tables(self, acquisition: str) -> dict:
        path = self.acquisition(acquisition)
        with self.connection(path) as conn:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            return {
                "tables": [
                    {
                        "name": name,
                        "columns": [
                            dict(row)
                            for row in conn.execute(
                                f"PRAGMA table_info({_quote(name)})"
                            )
                        ],
                    }
                    for name in names
                ]
            }

    def read_table(
        self,
        acquisition: str,
        table: str,
        columns: list[str] | None,
        predicates: list[Predicate],
        offset: int,
        limit: int,
    ) -> dict:
        path = self.acquisition(acquisition)
        schema = {t["name"]: t["columns"] for t in self.tables(str(path))["tables"]}
        if table not in schema:
            raise ValueError(f"Unknown table {table!r}. Call list_metadata_tables")
        available = [c["name"] for c in schema[table]]
        columns = available if columns is None else columns
        if not columns or len(columns) > 50 or any(c not in available for c in columns):
            raise ValueError("Select 1 to 50 existing columns")
        operators = {
            "eq": "=",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }
        clauses, args = [], []
        for predicate in predicates:
            if predicate.column not in available:
                raise ValueError(f"Unknown column {predicate.column!r}")
            if predicate.operator == "is_null":
                clauses.append(f"{_quote(predicate.column)} IS NULL")
            else:
                if predicate.value is None:
                    raise ValueError("Use is_null to query NULL values")
                clauses.append(
                    f"{_quote(predicate.column)} {operators[predicate.operator]} ?"
                )
                args.append(predicate.value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        keys = [
            c["name"] for c in sorted(schema[table], key=lambda c: c["pk"]) if c["pk"]
        ]
        order = ", ".join(_quote(k) for k in keys) if keys else "rowid"
        with self.connection(path) as conn:
            query = f"SELECT {', '.join(_quote(c) for c in columns)} FROM {_quote(table)}{where} ORDER BY {order} LIMIT ? OFFSET ?"
            rows = conn.execute(query, [*args, limit + 1, offset]).fetchall()
        return {
            "table": table,
            "columns": columns,
            "offset": offset,
            "rows": [{k: _scalar(row[k]) for k in row.keys()} for row in rows[:limit]],
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    def inspect(self, acquisition: str) -> dict:
        path = self.acquisition(acquisition)
        with self.connection(path) as conn:
            types = [
                dict(r)
                for r in conn.execute(
                    "SELECT MsMsType AS msms_type, COUNT(*) AS frames, SUM(NumPeaks) AS stored_peaks FROM Frames GROUP BY MsMsType ORDER BY MsMsType"
                )
            ]
            span = dict(
                conn.execute(
                    "SELECT MIN(Time) AS rt_begin, MAX(Time) AS rt_end, COUNT(*) AS frames FROM Frames"
                ).fetchone()
            )
        return {
            "acquisition": str(path),
            "acquisition_type": get_acquisition_type(path),
            "summary": span,
            "frame_types": types,
            "files": self.identity(path),
            "next": "Use list_metadata_tables for schema, then query tools for acquisition objects. Use check_acquisition for supported-format checks.",
        }

    def identity(self, path: Path) -> list[dict]:
        return [
            {
                "name": name,
                "size_bytes": (path / name).stat().st_size,
                "mtime_ns": (path / name).stat().st_mtime_ns,
            }
            for name in ("analysis.tdf", "analysis.tdf_bin")
        ]

    def entities(
        self,
        acquisition: str,
        kind: str,
        rt: Interval | None,
        mz: Interval | None,
        group_or_target: int | None,
        offset: int,
        limit: int,
    ) -> dict:
        path = self.acquisition(acquisition)
        mode = get_acquisition_type(path)
        required = {
            "precursor": "DDA",
            "dia_window": "DIA",
            "prm_target": "PRM",
            "prm_transition": "PRM",
        }[kind]
        if mode != required:
            raise ValueError(
                f"{kind} queries require {required}, but this acquisition is {mode}"
            )
        attribute = {
            "precursor": "precursors",
            "dia_window": "windows",
            "prm_target": "targets",
            "prm_transition": "transitions",
        }[kind]
        matches = []
        with READERS[required](path) as reader:
            for index, obj in enumerate(getattr(reader, attribute)):
                time = obj.time if kind == "prm_target" else obj.rt
                mass = (
                    (
                        obj.monoisotopic_mz
                        if obj.monoisotopic_mz is not None
                        else obj.largest_peak_mz
                    )
                    if kind == "precursor"
                    else (
                        obj.monoisotopic_mz
                        if kind == "prm_target"
                        else obj.isolation_mz
                    )
                )
                if rt and not rt.contains(time):
                    continue
                if mz and not mz.contains(mass):
                    continue
                if group_or_target is not None:
                    actual = (
                        obj.window_group
                        if kind == "dia_window"
                        else (
                            obj.target_id
                            if kind == "prm_target"
                            else obj.target.target_id
                        )
                    )
                    if actual != group_or_target:
                        continue
                item = _metadata(obj)
                if kind != "prm_target":
                    item["selection"] = {
                        "kind": kind,
                        "id": obj.precursor_id if kind == "precursor" else index,
                    }
                matches.append(item)
        return _page(matches, offset, limit)

    def frame_budget(self, td: TimsData, ids: list[int]) -> None:
        if len(set(ids)) > 128:
            raise ValueError("A spectrum selection may reference at most 128 frames")
        for fid in set(ids):
            if td.frame_metadata(fid).num_peaks > self.max_frame_peaks:
                raise ValueError(
                    f"Frame {fid} exceeds max-frame-peaks. Raise the server startup limit deliberately for this dataset"
                )

    def spectrum(
        self, acquisition: str, selection: SpectrumSelection, processing: Processing
    ) -> tuple[np.ndarray, dict]:
        path = self.acquisition(acquisition)
        options = _options(processing)
        if selection.kind == "frame":
            with TimsData(path) as td:
                self.frame_budget(td, [selection.id])
                scan_range = None
                if selection.scan_begin is not None and selection.scan_end is not None:
                    if selection.scan_end > td.frame_metadata(selection.id).num_scans:
                        raise ValueError("scan_end exceeds the frame's scan count")
                    scan_range = (selection.scan_begin, selection.scan_end)
                extract = (
                    get_raw_peaks
                    if processing.mode == "raw"
                    else get_centroided_spectrum
                )
                peaks = extract(td, selection.id, scan_range=scan_range, **options)
                frames = [asdict(td.frame_metadata(selection.id))]
        else:
            required = {
                "precursor": "DDA",
                "dia_window": "DIA",
                "prm_transition": "PRM",
            }[selection.kind]
            if get_acquisition_type(path) != required:
                raise ValueError(f"This selection requires a {required} acquisition")
            with READERS[required](path) as reader:
                if selection.kind == "precursor":
                    if processing != Processing():
                        raise ValueError(
                            "Precursor spectra use the existing mobility-collapsed picker. Omit processing overrides, or select individual PASEF frames for custom processing"
                        )
                    assert isinstance(reader, DDA)
                    obj = reader.precursors[selection.id]
                    ids = list(
                        dict.fromkeys(
                            info.frame_id for info in obj.pasef_frame_msms_infos
                        )
                    )
                    self.frame_budget(reader.timsdata, ids)
                    peaks = obj.peaks
                else:
                    windows = list(
                        getattr(
                            reader, "windows" if required == "DIA" else "transitions"
                        )
                    )
                    if selection.id >= len(windows):
                        raise ValueError(
                            f"Selection index {selection.id} is out of range. Query the acquisition's windows first"
                        )
                    obj = windows[selection.id]
                    ids = [obj.frame_id]
                    self.frame_budget(reader.timsdata, ids)
                    peaks = (
                        obj.raw_peaks(**options)
                        if processing.mode == "raw"
                        else obj.centroid(**options)
                    )
                frames = [asdict(reader.timsdata.frame_metadata(fid)) for fid in ids]
        columns = (
            ["mz", "intensity"]
            if selection.kind == "precursor"
            else ["mz", "intensity", processing.ion_mobility_type]
        )
        keep = np.ones(len(peaks), dtype=bool)
        for column, interval in (
            (0, selection.mz_range),
            (2, selection.mobility_range),
        ):
            if interval is not None:
                keep &= (peaks[:, column] >= interval.lower) & (
                    peaks[:, column] < interval.upper
                )
        peaks = peaks[keep]
        metadata = {
            "tdfpy_version": __version__,
            "acquisition": str(path),
            "input_files": self.identity(path),
            "selection": selection.model_dump(),
            "processing": processing.model_dump(),
            "columns": columns,
            "frames": frames,
            "intensity_units": "normalized to a 100 ms accumulation window",
            "identity_note": "File size and modification time are recorded, not content hashes",
        }
        return peaks, metadata

    def preview(self, peaks: np.ndarray, metadata: dict, limit: int) -> dict:
        # Display the strongest peaks only. All statistics use the full result.
        indices = np.argsort(-peaks[:, 1], kind="stable")[:limit] if limit else []
        return {
            "metadata": metadata,
            "peak_count": len(peaks),
            "intensity_sum": float(peaks[:, 1].sum(dtype=np.float64)),
            "mz_range": [float(peaks[:, 0].min()), float(peaks[:, 0].max())]
            if len(peaks)
            else None,
            "preview": peaks[indices].tolist(),
            "preview_order": "descending intensity",
            "preview_truncated": len(peaks) > limit,
        }

    def write_artifact(self, arrays: Iterator[tuple[str, np.ndarray, dict]]) -> dict:
        if self.output_dir.resolve(strict=True) != self.output_dir:
            raise ValueError("Output directory changed since server startup")
        artifact_id = uuid4().hex
        path = self.output_dir / f"{artifact_id}.npz"
        entries = []
        size = 0
        created = False
        try:
            output = path.open("xb")
            created = True
            with (
                output,
                zipfile.ZipFile(
                    output, "w", compression=zipfile.ZIP_DEFLATED
                ) as archive,
            ):
                for name, peaks, metadata in arrays:
                    size += peaks.nbytes
                    if size > MAX_ARTIFACT_BYTES:
                        raise ValueError(
                            "Export exceeds 512 MiB of numerical arrays. Request a smaller batch"
                        )
                    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, peaks, allow_pickle=False)
                    entries.append(
                        {
                            "array": name,
                            "shape": list(peaks.shape),
                            "metadata": metadata,
                        }
                    )
                text = json.dumps({"spectra": entries}, allow_nan=False)
                with archive.open("metadata.npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(
                        stream, np.asarray(text), allow_pickle=False
                    )
        except BaseException:
            if created:
                path.unlink(missing_ok=True)
            raise
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "artifact_id": artifact_id,
            "path": str(path),
            "uri": f"tdfpy://artifacts/{artifact_id}",
            "sha256": checksum,
            "size_bytes": path.stat().st_size,
            "spectra": len(entries),
            "load": "numpy.load(path, allow_pickle=False). Read metadata with json.loads(str(data['metadata']))",
        }

    def export_batch(
        self, acquisition: str, indices: list[int], processing: Processing
    ) -> dict:
        if processing.mode != "centroid":
            raise ValueError("Window batches currently support centroided output only")
        path = self.acquisition(acquisition)
        mode = get_acquisition_type(path)
        if mode not in {"DIA", "PRM"}:
            raise ValueError("Window batches require DIA or PRM")
        options = _options(processing)
        with READERS[mode](path) as reader:
            windows = list(
                getattr(reader, "windows" if mode == "DIA" else "transitions")
            )
            if any(i < 0 or i >= len(windows) for i in indices):
                raise ValueError(
                    "A window index is out of range. Query the windows first"
                )
            selected = [windows[i] for i in indices]
            self.frame_budget(reader.timsdata, [w.frame_id for w in selected])

            def arrays() -> Iterator[tuple[str, np.ndarray, dict]]:
                for number, (window, peaks) in enumerate(
                    iter_window_spectra(selected, **options)
                ):
                    yield (
                        f"spectrum_{number:05d}",
                        peaks,
                        {
                            "acquisition": str(path),
                            "input_files": self.identity(path),
                            "tdfpy_version": __version__,
                            "selection": {
                                "kind": "dia_window"
                                if mode == "DIA"
                                else "prm_transition",
                                "id": indices[number],
                            },
                            "window": _metadata(window),
                            "processing": processing.model_dump(),
                            "columns": [
                                "mz",
                                "intensity",
                                processing.ion_mobility_type,
                            ],
                            "intensity_units": "normalized to a 100 ms accumulation window",
                        },
                    )

            return self.write_artifact(arrays())

    def artifact(
        self, artifact_id: str, array: str | None, offset: int, limit: int
    ) -> dict:
        if len(artifact_id) != 32 or any(
            c not in "0123456789abcdef" for c in artifact_id
        ):
            raise ValueError("Use an artifact_id returned by an export tool")
        path = self.output_dir / f"{artifact_id}.npz"
        if path.is_symlink() or path.resolve(strict=True).parent != self.output_dir:
            raise ValueError("Artifact must be a regular file in the output directory")
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["metadata"]))
            if array is None:
                return {"artifact_id": artifact_id, **manifest}
            names = {s["array"] for s in manifest["spectra"]}
            if array not in names:
                raise ValueError(f"Unknown array. Choose from {sorted(names)}")
            peaks = data[array]
            return {
                "array": array,
                "shape": list(peaks.shape),
                "total": len(peaks),
                "offset": offset,
                "rows": peaks[offset : offset + limit].tolist(),
                "next_offset": offset + limit if offset + limit < len(peaks) else None,
            }

    def check(self, acquisition: str) -> dict:
        report = validate_acquisition(self.acquisition(acquisition))
        return {
            "valid": report.valid,
            **asdict(report),
            "issues": [asdict(issue) for issue in report.issues[:200]],
            "issue_count": len(report.issues),
            "issues_truncated": len(report.issues) > 200,
        }

    def check_frames(self, acquisition: str, offset: int, limit: int) -> dict:
        path = self.acquisition(acquisition)
        issues = []
        with TimsData(path) as td:
            ids = td.frame_ids[offset : offset + limit]
            for fid in ids:
                try:
                    self.frame_budget(td, [fid])
                    td.read_frame_arrays(fid)
                except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
                    issues.append(
                        {
                            "frame_id": fid,
                            "message": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
            return {
                "passed": not issues,
                "frames_checked": list(ids),
                "issues": issues,
                "total_frames": len(td.frame_ids),
                "next_offset": offset + limit
                if offset + limit < len(td.frame_ids)
                else None,
                "scope": "Binary decoder checks for this page only. This is not vendor validation",
            }

    def convert(
        self,
        acquisition: str,
        frame_id: int,
        conversion: str,
        values: list[float],
        mz: float | None,
        charge: int | None,
    ) -> dict:
        path = self.acquisition(acquisition)
        methods = {
            "tof_to_mz": "indexToMz",
            "mz_to_tof": "mzToIndex",
            "scan_to_ook0": "scanNumToOneOverK0",
            "ook0_to_scan": "oneOverK0ToScanNum",
            "scan_to_voltage": "scanNumToVoltage",
        }
        with TimsData(path) as td:
            td.frame_metadata(frame_id)
            if conversion in methods:
                result = getattr(td, methods[conversion])(frame_id, values)
            else:
                if mz is None or charge is None or mz <= 0 or charge <= 0:
                    raise ValueError(
                        "CCS conversions require a positive m/z and explicit positive charge magnitude"
                    )
                convert = (
                    one_over_k0_to_ccs
                    if conversion == "ook0_to_ccs"
                    else ccs_to_one_over_k0
                )
                result = np.asarray([convert(v, charge, mz) for v in values])
            if not np.all(np.isfinite(result)):
                raise ValueError("Conversion produced non-finite values")
            return {
                "conversion": conversion,
                "values": result.tolist(),
                "frame_id": frame_id,
                "frame": asdict(td.frame_metadata(frame_id)),
                "mz": mz,
                "charge_magnitude": charge,
            }
