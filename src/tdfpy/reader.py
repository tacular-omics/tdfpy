import logging
import sqlite3
import warnings
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from .elems import (
    Calibration,
    DDAMs1Frame,
    DIAMs1Frame,
    DiaWindow,
    DiaWindowGroup,
    MetaData,
    MsMsType,
    PasefFrameMsmsInfo,
    Polarity,
    Precursor,
    PRMMs1Frame,
    PrmTarget,
    PrmTransition,
    _parse_polarity,
)
from .errors import AcquisitionTypeError, ReaderClosedError, TdfpyError
from .lookup import (
    DiaWindowLookup,
    Ms1FrameLookup,
    PrecursorLookup,
    PrmTargetLookup,
    PrmTransitionLookup,
)
from .tdf import PandasTdf
from .timsdata import TimsData

__all__ = ["DDA", "DIA", "PRM", "AcquisitionType", "get_acquisition_type"]

logger = logging.getLogger(__name__)


class AcquisitionType(StrEnum):
    """Acquisition mode of a `.d` folder, returned by :func:`get_acquisition_type`.

    A ``StrEnum``, so ``AcquisitionType.DDA == "DDA"``.
    """

    DDA = "DDA"
    DIA = "DIA"
    PRM = "PRM"
    UNKNOWN = "unknown"


@contextmanager
def _tdf_connection(tdf_path: Path) -> Iterator[sqlite3.Connection]:
    """Read-only SQLite connection to ``analysis.tdf``; SQLite errors become :class:`TdfpyError`."""
    try:
        with closing(sqlite3.connect(tdf_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            yield conn
    except sqlite3.Error as exc:
        raise TdfpyError(f"Failed to read TDF database {tdf_path}: {exc}") from exc


def _rows(conn: sqlite3.Connection, table: str, columns: Iterable[str]) -> list[tuple[Any, ...]]:
    """Every row of ``table`` as plain tuples of ``columns``, in table order.

    Reading tuples straight from SQLite is ~20x faster than building a
    DataFrame and walking it with ``iterrows``, which dominated reader start-up.
    """
    return conn.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()


#: ``Frames`` columns read for every MS1 ``Frame``, in the order of ``_frame_fields``.
_FRAME_COLUMNS = (
    "Id",
    "Time",
    "Polarity",
    "ScanMode",
    "MsMsType",
    "TimsId",
    "MaxIntensity",
    "SummedIntensities",
    "NumScans",
    "NumPeaks",
    "MzCalibration",
    "T1",
    "T2",
    "TimsCalibration",
    "PropertyGroup",
    "AccumulationTime",
    "RampTime",
)


class _Frames:
    """The ``Frames`` table, read once: rows plus per-frame RT and polarity."""

    def __init__(self, conn: sqlite3.Connection, source: str):
        self.rows = _rows(conn, "Frames", _FRAME_COLUMNS)
        self.rt: dict[int, float] = {}
        self.polarity: dict[int, Polarity | None] = {}
        unknown: set[str] = set()
        for row in self.rows:
            frame_id = int(row[0])
            self.rt[frame_id] = float(row[1])
            polarity = _parse_polarity(row[2])
            if polarity is None:
                unknown.add(str(row[2]))
            self.polarity[frame_id] = polarity
        if unknown:
            warnings.warn(
                f"{source}: Frames.Polarity has values other than '+' / '-' ({sorted(unknown)}); those frames get polarity=None.",
                UserWarning,
                stacklevel=3,
            )

    @cached_property
    def msms_types(self) -> set[int]:
        return {int(row[4]) for row in self.rows}

    def ms1_fields(self, timsdata: TimsData, expected_ms2: MsMsType) -> Iterator[dict[str, Any]]:
        """Keyword arguments for every MS1 ``Frame``. Raises on an MsMsType that is neither MS1 nor ``expected_ms2``."""
        for row in self.rows:
            (
                frame_id,
                rt,
                _,
                scan_mode,
                msms_type,
                tims_id,
                max_int,
                summed,
                num_scans,
                num_peaks,
                mz_cal,
                t1,
                t2,
                tims_cal,
                prop_group,
                acc_time,
                ramp_time,
            ) = row
            frame_id, msms_type = int(frame_id), int(msms_type)
            if msms_type == MsMsType.MS1:
                yield {
                    "_timsdata": timsdata,
                    "frame_id": frame_id,
                    "rt": float(rt),
                    "polarity": self.polarity[frame_id],
                    "scan_mode": int(scan_mode),
                    "msms_type": MsMsType.MS1,
                    "tims_id": None if tims_id is None else int(tims_id),
                    "base_peak_intensity": int(max_int),
                    "total_ion_current": int(summed),
                    "num_scans": int(num_scans),
                    "num_peaks": int(num_peaks),
                    "mz_calibration_id": int(mz_cal),
                    "t1": float(t1),
                    "t2": float(t2),
                    "tims_calibration_id": int(tims_cal),
                    "property_group_id": None if prop_group is None else int(prop_group),
                    "accumulation_time": float(acc_time),
                    "ramp_time": float(ramp_time),
                }
            elif msms_type != expected_ms2:
                raise _unrecognised_msms_type(msms_type, frame_id)


def _unrecognised_msms_type(msms_type: int, frame_id: int) -> TdfpyError:
    return TdfpyError(
        f"Unrecognised MsMsType {msms_type} for frame {frame_id}. "
        "Expected one of "
        f"{[(t.name, t.value) for t in MsMsType]}. This frame's "
        "acquisition type may not match the reader class you are using "
        "(DDA/DIA/PRM)."
    )


def _acquisition_type_of(msms_types: set[int]) -> AcquisitionType:
    """Acquisition type from the set of ``MsMsType`` values in the ``Frames`` table."""
    for acq, msms in ((AcquisitionType.DDA, MsMsType.DDA_MS2), (AcquisitionType.DIA, MsMsType.DIA_MS2), (AcquisitionType.PRM, MsMsType.PRM_MS2)):
        if msms.value in msms_types:
            return acq
    return AcquisitionType.UNKNOWN


def _require_acquisition_type(msms_types: set[int], expected: AcquisitionType) -> None:
    """Raise :class:`AcquisitionTypeError` if the frames are not an ``expected`` acquisition."""
    found = _acquisition_type_of(msms_types)
    if found not in (expected, AcquisitionType.UNKNOWN):
        raise AcquisitionTypeError(f"not a {expected} acquisition (found {found}); use tdfpy.get_acquisition_type()")


def get_acquisition_type(analysis_dir: str | Path) -> AcquisitionType:
    """
    Determine the acquisition type (DDA, DIA, or PRM) of a .d folder by
    examining the MsMsType values in the Frames table.

    Args:
        analysis_dir: Path to the .d folder

    Returns:
        ``AcquisitionType.DDA``, ``.DIA`` or ``.PRM`` for the first MS/MS frame
        type found (checked in that order), or ``AcquisitionType.UNKNOWN``.

    Raises:
        FileNotFoundError: If analysis.tdf does not exist
    """
    analysis_tdf_path = Path(analysis_dir) / "analysis.tdf"
    if not analysis_tdf_path.exists():
        raise FileNotFoundError(f"analysis.tdf not found at {analysis_tdf_path}")

    with _tdf_connection(analysis_tdf_path) as conn:
        msms_types = {int(t) for (t,) in conn.execute("SELECT DISTINCT MsMsType FROM Frames")}
    acq = _acquisition_type_of(msms_types)
    if acq is not AcquisitionType.UNKNOWN:
        return acq

    logger.warning(
        "get_acquisition_type(%s): no known MS2 MsMsType found (present: %s); returning 'unknown'. Expected one of DDA_MS2=8, DIA_MS2=9, PRM_MS2=10.",
        analysis_dir,
        sorted(msms_types),
    )
    return AcquisitionType.UNKNOWN


class _DFolder:
    """Shared base of the DDA, DIA and PRM readers."""

    def __init__(self, analysis_dir: str | Path):
        self._analysis_dir = str(analysis_dir)
        self._closed = False

        # assert paths exist
        if not self.analysis_tdf_path.exists():
            raise FileNotFoundError(f"analysis.tdf not found at {self.analysis_tdf_path}")
        if not self.analysis_tdf_bin_path.exists():
            raise FileNotFoundError(f"analysis.tdf_bin not found at {self.analysis_tdf_bin_path}")
        if not self.analysis_path.exists():
            raise FileNotFoundError(f"Analysis directory not found at {self.analysis_path}")

        # Lazily load
        self._timsdata = None
        self._metadata = None
        self._calibration = None

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise ReaderClosedError(f"{type(self).__name__} reader has been closed. Open a new one, and read inside its `with` block.")

    @property
    def timsdata(self) -> TimsData:
        """The open :class:`TimsData` handle on ``analysis.tdf_bin``.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        self._raise_if_closed()
        if self._timsdata is None:
            self._timsdata = TimsData(str(self.analysis_path))
        return self._timsdata

    @property
    def pandas_tdf(self) -> PandasTdf:
        """A :class:`PandasTdf` view of ``analysis.tdf``.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        self._raise_if_closed()
        return PandasTdf(str(self.analysis_tdf_path))

    @property
    def metadata(self) -> MetaData:
        """Global metadata about the acquisition.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        self._raise_if_closed()
        if self._metadata is None:
            self._metadata = MetaData(MappingProxyType(self.pandas_tdf.global_metadata.set_index("Key")["Value"].to_dict()))
        return self._metadata

    @property
    def calibration(self) -> Calibration:
        """Calibration information.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        self._raise_if_closed()
        if self._calibration is None:
            self._calibration = Calibration(MappingProxyType(self.pandas_tdf.calibration_info.set_index("KeyName")["Value"].to_dict()))
        return self._calibration

    @property
    def analysis_tdf_path(self) -> Path:
        return Path(self._analysis_dir) / "analysis.tdf"

    @property
    def analysis_tdf_bin_path(self) -> Path:
        return Path(self._analysis_dir) / "analysis.tdf_bin"

    @property
    def analysis_path(self) -> Path:
        return Path(self._analysis_dir)

    def _check_open(self) -> None:
        """Raise ``ReaderClosedError`` if the reader or its ``TimsData`` is closed."""
        self._raise_if_closed()
        if self.timsdata.handle is None:
            raise ReaderClosedError(f"The TimsData of this {type(self).__name__} reader was closed.")

    def close(self) -> None:
        """Close the TimsData connection."""
        if not getattr(self, "_closed", True):
            td = getattr(self, "_timsdata", None)
            if td is not None:
                td.close()
            self._closed = True

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - closes connection."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


class DDA(_DFolder):
    """Open a DDA (Data-Dependent Acquisition) `.d` folder.

    Use as a context manager so the TimsData connection is closed when done.
    Exposes MS1 frames via `ms1` and precursors via `precursors`.

    Args:
        analysis_dir: Path to the `.d` folder containing `analysis.tdf` and `analysis.tdf_bin`.

    Raises:
        FileNotFoundError: If the `.d` folder or required files are missing.
        AcquisitionTypeError: If the folder holds a DIA or PRM acquisition.
        TdfpyError: If a frame has an MsMsType other than MS1 or DDA MS/MS.

    Note:
        `PasefFrameMsMsInfo` rows with a NULL `Precursor` are tolerated: the
        `PasefFrameMsmsInfo` element is built with `precursor_id=None` and
        logged, rather than failing the whole file. Such rows are not
        reachable from any `Precursor`.

    Example:
        ```python
        with DDA("/path/to/data.d") as dda:
            for frame in dda.ms1:
                print(frame.frame_id, frame.rt)
        ```
    """

    def __init__(self, analysis_dir: str | Path):
        super().__init__(analysis_dir)

        with _tdf_connection(self.analysis_tdf_path) as conn:
            frames = _Frames(conn, self._analysis_dir)
            _require_acquisition_type(frames.msms_types, AcquisitionType.DDA)
            pasef_rows = _rows(
                conn,
                "PasefFrameMsMsInfo",
                ("Frame", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth", "CollisionEnergy", "Precursor"),
            )
            precursor_rows = _rows(
                conn,
                "Precursors",
                ("Id", "LargestPeakMz", "AverageMz", "MonoisotopicMz", "Charge", "ScanNumber", "Intensity", "Parent"),
            )
        timsdata = self.timsdata

        pasef_msms_infos: dict[int, list[PasefFrameMsmsInfo]] = {}
        # PasefFrameMsMsInfo rows whose Precursor is NULL cannot be attached to
        # any Precursor, so they are counted and logged rather than raised on.
        unassigned: list[PasefFrameMsmsInfo] = []
        for frame_id, scan_begin, scan_end, isolation_mz, isolation_width, collision_energy, precursor_id in pasef_rows:
            frame_id = int(frame_id)
            pasef_info = PasefFrameMsmsInfo(
                _timsdata=timsdata,
                frame_id=frame_id,
                scan_num_begin=int(scan_begin),
                scan_num_end=int(scan_end),
                isolation_mz=float(isolation_mz),
                isolation_width=float(isolation_width),
                collision_energy=float(collision_energy),
                precursor_id=None if precursor_id is None else int(precursor_id),
                rt=frames.rt[frame_id],
                polarity=frames.polarity[frame_id],
            )
            if pasef_info.precursor_id is None:
                unassigned.append(pasef_info)
            else:
                pasef_msms_infos.setdefault(pasef_info.precursor_id, []).append(pasef_info)

        if unassigned:
            logger.warning(
                "%s: %d PasefFrameMsMsInfo row(s) have a NULL Precursor and are "
                "not reachable from any Precursor (e.g. frames %s). They are "
                "kept with precursor_id=None.",
                self._analysis_dir,
                len(unassigned),
                sorted({i.frame_id for i in unassigned})[:10],
            )

        precursors: dict[int, Precursor] = {}
        frame_to_precursors: dict[int, list[Precursor]] = {}
        for precursor_id, largest_peak_mz, average_mz, monoisotopic_mz, charge, scan_number, intensity, frame_id in precursor_rows:
            precursor_id, frame_id = int(precursor_id), int(frame_id)
            precursor = Precursor(
                _timsdata=timsdata,
                precursor_id=precursor_id,
                largest_peak_mz=float(largest_peak_mz),
                average_mz=float(average_mz),
                monoisotopic_mz=None if monoisotopic_mz is None else float(monoisotopic_mz),
                charge=None if charge is None else int(charge),
                scan_number=float(scan_number),
                intensity=float(intensity),
                parent_frame_id=frame_id,
                pasef_frame_msms_infos=tuple(pasef_msms_infos.get(precursor_id, [])),
                rt=frames.rt[frame_id],
            )
            precursors[precursor_id] = precursor
            frame_to_precursors.setdefault(frame_id, []).append(precursor)

        ms1_frames: dict[int, DDAMs1Frame] = {
            fields["frame_id"]: DDAMs1Frame(**fields, precursors=tuple(frame_to_precursors.get(fields["frame_id"], [])))
            for fields in frames.ms1_fields(timsdata, MsMsType.DDA_MS2)
        }

        self._precursor_lookup = PrecursorLookup(precursors)
        self._ms1_frames_lookup = Ms1FrameLookup(ms1_frames)

        logger.info(
            "Opened DDA .d folder %s: %d MS1 frames, %d precursors, %d PASEF windows.",
            self._analysis_dir,
            len(ms1_frames),
            len(precursors),
            sum(len(v) for v in pasef_msms_infos.values()),
        )

    @property
    def ms1(self) -> Ms1FrameLookup[DDAMs1Frame]:
        """Lookup for MS1 frames. Supports indexing by frame ID."""
        self._check_open()
        return self._ms1_frames_lookup

    @property
    def precursors(self) -> PrecursorLookup:
        """Lookup for all precursors. Supports indexing by precursor ID and `.query()`."""
        self._check_open()
        return self._precursor_lookup


class DIA(_DFolder):
    """Open a DIA (Data-Independent Acquisition) `.d` folder.

    Use as a context manager so the TimsData connection is closed when done.
    Exposes MS1 frames via `ms1`, per-frame windows via `windows`, and the
    window definitions via `window_groups`.

    Args:
        analysis_dir: Path to the `.d` folder containing `analysis.tdf` and `analysis.tdf_bin`.

    Raises:
        FileNotFoundError: If the `.d` folder or required files are missing.
        AcquisitionTypeError: If the folder holds a DDA or PRM acquisition.
        TdfpyError: If a frame has an MsMsType other than MS1 or DIA MS/MS.

    Example:
        ```python
        with DIA("/path/to/data.d") as dia:
            for group in dia.window_groups:
                print(group.window_group_id, group.isolation_mz)
            for window in dia.windows[1]:
                print(window.frame_id, window.rt)
        ```
    """

    def __init__(self, analysis_dir: str | Path):
        super().__init__(analysis_dir)

        with _tdf_connection(self.analysis_tdf_path) as conn:
            frames = _Frames(conn, self._analysis_dir)
            _require_acquisition_type(frames.msms_types, AcquisitionType.DIA)
            window_rows = _rows(
                conn,
                "DiaFrameMsMsWindows",
                ("WindowGroup", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth", "CollisionEnergy"),
            )
            frame_group_rows = _rows(conn, "DiaFrameMsMsInfo", ("Frame", "WindowGroup"))
        timsdata = self.timsdata

        window_groups: dict[int, list[DiaWindowGroup]] = {}
        for window_index, (group_id, scan_begin, scan_end, isolation_mz, isolation_width, collision_energy) in enumerate(window_rows):
            group = DiaWindowGroup(
                window_index=window_index,
                window_group_id=int(group_id),
                scan_num_begin=int(scan_begin),
                scan_num_end=int(scan_end),
                isolation_mz=float(isolation_mz),
                isolation_width=float(isolation_width),
                collision_energy=float(collision_energy),
            )
            window_groups.setdefault(group.window_group_id, []).append(group)

        # One DiaWindow per (frame, window-group row): each MS/MS frame repeats
        # every window of its group.
        all_windows: list[DiaWindow] = []
        for frame_id, group_id in frame_group_rows:
            frame_id = int(frame_id)
            for group in window_groups.get(int(group_id), []):
                window = DiaWindow(
                    _timsdata=timsdata,
                    frame_id=frame_id,
                    window_index=group.window_index,
                    window_group_id=group.window_group_id,
                    scan_num_begin=group.scan_num_begin,
                    scan_num_end=group.scan_num_end,
                    isolation_mz=group.isolation_mz,
                    isolation_width=group.isolation_width,
                    collision_energy=group.collision_energy,
                    rt=frames.rt[frame_id],
                    polarity=frames.polarity[frame_id],
                )
                all_windows.append(window)

        ms1_frames: dict[int, DIAMs1Frame] = {fields["frame_id"]: DIAMs1Frame(**fields) for fields in frames.ms1_fields(timsdata, MsMsType.DIA_MS2)}

        self._window_groups = tuple(group for groups in window_groups.values() for group in groups)
        self._dia_windows_lookup = DiaWindowLookup(all_windows)
        self._ms1_frames_lookup = Ms1FrameLookup(ms1_frames)

        logger.info(
            "Opened DIA .d folder %s: %d MS1 frames, %d windows, %d window groups.",
            self._analysis_dir,
            len(ms1_frames),
            len(all_windows),
            len(window_groups),
        )

    @property
    def ms1(self) -> Ms1FrameLookup[DIAMs1Frame]:
        """Lookup for MS1 frames. Supports indexing by frame ID."""
        self._check_open()
        return self._ms1_frames_lookup

    @property
    def windows(self) -> DiaWindowLookup:
        """Lookup for all per-frame DIA windows. Indexed by window *group* ID; supports `.query()`."""
        self._check_open()
        return self._dia_windows_lookup

    @property
    def window_groups(self) -> tuple[DiaWindowGroup, ...]:
        """Every window definition (`DiaFrameMsMsWindows` row), grouped by window group ID."""
        self._check_open()
        return self._window_groups


class PRM(_DFolder):
    """Open a PRM (Parallel Reaction Monitoring) `.d` folder.

    Use as a context manager so the TimsData connection is closed when done.
    Exposes MS1 frames via `ms1`, PRM targets via `targets`, and individual
    transitions via `transitions`.

    Args:
        analysis_dir: Path to the `.d` folder containing `analysis.tdf` and `analysis.tdf_bin`.

    Raises:
        FileNotFoundError: If the `.d` folder or required files are missing.
        AcquisitionTypeError: If the folder holds a DDA or DIA acquisition.
        TdfpyError: If a frame has an MsMsType other than MS1 or PRM MS/MS.

    Example:
        ```python
        with PRM("/path/to/data.d") as prm:
            for target in prm.targets:
                print(target.precursor_mz, target.charge)
            for transition in prm.transitions:
                print(transition.frame_id, transition.target.target_id)
        ```
    """

    def __init__(self, analysis_dir: str | Path):
        super().__init__(analysis_dir)

        with _tdf_connection(self.analysis_tdf_path) as conn:
            frames = _Frames(conn, self._analysis_dir)
            _require_acquisition_type(frames.msms_types, AcquisitionType.PRM)
            target_rows = _rows(conn, "PrmTargets", ("Id", "ExternalId", "Time", "OneOverK0", "MonoisotopicMz", "Charge", "Description"))
            transition_rows = _rows(
                conn,
                "PrmFrameMsMsInfo",
                ("Frame", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth", "CollisionEnergy", "Target"),
            )
        timsdata = self.timsdata

        targets: dict[int, PrmTarget] = {}
        for target_id, external_id, rt, ook0, mz, charge, description in target_rows:
            target_id = int(target_id)
            targets[target_id] = PrmTarget(
                target_id=target_id,
                external_id=None if external_id is None else str(external_id),
                rt=float(rt),
                ook0=float(ook0),
                precursor_mz=float(mz),
                charge=int(charge),
                description="" if description is None else str(description),
            )

        all_transitions: list[PrmTransition] = []
        target_to_transitions: dict[int, list[PrmTransition]] = {}
        for frame_id, scan_begin, scan_end, isolation_mz, isolation_width, collision_energy, target_id in transition_rows:
            frame_id, target_id = int(frame_id), int(target_id)
            transition = PrmTransition(
                _timsdata=timsdata,
                frame_id=frame_id,
                scan_num_begin=int(scan_begin),
                scan_num_end=int(scan_end),
                isolation_mz=float(isolation_mz),
                isolation_width=float(isolation_width),
                collision_energy=float(collision_energy),
                target=targets[target_id],
                rt=frames.rt[frame_id],
                polarity=frames.polarity[frame_id],
            )
            all_transitions.append(transition)
            target_to_transitions.setdefault(target_id, []).append(transition)

        # Targets are frozen; the back-reference is set once, here, before any
        # target leaves the reader. `transitions` is excluded from eq/hash.
        for target_id, target in targets.items():
            object.__setattr__(target, "transitions", tuple(target_to_transitions.get(target_id, [])))

        ms1_frames: dict[int, PRMMs1Frame] = {fields["frame_id"]: PRMMs1Frame(**fields) for fields in frames.ms1_fields(timsdata, MsMsType.PRM_MS2)}

        self._prm_target_lookup = PrmTargetLookup(targets)
        self._prm_transition_lookup = PrmTransitionLookup(all_transitions)
        self._ms1_frames_lookup = Ms1FrameLookup(ms1_frames)

        logger.info(
            "Opened PRM .d folder %s: %d MS1 frames, %d targets, %d transitions.",
            self._analysis_dir,
            len(ms1_frames),
            len(targets),
            len(all_transitions),
        )

    @property
    def ms1(self) -> Ms1FrameLookup[PRMMs1Frame]:
        """Lookup for MS1 frames. Supports indexing by frame ID."""
        self._check_open()
        return self._ms1_frames_lookup

    @property
    def targets(self) -> PrmTargetLookup:
        """Lookup for all PRM targets. Supports indexing by target ID and `.query()`."""
        self._check_open()
        return self._prm_target_lookup

    @property
    def transitions(self) -> PrmTransitionLookup:
        """Lookup for all PRM transitions. Supports indexing by target ID and `.query()`."""
        self._check_open()
        return self._prm_transition_lookup
