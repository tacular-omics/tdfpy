import logging
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Self

import pandas as pd

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
)
from .errors import ReaderClosedError, TdfpyError
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
    UNKNOWN = "Unknown"


def _frame_fields(row: pd.Series, timsdata: TimsData) -> dict:
    """Keyword arguments shared by every MS1 ``Frame`` subclass, from one ``Frames`` row."""
    return {
        "_timsdata": timsdata,
        "frame_id": int(row["Id"]),
        "rt": float(row["Time"]),
        "polarity": Polarity.from_str(str(row["Polarity"])),
        "scan_mode": int(row["ScanMode"]),
        "msms_type": MsMsType(int(row["MsMsType"])),
        "tims_id": int(row["TimsId"]) if not pd.isna(row["TimsId"]) else None,
        "max_intensity": int(row["MaxIntensity"]),
        "summed_intensities": int(row["SummedIntensities"]),
        "num_scans": int(row["NumScans"]),
        "num_peaks": int(row["NumPeaks"]),
        "mz_calibration_id": int(row["MzCalibration"]),
        "t1": float(row["T1"]),
        "t2": float(row["T2"]),
        "tims_calibration_id": int(row["TimsCalibration"]),
        "property_group_id": int(row["PropertyGroup"]) if not pd.isna(row["PropertyGroup"]) else None,
        "accumulation_time": float(row["AccumulationTime"]),
        "ramp_time": float(row["RampTime"]),
    }


def _frame_rt_and_polarity(frames_df: pd.DataFrame) -> tuple[dict[int, float], dict[int, Polarity]]:
    ids = frames_df["Id"].astype(int).tolist()
    rts = dict(zip(ids, frames_df["Time"].astype(float).tolist(), strict=True))
    polarities = dict(zip(ids, (Polarity.from_str(str(p)) for p in frames_df["Polarity"]), strict=True))
    return rts, polarities


def _unrecognised_msms_type(msms_type: int, frame_id: int) -> TdfpyError:
    return TdfpyError(
        f"Unrecognised MsMsType {msms_type} for frame {frame_id}. "
        "Expected one of "
        f"{[(t.name, t.value) for t in MsMsType]}. This frame's "
        "acquisition type may not match the reader class you are using "
        "(DDA/DIA/PRM)."
    )


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

    pandas_tdf = PandasTdf(str(analysis_tdf_path))
    frames_df = pandas_tdf.frames

    # Get unique MsMsType values
    msms_types = set(frames_df["MsMsType"].unique())

    # Check for DDA (MS2 type 8)
    if MsMsType.DDA_MS2.value in msms_types:
        return AcquisitionType.DDA

    # Check for DIA (MS2 type 9)
    if MsMsType.DIA_MS2.value in msms_types:
        return AcquisitionType.DIA

    # Check for PRM (MS2 type 10)
    if MsMsType.PRM_MS2.value in msms_types:
        return AcquisitionType.PRM

    logger.warning(
        "get_acquisition_type(%s): no known MS2 MsMsType found (present: %s); returning 'Unknown'. Expected one of DDA_MS2=8, DIA_MS2=9, PRM_MS2=10.",
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

    @property
    def timsdata(self) -> TimsData:
        if self._timsdata is None:
            self._timsdata = TimsData(str(self.analysis_path))
        return self._timsdata

    @property
    def pandas_tdf(self) -> PandasTdf:
        return PandasTdf(str(self.analysis_tdf_path))

    @property
    def metadata(self) -> MetaData:
        """Global metadata about the acquisition."""
        if self._metadata is None:
            self._metadata = MetaData(MappingProxyType(self.pandas_tdf.global_metadata.set_index("Key")["Value"].to_dict()))
        return self._metadata

    @property
    def calibration(self) -> Calibration:
        """Calibration information."""
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
        if self._closed:
            raise ReaderClosedError(f"{type(self).__name__} reader has been closed. Open a new one, and read inside its `with` block.")
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

        tdf = PandasTdf(str(self.analysis_tdf_path))
        frames_df = tdf.frames
        frame_id_to_rt, frame_id_to_polarity = _frame_rt_and_polarity(frames_df)
        timsdata = self.timsdata

        pasef_msms_infos: dict[int, list[PasefFrameMsmsInfo]] = {}
        # PasefFrameMsMsInfo rows whose Precursor is NULL cannot be attached to
        # any Precursor, so they are counted and logged rather than raised on.
        unassigned: list[PasefFrameMsmsInfo] = []
        for _, row in tdf.pasef_frame_msms_info.iterrows():
            frame_id = int(row["Frame"])
            pasef_info = PasefFrameMsmsInfo(
                _timsdata=timsdata,
                frame_id=frame_id,
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
                precursor_id=int(row["Precursor"]) if not pd.isna(row["Precursor"]) else None,
                rt=frame_id_to_rt[frame_id],
                polarity=frame_id_to_polarity[frame_id],
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
        for _, row in tdf.precursors.iterrows():
            precursor_id = int(row["Id"])
            frame_id = int(row["Parent"])
            precursor = Precursor(
                _timsdata=timsdata,
                precursor_id=precursor_id,
                largest_peak_mz=float(row["LargestPeakMz"]),
                average_mz=float(row["AverageMz"]),
                monoisotopic_mz=float(row["MonoisotopicMz"]) if not pd.isna(row["MonoisotopicMz"]) else None,
                charge=int(row["Charge"]) if not pd.isna(row["Charge"]) else None,
                scan_number=float(row["ScanNumber"]),
                intensity=float(row["Intensity"]),
                parent_frame_id=frame_id,
                pasef_frame_msms_infos=tuple(pasef_msms_infos.get(precursor_id, [])),
                rt=frame_id_to_rt[frame_id],
            )
            precursors[precursor_id] = precursor
            frame_to_precursors.setdefault(frame_id, []).append(precursor)

        ms1_frames: dict[int, DDAMs1Frame] = {}
        for _, row in frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MsMsType.MS1:
                ms1_frames[frame_id] = DDAMs1Frame(
                    **_frame_fields(row, timsdata),
                    precursors=tuple(frame_to_precursors.get(frame_id, [])),
                )
            elif msms_type != MsMsType.DDA_MS2:
                raise _unrecognised_msms_type(msms_type, frame_id)

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

        tdf = PandasTdf(str(self.analysis_tdf_path))
        frames_df = tdf.frames
        frame_id_to_rt, frame_id_to_polarity = _frame_rt_and_polarity(frames_df)
        timsdata = self.timsdata

        window_groups: dict[int, list[DiaWindowGroup]] = {}
        for key, row in tdf.dia_frame_msms_windows.iterrows():
            group = DiaWindowGroup(
                window_index=int(key),  # type: ignore
                window_group_id=int(row["WindowGroup"]),
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
            )
            window_groups.setdefault(group.window_group_id, []).append(group)

        # One DiaWindow per (frame, window-group row): each MS/MS frame repeats
        # every window of its group.
        frame_to_windows: dict[int, list[DiaWindow]] = {}
        all_windows: list[DiaWindow] = []
        for _, row in tdf.dia_frame_msms_info.iterrows():
            frame_id = int(row["Frame"])
            for group in window_groups.get(int(row["WindowGroup"]), []):
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
                    rt=frame_id_to_rt[frame_id],
                    polarity=frame_id_to_polarity[frame_id],
                )
                frame_to_windows.setdefault(frame_id, []).append(window)
                all_windows.append(window)

        ms1_frames: dict[int, DIAMs1Frame] = {}
        for _, row in frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MsMsType.MS1:
                ms1_frames[frame_id] = DIAMs1Frame(
                    **_frame_fields(row, timsdata),
                    dia_windows=tuple(frame_to_windows.get(frame_id, [])),
                )
            elif msms_type != MsMsType.DIA_MS2:
                raise _unrecognised_msms_type(msms_type, frame_id)

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
        TdfpyError: If a frame has an MsMsType other than MS1 or PRM MS/MS.

    Example:
        ```python
        with PRM("/path/to/data.d") as prm:
            for target in prm.targets:
                print(target.monoisotopic_mz, target.charge)
            for transition in prm.transitions:
                print(transition.frame_id, transition.target.target_id)
        ```
    """

    def __init__(self, analysis_dir: str | Path):
        super().__init__(analysis_dir)

        tdf = PandasTdf(str(self.analysis_tdf_path))
        frames_df = tdf.frames
        frame_id_to_rt, frame_id_to_polarity = _frame_rt_and_polarity(frames_df)
        timsdata = self.timsdata

        targets: dict[int, PrmTarget] = {}
        for _, row in tdf.prm_targets.iterrows():
            target_id = int(row["Id"])
            targets[target_id] = PrmTarget(
                target_id=target_id,
                external_id=str(row["ExternalId"]) if not pd.isna(row["ExternalId"]) else None,
                rt=float(row["Time"]),
                ook0=float(row["OneOverK0"]),
                monoisotopic_mz=float(row["MonoisotopicMz"]),
                charge=int(row["Charge"]),
                description=str(row["Description"]) if not pd.isna(row["Description"]) else "",
            )

        all_transitions: list[PrmTransition] = []
        frame_to_transitions: dict[int, list[PrmTransition]] = {}
        target_to_transitions: dict[int, list[PrmTransition]] = {}
        for _, row in tdf.prm_frame_msms_info.iterrows():
            frame_id = int(row["Frame"])
            target_id = int(row["Target"])
            transition = PrmTransition(
                _timsdata=timsdata,
                frame_id=frame_id,
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
                target=targets[target_id],
                rt=frame_id_to_rt[frame_id],
                polarity=frame_id_to_polarity[frame_id],
            )
            all_transitions.append(transition)
            frame_to_transitions.setdefault(frame_id, []).append(transition)
            target_to_transitions.setdefault(target_id, []).append(transition)

        # Targets are frozen; the back-reference is set once, here, before any
        # target leaves the reader. `transitions` is excluded from eq/hash.
        for target_id, target in targets.items():
            object.__setattr__(target, "transitions", tuple(target_to_transitions.get(target_id, [])))

        ms1_frames: dict[int, PRMMs1Frame] = {}
        for _, row in frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MsMsType.MS1:
                ms1_frames[frame_id] = PRMMs1Frame(
                    **_frame_fields(row, timsdata),
                    prm_transitions=tuple(frame_to_transitions.get(frame_id, [])),
                )
            elif msms_type != MsMsType.PRM_MS2:
                raise _unrecognised_msms_type(msms_type, frame_id)

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
