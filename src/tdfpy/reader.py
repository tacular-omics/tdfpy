import logging
from collections.abc import Generator
from pathlib import Path
from typing import Literal, Self

import pandas as pd

from .elems import (
    Calibration,
    DDAMs1Frame,
    DIAMs1Frame,
    DiaWindow,
    DiaWindowGroup,
    Frame,
    MetaData,
    MsMsType,
    PRMMs1Frame,
    PasefFrameMsmsInfo,
    Polarity,
    Precursor,
    PrmTarget,
    PrmTransition,
)
from .lookup import (
    DiaWindowLookup,
    Ms1FrameLookup,
    PrecursorLookup,
    PrmTargetLookup,
    PrmTransitionLookup,
)
from .tdf import PandasTdf
from .timsdata import TimsData

logger = logging.getLogger(__name__)


def get_acquisition_type(analysis_dir: str | Path) -> Literal["DDA", "DIA", "PRM", "Unknown"]:
    """
    Determine the acquisition type (DDA, DIA, or PRM) of a .d folder by
    examining the MsMsType values in the Frames table.

    Args:
        analysis_dir: Path to the .d folder

    Returns:
        "DDA" if DDA acquisition detected
        "DIA" if DIA acquisition detected
        "PRM" if PRM acquisition detected
        "Unknown" if type cannot be determined

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
        return "DDA"

    # Check for DIA (MS2 type 9)
    if MsMsType.DIA_MS2.value in msms_types:
        return "DIA"

    # Check for PRM (MS2 type 10)
    if MsMsType.PRM_MS2.value in msms_types:
        return "PRM"

    logger.warning(
        "get_acquisition_type(%s): no known MS2 MsMsType found (present: %s); "
        "returning 'Unknown'. Expected one of DDA_MS2=8, DIA_MS2=9, PRM_MS2=10.",
        analysis_dir,
        sorted(msms_types),
    )
    return "Unknown"


# abstract base class for DFolder and DDA_Dfolder
class _DFolder:
    def __init__(self, analysis_dir: str | Path):
        self._analysis_dir = str(analysis_dir)
        self._closed = False

        # assert paths exist
        if not self.analysis_tdf_path.exists():
            raise FileNotFoundError(
                f"analysis.tdf not found at {self.analysis_tdf_path}"
            )
        if not self.analysis_tdf_bin_path.exists():
            raise FileNotFoundError(
                f"analysis.tdf_bin not found at {self.analysis_tdf_bin_path}"
            )
        if not self.analysis_path.exists():
            raise FileNotFoundError(
                f"Analysis directory not found at {self.analysis_path}"
            )

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
            self._metadata = MetaData(
                df=self.pandas_tdf.global_metadata.set_index("Key")["Value"]
            )
        return self._metadata

    @property
    def calibration(self) -> Calibration:
        """Calibration information."""
        if self._calibration is None:
            self._calibration = Calibration(
                df=self.pandas_tdf.calibration_info.set_index("KeyName")["Value"]
            )
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
        """Ensure connection is still open."""
        if self._closed:
            raise RuntimeError(
                "DFolder has been closed. Create a new DFolder instance or use a context manager."
            )
        if self.timsdata.handle is None:
            raise RuntimeError("TimsData connection has been unexpectedly closed.")

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

    Use as a context manager to ensure the TimsData connection is closed when done.
    Exposes MS1 frames via `ms1` and precursors via `precursors`.

    Args:
        analysis_dir: Path to the `.d` folder containing `analysis.tdf` and `analysis.tdf_bin`.

    Raises:
        FileNotFoundError: If the `.d` folder or required files are missing.

    Note:
        `PasefFrameMsMsInfo` rows with a NULL `Precursor` are tolerated: the
        `PasefFrameMsmsInfo` element is built with `precursor=None` (matching
        the dataclass's `int | None` field) and logged, rather than failing the
        whole file. Such rows are not reachable from any `Precursor`.

    Example:
        ```python
        with DDA("/path/to/data.d") as dda:
            for frame in dda.ms1:
                print(frame.frame_id, frame.time)
        ```
    """

    def __init__(self, analysis_dir: str | Path):
        super().__init__(analysis_dir)

        self._precursor_df = PandasTdf(str(self.analysis_tdf_path)).precursors
        self._frames_df = PandasTdf(str(self.analysis_tdf_path)).frames
        self._pasef_frame_msms_info_df = PandasTdf(
            str(self.analysis_tdf_path)
        ).pasef_frame_msms_info

        frame_id_to_rt = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            time = float(row["Time"])
            frame_id_to_rt[frame_id] = time

        frame_to_polarity: dict[int, str] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            polarity = str(row["Polarity"])
            frame_to_polarity[frame_id] = polarity

        self._pasef_msms_infos: dict[int, list[PasefFrameMsmsInfo]] = {}
        # PasefFrameMsMsInfo rows whose Precursor is NULL cannot be attached to
        # any Precursor, so they are kept here rather than dropped or raised on.
        self._unassigned_pasef_msms_infos: list[PasefFrameMsmsInfo] = []
        for _, row in self._pasef_frame_msms_info_df.iterrows():
            frame_id = int(row["Frame"])
            polarity = frame_to_polarity[frame_id]
            pasef_info = PasefFrameMsmsInfo(
                _timsdata=self.timsdata,
                frame_id=frame_id,
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
                precursor=int(row["Precursor"])
                if not pd.isna(row["Precursor"])
                else None,
                rt=frame_id_to_rt[frame_id],
                polarity=Polarity.from_str(polarity),
            )

            if pasef_info.precursor is None:
                self._unassigned_pasef_msms_infos.append(pasef_info)
                continue

            if pasef_info.precursor not in self._pasef_msms_infos:
                self._pasef_msms_infos[pasef_info.precursor] = []
            self._pasef_msms_infos[pasef_info.precursor].append(pasef_info)

        if self._unassigned_pasef_msms_infos:
            logger.warning(
                "%s: %d PasefFrameMsMsInfo row(s) have a NULL Precursor and are "
                "not reachable from any Precursor (e.g. frames %s). They are "
                "kept with precursor=None.",
                self._analysis_dir,
                len(self._unassigned_pasef_msms_infos),
                sorted({i.frame_id for i in self._unassigned_pasef_msms_infos})[:10],
            )

        self._precursors: dict[int, Precursor] = {}
        self._frame_to_precursors: dict[int, list[Precursor]] = {}
        for _, row in self._precursor_df.iterrows():
            precursor_id = int(row["Id"])
            frame_id = int(row["Parent"])

            precursor = Precursor(
                _timsdata=self.timsdata,
                precursor_id=precursor_id,
                largest_peak_mz=float(row["LargestPeakMz"]),
                average_mz=float(row["AverageMz"]),
                monoisotopic_mz=float(row["MonoisotopicMz"])
                if not pd.isna(row["MonoisotopicMz"])
                else None,
                charge=int(row["Charge"]) if not pd.isna(row["Charge"]) else None,
                scan_number=int(row["ScanNumber"]),
                intensity=float(row["Intensity"]),
                parent_frame=int(row["Parent"]),
                pasef_frame_msms_infos=tuple(
                    self._pasef_msms_infos.get(precursor_id, [])
                ),
                rt=frame_id_to_rt[frame_id],
            )
            self._precursors[precursor_id] = precursor
            if frame_id not in self._frame_to_precursors:
                self._frame_to_precursors[frame_id] = []
            self._frame_to_precursors[frame_id].append(precursor)

        self._precursor_lookup = PrecursorLookup(self._precursors)

        self._frames: dict[int, Frame] = {}
        self._ms1_frames: dict[int, DDAMs1Frame] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MsMsType.MS1.value:
                precursors_for_frame: list[Precursor] = self._frame_to_precursors.get(
                    frame_id, []
                )
                frame = DDAMs1Frame(
                    _timsdata=self.timsdata,
                    frame_id=frame_id,
                    time=float(row["Time"]),
                    polarity=Polarity.from_str(str(row["Polarity"])),
                    scan_mode=int(row["ScanMode"]),
                    msms_type=msms_type,
                    tims_id=int(row["TimsId"]) if not pd.isna(row["TimsId"]) else None,
                    max_intensity=int(row["MaxIntensity"]),
                    summed_intensities=int(row["SummedIntensities"]),
                    num_scans=int(row["NumScans"]),
                    num_peaks=int(row["NumPeaks"]),
                    mz_calibration=int(row["MzCalibration"]),
                    t1=float(row["T1"]),
                    t2=float(row["T2"]),
                    tims_calibration=int(row["TimsCalibration"]),
                    property_group=int(row["PropertyGroup"])
                    if not pd.isna(row["PropertyGroup"])
                    else None,
                    accumulation_time=float(row["AccumulationTime"]),
                    ramp_time=float(row["RampTime"]),
                    precursors=tuple(precursors_for_frame),
                )
                self._ms1_frames[frame_id] = frame
                self._frames[frame_id] = frame
            elif msms_type == MsMsType.DDA_MS2.value:
                pass
            else:
                raise ValueError(
                    f"Unrecognised MsMsType {msms_type} for frame {frame_id}. "
                    "Expected one of "
                    f"{[(t.name, t.value) for t in MsMsType]}. This frame's "
                    "acquisition type may not match the reader class you are using "
                    "(DDA/DIA/PRM)."
                )

        self._ms1_frames_lookup = Ms1FrameLookup(self._ms1_frames)

        logger.info(
            "Opened DDA .d folder %s: %d MS1 frames, %d precursors, %d PASEF windows.",
            self._analysis_dir,
            len(self._ms1_frames),
            len(self._precursors),
            sum(len(v) for v in self._pasef_msms_infos.values()),
        )

        # remove uneeded dataframes to save memory
        del self._precursor_df
        del self._frames_df
        del self._pasef_frame_msms_info_df

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

    Use as a context manager to ensure the TimsData connection is closed when done.
    Exposes MS1 frames via `ms1`, individual windows via `windows`, and window groups
    via `window_groups`.

    Args:
        analysis_dir: Path to the `.d` folder containing `analysis.tdf` and `analysis.tdf_bin`.

    Raises:
        FileNotFoundError: If the `.d` folder or required files are missing.

    Example:
        ```python
        with DIA("/path/to/data.d") as dia:
            for group in dia.window_groups:
                for window in group.windows:
                    print(window.isolation_mz)
        ```
    """

    def __init__(self, analysis_dir: str | Path):
        super().__init__(analysis_dir)

        # frames
        self._frames_df = PandasTdf(str(self.analysis_tdf_path)).frames

        frame_id_to_rt = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            time = float(row["Time"])
            frame_id_to_rt[frame_id] = time

        frame_id_to_polarity = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            polarity = str(row["Polarity"])
            frame_id_to_polarity[frame_id] = polarity

        # window groups
        self._dia_frame_msms_windows_df = PandasTdf(
            str(self.analysis_tdf_path)
        ).dia_frame_msms_windows

        # frame to window groups
        self._dia_frame_msms_info = PandasTdf(
            str(self.analysis_tdf_path)
        ).dia_frame_msms_info

        self._dia_window_groups: dict[int, list[DiaWindowGroup]] = {}
        for key, row in self._dia_frame_msms_windows_df.iterrows():
            window_id = int(key)  # type: ignore
            window = DiaWindowGroup(
                window_index=window_id,
                window_group=int(row["WindowGroup"]),
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
            )
            if window.window_group not in self._dia_window_groups:
                self._dia_window_groups[window.window_group] = []
            self._dia_window_groups[window.window_group].append(window)

        # now we need to create DiaWindow objects which have an additional frame_id
        # create dia windows
        self._dia_windows: dict[int, list[DiaWindow]] = {}
        self._all_dia_windows: list[DiaWindow] = []
        for _, row in self._dia_frame_msms_info.iterrows():
            frame_id = int(row["Frame"])
            window_group_id = int(row["WindowGroup"])
            # each frame can have multiple window groups
            window_groups = self._dia_window_groups.get(window_group_id, [])
            if frame_id not in self._dia_windows:
                self._dia_windows[frame_id] = []
            for window_group in window_groups:
                dia_window = DiaWindow(
                    _timsdata=self.timsdata,
                    frame_id=frame_id,
                    window_index=window_group.window_index,
                    window_group=window_group.window_group,
                    scan_num_begin=window_group.scan_num_begin,
                    scan_num_end=window_group.scan_num_end,
                    isolation_mz=window_group.isolation_mz,
                    isolation_width=window_group.isolation_width,
                    collision_energy=window_group.collision_energy,
                    rt=frame_id_to_rt[frame_id],
                    polarity=Polarity.from_str(frame_id_to_polarity[frame_id]),
                )
                self._dia_windows[frame_id].append(dia_window)
                self._all_dia_windows.append(dia_window)

        self._dia_windows_lookup = DiaWindowLookup(self._all_dia_windows)

        self._frames: dict[int, Frame] = {}
        self._ms1_frames: dict[int, DIAMs1Frame] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MsMsType.MS1.value:
                frame = DIAMs1Frame(
                    _timsdata=self.timsdata,
                    frame_id=frame_id,
                    time=float(row["Time"]),
                    polarity=Polarity.from_str(str(row["Polarity"])),
                    scan_mode=int(row["ScanMode"]),
                    msms_type=msms_type,
                    tims_id=int(row["TimsId"]) if not pd.isna(row["TimsId"]) else None,
                    max_intensity=int(row["MaxIntensity"]),
                    summed_intensities=int(row["SummedIntensities"]),
                    num_scans=int(row["NumScans"]),
                    num_peaks=int(row["NumPeaks"]),
                    mz_calibration=int(row["MzCalibration"]),
                    t1=float(row["T1"]),
                    t2=float(row["T2"]),
                    tims_calibration=int(row["TimsCalibration"]),
                    property_group=int(row["PropertyGroup"])
                    if not pd.isna(row["PropertyGroup"])
                    else None,
                    accumulation_time=float(row["AccumulationTime"]),
                    ramp_time=float(row["RampTime"]),
                    dia_windows=tuple(self._dia_windows.get(frame_id, [])),
                )
                self._ms1_frames[frame_id] = frame
                self._frames[frame_id] = frame
            elif msms_type == MsMsType.DIA_MS2.value:
                pass
            else:
                raise ValueError(
                    f"Unrecognised MsMsType {msms_type} for frame {frame_id}. "
                    "Expected one of "
                    f"{[(t.name, t.value) for t in MsMsType]}. This frame's "
                    "acquisition type may not match the reader class you are using "
                    "(DDA/DIA/PRM)."
                )

        self._ms1_frames_lookup = Ms1FrameLookup(self._ms1_frames)

        logger.info(
            "Opened DIA .d folder %s: %d MS1 frames, %d windows, %d window groups.",
            self._analysis_dir,
            len(self._ms1_frames),
            len(self._all_dia_windows),
            len(self._dia_window_groups),
        )

        # remove uneeded dataframes to save memory
        del self._frames_df
        del self._dia_frame_msms_windows_df
        del self._dia_frame_msms_info

    @property
    def ms1(self) -> Ms1FrameLookup[DIAMs1Frame]:
        """Lookup for MS1 frames. Supports indexing by frame ID."""
        self._check_open()
        return self._ms1_frames_lookup

    @property
    def windows(self) -> DiaWindowLookup:
        """Lookup for all DIA windows. Supports indexing by window *group* ID and `.query()`."""
        self._check_open()
        return self._dia_windows_lookup

    @property
    def window_groups(self) -> Generator[DiaWindowGroup, None, None]:
        """Iterate over all DiaWindowGroup objects across all window groups."""
        self._check_open()
        for window_group_list in self._dia_window_groups.values():
            yield from window_group_list


class PRM(_DFolder):
    """Open a PRM (Parallel Reaction Monitoring) `.d` folder.

    Use as a context manager to ensure the TimsData connection is closed when done.
    Exposes MS1 frames via `ms1`, PRM targets via `targets`, and individual
    transitions via `transitions`.

    Args:
        analysis_dir: Path to the `.d` folder containing `analysis.tdf` and `analysis.tdf_bin`.

    Raises:
        FileNotFoundError: If the `.d` folder or required files are missing.

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

        # frames
        self._frames_df = PandasTdf(str(self.analysis_tdf_path)).frames

        frame_id_to_rt: dict[int, float] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            frame_id_to_rt[frame_id] = float(row["Time"])

        frame_id_to_polarity: dict[int, str] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            frame_id_to_polarity[frame_id] = str(row["Polarity"])

        # targets
        self._prm_targets_df = PandasTdf(str(self.analysis_tdf_path)).prm_targets

        self._prm_targets: dict[int, PrmTarget] = {}
        for _, row in self._prm_targets_df.iterrows():
            target_id = int(row["Id"])
            target = PrmTarget(
                target_id=target_id,
                external_id=str(row["ExternalId"]) if not pd.isna(row["ExternalId"]) else None,
                time=float(row["Time"]),
                one_over_k0=float(row["OneOverK0"]),
                monoisotopic_mz=float(row["MonoisotopicMz"]),
                charge=int(row["Charge"]),
                description=str(row["Description"]) if not pd.isna(row["Description"]) else "",
            )
            self._prm_targets[target_id] = target

        self._prm_target_lookup = PrmTargetLookup(self._prm_targets)

        # transitions (PrmFrameMsMsInfo)
        self._prm_frame_msms_info_df = PandasTdf(
            str(self.analysis_tdf_path)
        ).prm_frame_msms_info

        self._all_prm_transitions: list[PrmTransition] = []
        self._frame_to_transitions: dict[int, list[PrmTransition]] = {}
        target_to_transitions: dict[int, list[PrmTransition]] = {}
        for _, row in self._prm_frame_msms_info_df.iterrows():
            frame_id = int(row["Frame"])
            target_id = int(row["Target"])
            target = self._prm_targets[target_id]
            transition = PrmTransition(
                _timsdata=self.timsdata,
                frame_id=frame_id,
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
                target=target,
                rt=frame_id_to_rt[frame_id],
                polarity=Polarity.from_str(frame_id_to_polarity[frame_id]),
            )
            self._all_prm_transitions.append(transition)
            if frame_id not in self._frame_to_transitions:
                self._frame_to_transitions[frame_id] = []
            self._frame_to_transitions[frame_id].append(transition)
            if target_id not in target_to_transitions:
                target_to_transitions[target_id] = []
            target_to_transitions[target_id].append(transition)

        # populate target → transitions back-references
        for target_id, target in self._prm_targets.items():
            target.transitions = tuple(target_to_transitions.get(target_id, []))

        self._prm_transition_lookup = PrmTransitionLookup(self._all_prm_transitions)

        # MS1 frames
        self._frames: dict[int, Frame] = {}
        self._ms1_frames: dict[int, PRMMs1Frame] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MsMsType.MS1.value:
                frame = PRMMs1Frame(
                    _timsdata=self.timsdata,
                    frame_id=frame_id,
                    time=float(row["Time"]),
                    polarity=Polarity.from_str(str(row["Polarity"])),
                    scan_mode=int(row["ScanMode"]),
                    msms_type=msms_type,
                    tims_id=int(row["TimsId"]) if not pd.isna(row["TimsId"]) else None,
                    max_intensity=int(row["MaxIntensity"]),
                    summed_intensities=int(row["SummedIntensities"]),
                    num_scans=int(row["NumScans"]),
                    num_peaks=int(row["NumPeaks"]),
                    mz_calibration=int(row["MzCalibration"]),
                    t1=float(row["T1"]),
                    t2=float(row["T2"]),
                    tims_calibration=int(row["TimsCalibration"]),
                    property_group=int(row["PropertyGroup"])
                    if not pd.isna(row["PropertyGroup"])
                    else None,
                    accumulation_time=float(row["AccumulationTime"]),
                    ramp_time=float(row["RampTime"]),
                    prm_transitions=tuple(self._frame_to_transitions.get(frame_id, [])),
                )
                self._ms1_frames[frame_id] = frame
                self._frames[frame_id] = frame
            elif msms_type == MsMsType.PRM_MS2.value:
                pass
            else:
                raise ValueError(
                    f"Unrecognised MsMsType {msms_type} for frame {frame_id}. "
                    "Expected one of "
                    f"{[(t.name, t.value) for t in MsMsType]}. This frame's "
                    "acquisition type may not match the reader class you are using "
                    "(DDA/DIA/PRM)."
                )

        self._ms1_frames_lookup = Ms1FrameLookup(self._ms1_frames)

        logger.info(
            "Opened PRM .d folder %s: %d MS1 frames, %d targets, %d transitions.",
            self._analysis_dir,
            len(self._ms1_frames),
            len(self._prm_targets),
            len(self._all_prm_transitions),
        )

        # remove unneeded dataframes to save memory
        del self._frames_df
        del self._prm_targets_df
        del self._prm_frame_msms_info_df

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
