from pathlib import Path
from typing import Self, Literal
from collections.abc import Generator

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
    PasefFrameMsmsInfo,
    Precursor, Polarity,
)
from .lookup import DiaWindowLookup, Ms1FrameLookup, PrecursorLookup
from .tdf import PandasTdf
from .timsdata import TimsData


def get_acquisition_type(analysis_dir: str) -> Literal["DDA", "DIA", "PRM", "Unknown"]:
    """
    Determine the acquisition type (DDA or DIA) of a .d folder by examining
    the MsMsType values in the Frames table.
    
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
    
    # PRM typically shows as type 2, but we can also check for the presence of certain tables
    # For now, return Unknown for other types
    return "Unknown"


# abstract base class for DFolder and DDA_Dfolder
class _DFolder:
    def __init__(self, analysis_dir: str):
        self._analysis_dir = analysis_dir
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
            self._metadata = MetaData(df=self.pandas_tdf.calibration_info)
        return self._metadata

    @property
    def calibration(self) -> Calibration:
        """Calibration information."""
        if self._calibration is None:
            self._calibration = Calibration(df=self.pandas_tdf.calibration_info)
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
        if not self._closed:
            self.timsdata.close()
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
    def __init__(self, analysis_dir: str):
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
                raise ValueError(
                    f"PASEF MS/MS info with null precursor found for frame {pasef_info.frame_id}. All PASEF MS/MS info must have a valid precursor ID."
                )

            if pasef_info.precursor not in self._pasef_msms_infos:
                self._pasef_msms_infos[pasef_info.precursor] = []
            self._pasef_msms_infos[pasef_info.precursor].append(pasef_info)

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
                    tims_id=int(row["TimsId"]),
                    max_intensity=int(row["MaxIntensity"]),
                    summed_intensities=int(row["SummedIntensities"]),
                    num_scans=int(row["NumScans"]),
                    num_peaks=int(row["NumPeaks"]),
                    mz_calibration=int(row["MzCalibration"]),
                    t1=float(row["T1"]),
                    t2=float(row["T2"]),
                    tims_calibration=int(row["TimsCalibration"]),
                    property_group=int(row["PropertyGroup"]),
                    accumulation_time=float(row["AccumulationTime"]),
                    ramp_time=float(row["RampTime"]),
                    precursors=tuple(precursors_for_frame),
                )
                self._ms1_frames[frame_id] = frame
                self._frames[frame_id] = frame
            elif msms_type == MsMsType.DDA_MS2.value:
                pass
            else:
                raise ValueError(f"Unknown MsMsType {msms_type} for frame {frame_id}")

        self._ms1_frames_lookup = Ms1FrameLookup(self._ms1_frames)

        # remove uneeded dataframes to save memory
        del self._precursor_df
        del self._frames_df
        del self._pasef_frame_msms_info_df

    @property
    def ms1(self) -> Ms1FrameLookup[DDAMs1Frame]:
        self._check_open()
        return self._ms1_frames_lookup

    @property
    def precursors(self) -> PrecursorLookup:
        self._check_open()
        return self._precursor_lookup


class DIA(_DFolder):
    # doesnt have precursors frame

    def __init__(self, analysis_dir: str):
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
                    polarity=Polarity.from_str(frame_id_to_polarity[frame_id])
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
                    t1=int(row["T1"]),
                    t2=int(row["T2"]),
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
                raise ValueError(f"Unknown MsMsType {msms_type} for frame {frame_id}")

        self._ms1_frames_lookup = Ms1FrameLookup(self._ms1_frames)

        # remove uneeded dataframes to save memory
        del self._frames_df
        del self._dia_frame_msms_windows_df
        del self._dia_frame_msms_info

    @property
    def ms1(self) -> Ms1FrameLookup[DIAMs1Frame]:
        self._check_open()
        return self._ms1_frames_lookup

    @property
    def windows(self) -> DiaWindowLookup:
        self._check_open()
        return self._dia_windows_lookup

    @property
    def window_groups(self) -> Generator[DiaWindowGroup, None, None]:
        self._check_open()
        for window_group_list in self._dia_window_groups.values():
            yield from window_group_list


class PRM(_DFolder):
    # not implemented yet
    pass
