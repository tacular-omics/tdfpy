from functools import partial
from numpy.random import Generator
from .centroiding import get_centroided_spectrum
import datetime
import warnings
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass
from .timsdata import TimsData, oneOverK0ToCCSforMz
from .tdf import PandasTdf
import numpy as np
import pandas as pd
import numpy.typing as npt

MS1_MSMSTYPE = 0
DDA_MS2_MSMSTYPE = 8
DIA_MS2_MSMSTYPE = 9


@dataclass
class _TdfData:
    _timsdata: TimsData

    @property
    def timsdata(self) -> TimsData:
        if self._timsdata.handle is None:
            raise RuntimeError(
                "TimsData connection is closed. Keep DFolder instance alive or use context manager."
            )
        return self._timsdata


@dataclass
class PasefFrameMsmsInfo(_TdfData):
    """
    CREATE TABLE PasefFrameMsMsInfo (
        Frame INTEGER NOT NULL,
        ScanNumBegin INTEGER NOT NULL,
        ScanNumEnd INTEGER NOT NULL,
        IsolationMz REAL NOT NULL,
        IsolationWidth REAL NOT NULL,
        CollisionEnergy REAL NOT NULL,
        Precursor INTEGER,
        PRIMARY KEY(Frame, ScanNumBegin),
        FOREIGN KEY(Frame) REFERENCES Frames(Id),
        FOREIGN KEY(Precursor) REFERENCES Precursors(Id)
        ) WITHOUT ROWID
    """

    frame_id: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float
    precursor: int | None

    @property
    def unique_id(self) -> tuple[int, int | None]:
        if self.precursor is None:
            warnings.warn(
                f"Precursor is None for frame {self.frame_id}. Unique ID will be (frame_id, None).",
                UserWarning,
                stacklevel=2,
            )
        return (self.frame_id, self.precursor)

    @property
    def peaks(self) -> np.ndarray:
        scan = self.timsdata.extractCentroidedSpectrumForFrame(
            self.frame_id, self.scan_num_begin, self.scan_num_end
        )
        if scan is None:
            raise ValueError(
                f"Could not extract spectrum for frame {self.frame_id} with scan range {self.scan_num_begin}-{self.scan_num_end}."
            )
        mz_array = np.array(scan[0])
        intensity_array = np.array(scan[1])
        peaks = np.stack((mz_array, intensity_array), axis=-1)
        return peaks


@dataclass
class Precursor(_TdfData):
    """
    CREATE TABLE Precursors (Id INTEGER PRIMARY KEY,
        LargestPeakMz REAL NOT NULL,
        AverageMz REAL NOT NULL,
        MonoisotopicMz REAL,
        Charge INTEGER,
        ScanNumber REAL NOT NULL,
        Intensity REAL NOT NULL,
        Parent INTEGER,
        FOREIGN KEY(Parent) REFERENCES Frames(Id)
        )
    """

    precurosr_id: int
    largest_peak_mz: float
    average_mz: float
    monoisotopic_mz: float | None
    charge: int | None
    scan_number: int
    intensity: float
    parent_frame: int
    pasef_frame_msms_infos: tuple[PasefFrameMsmsInfo, ...]

    @property
    def ook0(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.parent_frame, [self.scan_number])[
            0
        ]

    @property
    def ccs(self) -> float:
        return oneOverK0ToCCSforMz(
            self.ook0, self.charge or 1, self.monoisotopic_mz or self.largest_peak_mz
        )

    @property
    def voltage(self) -> float:
        return self.timsdata.scanNumToVoltage(self.parent_frame, [self.scan_number])[0]

    @property
    def peaks(self) -> npt.NDArray[np.float64]:
        prec_map: dict[int, Any] = self.timsdata.readPasefMsMs([self.precurosr_id])
        if self.precurosr_id not in prec_map:
            raise ValueError(f"Precursor ID {self.precurosr_id} not found in TimsData.")
        scan = prec_map[self.precurosr_id]
        mz_array = np.array(scan[0])
        intensity_array = np.array(scan[1])
        peaks = np.stack((mz_array, intensity_array), axis=-1)
        return peaks

    @property
    def pasef_peaks(self) -> list[np.ndarray]:
        return [pasef_info.peaks for pasef_info in self.pasef_frame_msms_infos]


@dataclass
class _DiaWindowGroup:
    """
    CREATE TABLE DiaFrameMsMsWindows (
        WindowGroup INTEGER NOT NULL,
        ScanNumBegin INTEGER NOT NULL,
        ScanNumEnd INTEGER NOT NULL,
        IsolationMz REAL NOT NULL,
        IsolationWidth REAL NOT NULL,
        CollisionEnergy REAL NOT NULL,
        PRIMARY KEY(WindowGroup, ScanNumBegin),
        FOREIGN KEY (WindowGroup) REFERENCES DiaFrameMsMsWindowGroups (Id)
        ) WITHOUT ROWID
    """

    window_id: int
    window_group: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float


@dataclass
class Frame(_TdfData):
    """
    CREATE TABLE Frames (
        Id INTEGER PRIMARY KEY,
        Time REAL NOT NULL,
        Polarity CHAR(1) CHECK (Polarity IN ('+', '-')) NOT NULL,
        ScanMode INTEGER NOT NULL,
        MsMsType INTEGER NOT NULL,
        TimsId INTEGER,
        MaxIntensity INTEGER NOT NULL,
        SummedIntensities INTEGER NOT NULL,
        NumScans INTEGER NOT NULL,
        NumPeaks INTEGER NOT NULL,
        MzCalibration INTEGER NOT NULL,
        T1 REAL NOT NULL,
        T2 REAL NOT NULL,
        TimsCalibration INTEGER NOT NULL,
        PropertyGroup INTEGER,
        AccumulationTime REAL NOT NULL,
        RampTime REAL NOT NULL,
        FOREIGN KEY (MzCalibration) REFERENCES MzCalibration (Id),
        FOREIGN KEY (TimsCalibration) REFERENCES TimsCalibration (Id),
        FOREIGN KEY (PropertyGroup) REFERENCES PropertyGroups (Id)
    )
    """

    frame_id: int
    time: float
    polarity: str
    scan_mode: int
    msms_type: int
    tims_id: int | None
    max_intensity: int
    summed_intensities: int
    num_scans: int
    num_peaks: int
    mz_calibration: int
    t1: float
    t2: float
    tims_calibration: int
    property_group: int | None
    accumulation_time: float
    ramp_time: float

    @property
    def is_positive(self) -> bool:
        return self.polarity == "+"

    @property
    def is_negative(self) -> bool:
        return self.polarity == "-"

    @property
    def peaks(self) -> list[npt.NDArray[np.float64]]:
        """Read raw peaks for this frame and return as list of (mz, intensity) arrays."""
        d: list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = (
            self.timsdata.readScans(self.frame_id, 0, self.num_scans)
        )
        mz_int_arrays = []
        for index_array, int_array in d:
            mz_array = self.timsdata.indexToMz(self.frame_id, index_array)
            mz_int_arrays.append(
                np.stack((mz_array, int_array), axis=-1).astype(np.float64)
            )
        return mz_int_arrays

    def centroid(
        self,
        mz_tolerance: float = 8,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        im_tolerance: float = 0.05,
        im_tolerance_type: Literal["relative", "absolute"] = "relative",
        min_peaks: int = 3,
        max_peaks: int | None = None,
        noise_filter=None,
        ion_mobility_type: Literal["ccs", "ook0", "voltage"] = "ook0",
    ) -> np.ndarray:
        """Centroid the spectrum for this frame using the specified parameters."""
        get_spectrum = partial(
            get_centroided_spectrum,
            self.timsdata,
            frame_id=self.frame_id,
            spectrum_index=None,
            ion_mobility_type=ion_mobility_type,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            im_tolerance=im_tolerance,
            im_tolerance_type=im_tolerance_type,
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            noise_filter=noise_filter,
        )

        try:
            return get_spectrum(use_rust=True)
        except Exception:
            return get_spectrum(use_rust=False)


@dataclass
class DDAMs1Frame(Frame):
    precursors: tuple[Precursor, ...]


@dataclass
class DiaWindow(_DiaWindowGroup, _TdfData):
    frame_id: int

    @property
    def peaks(self) -> list[npt.NDArray[np.float64]]:
        """Read raw peaks for this DIA window and return as list of (mz, intensity) arrays."""
        d: list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = (
            self.timsdata.readScans(
                self.frame_id, self.scan_num_begin, self.scan_num_end
            )
        )
        mz_int_arrays = []
        for index_array, int_array in d:
            mz_array = self.timsdata.indexToMz(self.frame_id, index_array)
            mz_int_arrays.append(
                np.stack((mz_array, int_array), axis=-1).astype(np.float64)
            )
        return mz_int_arrays

    def centroid(
        self,
        mz_tolerance: float = 8,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        im_tolerance: float = 0.05,
        im_tolerance_type: Literal["relative", "absolute"] = "relative",
        min_peaks: int = 3,
        max_peaks: int | None = None,
        noise_filter=None,
        ion_mobility_type: Literal["ccs", "ook0", "voltage"] = "ook0",
    ) -> np.ndarray:
        """Centroid the spectrum for this DIA window using the specified parameters."""
        get_spectrum = partial(
            get_centroided_spectrum,
            self.timsdata,
            frame_id=self.frame_id,
            spectrum_index=None,
            ion_mobility_type=ion_mobility_type,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            im_tolerance=im_tolerance,
            im_tolerance_type=im_tolerance_type,
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            noise_filter=noise_filter,
        )

        try:
            return get_spectrum(use_rust=True)
        except Exception:
            return get_spectrum(use_rust=False)

    @property
    def ook0_begin(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.frame_id, [self.scan_num_begin])[0]

    @property
    def ook0_end(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.frame_id, [self.scan_num_end])[0]

    @property
    def ccs_begin(self) -> float:
        return oneOverK0ToCCSforMz(self.ook0_begin, 1, self.isolation_mz)

    @property
    def ccs_end(self) -> float:
        return oneOverK0ToCCSforMz(self.ook0_end, 1, self.isolation_mz)

    @property
    def voltage_begin(self) -> float:
        return self.timsdata.scanNumToVoltage(self.frame_id, [self.scan_num_begin])[0]

    @property
    def voltage_end(self) -> float:
        return self.timsdata.scanNumToVoltage(self.frame_id, [self.scan_num_end])[0]


@dataclass
class DIAMs1Frame(Frame):
    dia_windows: tuple[DiaWindow, ...]


@dataclass
class _KeyDf:
    df: pd.DataFrame

    def __getitem__(self, key: str) -> str:
        if key not in self.df.index:
            raise KeyError(f"Calibration ID {key} not found.")
        return self.df.loc[key]


@dataclass
class Calibration(_KeyDf):
    """
    Calibration Table
        +	CalibrationDateTime	2018-08-21T16:50:31+02:00
        +	CalibrationUser	Demo User
        +	CalibrationSoftware	Bruker otofControl
        +	CalibrationSoftwareVersion	5.1.81.714-13047
        +	MzCalibrationMode	3
        +	MzStandardDeviationPPM	0.130754
        +	ReferenceMassList	Tuning Mix ES-TOF (ESI)
        +	MzCalibrationSpectrumDescription
        +	ReferenceMassPeakNames
        +	ReferencePeakMasses
        +	MeasuredTimesOfFlight
        +	MeasuredMassPeakIntensities
        +	MassesPreviousCalibration
        +	MassesCorrectedCalibration
        +	MobilityCalibrationDateTime	2018-08-21T16:49:17+02:00
        +	MobilityCalibrationUser	Demo User
        +	MobilityStandardDeviationPercent	0.000932
        +	ReferenceMobilityList	Tuning Mix ES-TOF (ESI)
        +	CalibrationMobilogramDescription
        +	ReferenceMobilityPeakNames
        +	ReferencePeakMobilities
        +	MeasuredTimsVoltages
        +	MeasuredMobilityPeakIntensities
        +	MobilitiesPreviousCalibration
        +	MobilitiesCorrectedCalibration
    """

    @property
    def date(self) -> datetime.datetime:
        # Example: CalibrationDateTime 2018-08-21T16:50:31+02:00
        cdatetime = self["CalibrationDateTime"]
        return datetime.datetime.fromisoformat(cdatetime)

    @property
    def user(self) -> str:
        return self["CalibrationUser"]

    @property
    def software(self) -> str:
        return self["CalibrationSoftware"]

    @property
    def software_version(self) -> str:
        return self["CalibrationSoftwareVersion"]

    @property
    def mode(self) -> str:
        return self["CalibrationMode"]

    @property
    def std_ppm(self) -> float:
        return float(self["CalibrationStdPpm"])

    @property
    def reference_masses(self) -> str:
        return self["ReferenceMassList"]

    @property
    def mobility_calibration_date(self) -> datetime.datetime:
        mcdatetime = self["MobilityCalibrationDateTime"]
        return datetime.datetime.fromisoformat(mcdatetime)

    @property
    def mobility_calibration_user(self) -> str:
        return self["MobilityCalibrationUser"]

    @property
    def mobility_standard_deviation_percent(self) -> float:
        return float(self["MobilityStandardDeviationPercent"])

    @property
    def reference_mobility_list(self) -> str:
        return self["ReferenceMobilityList"]


@dataclass
class MetaData(_KeyDf):
    """
    GlobalMetaData Table:
        SchemaType	TDF
        SchemaVersionMajor	3
        SchemaVersionMinor	1
        AcquisitionSoftwareVendor	Bruker
        InstrumentVendor	Bruker
        TimsCompressionType	2
        ClosedProperly	1
        MaxNumPeaksPerScan	1412
        AnalysisId	00000000-0000-0000-0000-000000000000
        DigitizerNumSamples	397600
        PeakListIndexScaleFactor	1
        MzAcqRangeLower	100.000000
        MzAcqRangeUpper	1700.000000
        OneOverK0AcqRangeLower	0.578703
        OneOverK0AcqRangeUpper	1.524471
        AcquisitionSoftware	Bruker otofControl
        AcquisitionSoftwareVersion	5.1.81.714-13047-vc110
        AcquisitionFirmwareVersion	<unknown>
        AcquisitionDateTime	2018-08-21T20:40:14.356+02:00
        InstrumentName	timsTOF Pro
        InstrumentFamily	9
        InstrumentRevision	1
        InstrumentSourceType	11
        InstrumentSerialNumber	1844426.34
        OperatorName	Demo User
        Description
        SampleName	200ngHeLaDIAPASEF_CE8V1st10VPASEF
        MethodName	20180813diagonSWATHPASEFref.m
    """

    @property
    def schema_type(self) -> str:
        """Schema type (typically 'TDF')."""
        return self["SchemaType"]

    @property
    def schema_version_major(self) -> int:
        """Major version of the TDF schema."""
        return int(self["SchemaVersionMajor"])

    @property
    def schema_version_minor(self) -> int:
        """Minor version of the TDF schema."""
        return int(self["SchemaVersionMinor"])

    @property
    def acquisition_software_vendor(self) -> str:
        """Vendor of acquisition software."""
        return self["AcquisitionSoftwareVendor"]

    @property
    def instrument_vendor(self) -> str:
        """Instrument vendor."""
        return self["InstrumentVendor"]

    @property
    def tims_compression_type(self) -> int:
        """TIMS data compression type."""
        return int(self["TimsCompressionType"])

    @property
    def closed_properly(self) -> bool:
        """Whether the acquisition was closed properly."""
        return bool(int(self["ClosedProperly"]))

    @property
    def max_num_peaks_per_scan(self) -> int:
        """Maximum number of peaks per scan."""
        return int(self["MaxNumPeaksPerScan"])

    @property
    def analysis_id(self) -> str:
        """Analysis UUID."""
        return self["AnalysisId"]

    @property
    def digitizer_num_samples(self) -> int:
        """Number of digitizer samples."""
        return int(self["DigitizerNumSamples"])

    @property
    def peak_list_index_scale_factor(self) -> int:
        """Peak list index scale factor."""
        return int(self["PeakListIndexScaleFactor"])

    @property
    def mz_acq_range_lower(self) -> float:
        """Lower m/z acquisition range."""
        return float(self["MzAcqRangeLower"])

    @property
    def mz_acq_range_upper(self) -> float:
        """Upper m/z acquisition range."""
        return float(self["MzAcqRangeUpper"])

    @property
    def mz_acq_range(self) -> tuple[float, float]:
        """M/z acquisition range as (lower, upper) tuple."""
        return (self.mz_acq_range_lower, self.mz_acq_range_upper)

    @property
    def one_over_k0_acq_range_lower(self) -> float:
        """Lower 1/K0 acquisition range."""
        return float(self["OneOverK0AcqRangeLower"])

    @property
    def one_over_k0_acq_range_upper(self) -> float:
        """Upper 1/K0 acquisition range."""
        return float(self["OneOverK0AcqRangeUpper"])

    @property
    def one_over_k0_acq_range(self) -> tuple[float, float]:
        """1/K0 acquisition range as (lower, upper) tuple."""
        return (self.one_over_k0_acq_range_lower, self.one_over_k0_acq_range_upper)

    # Acquisition software information
    @property
    def acquisition_software(self) -> str:
        """Acquisition software name."""
        return self["AcquisitionSoftware"]

    @property
    def acquisition_software_version(self) -> str:
        """Acquisition software version."""
        return self["AcquisitionSoftwareVersion"]

    @property
    def acquisition_firmware_version(self) -> str:
        """Acquisition firmware version."""
        return self["AcquisitionFirmwareVersion"]

    @property
    def acquisition_datetime(self) -> datetime.datetime:
        """Acquisition date and time."""
        adatetime = self["AcquisitionDateTime"]
        return datetime.datetime.fromisoformat(adatetime)

    # Instrument information
    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        return self["InstrumentName"]

    @property
    def instrument_family(self) -> int:
        """Instrument family code."""
        return int(self["InstrumentFamily"])

    @property
    def instrument_revision(self) -> int:
        """Instrument revision number."""
        return int(self["InstrumentRevision"])

    @property
    def instrument_source_type(self) -> int:
        """Instrument source type code."""
        return int(self["InstrumentSourceType"])

    @property
    def instrument_serial_number(self) -> str:
        """Instrument serial number."""
        return self["InstrumentSerialNumber"]

    # Sample and method information
    @property
    def operator_name(self) -> str:
        """Operator name."""
        return self["OperatorName"]

    @property
    def description(self) -> str:
        """Sample/acquisition description."""
        return self["Description"]

    @property
    def sample_name(self) -> str:
        """Sample name."""
        return self["SampleName"]

    @property
    def method_name(self) -> str:
        """Acquisition method name."""
        return self["MethodName"]


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

        self._timsdata = TimsData(str(self.analysis_path))

        pandas_tdf = PandasTdf(str(self.analysis_tdf_path))
        self._metadata = MetaData(df=pandas_tdf.global_metadata)
        self._calibration = Calibration(df=pandas_tdf.calibration_info)

    @property
    def metadata(self) -> MetaData:
        """Global metadata about the acquisition."""
        return self._metadata

    @property
    def calibration(self) -> Calibration:
        """Calibration information."""
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
        if self._timsdata.handle is None:
            raise RuntimeError("TimsData connection has been unexpectedly closed.")

    def close(self) -> None:
        """Close the TimsData connection."""
        if not self._closed:
            self._timsdata.close()
            self._closed = True

    def __enter__(self) -> "_DFolder":
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

        self._pasef_msms_infos: dict[int, list[PasefFrameMsmsInfo]] = {}
        for _, row in self._pasef_frame_msms_info_df.iterrows():
            pasef_info = PasefFrameMsmsInfo(
                _timsdata=self._timsdata,
                frame_id=int(row["Frame"]),
                scan_num_begin=int(row["ScanNumBegin"]),
                scan_num_end=int(row["ScanNumEnd"]),
                isolation_mz=float(row["IsolationMz"]),
                isolation_width=float(row["IsolationWidth"]),
                collision_energy=float(row["CollisionEnergy"]),
                precursor=int(row["Precursor"]) if not pd.isna(row["Precursor"]) else None,
            )
            if pasef_info.frame_id not in self._pasef_msms_infos:
                self._pasef_msms_infos[pasef_info.frame_id] = []
            self._pasef_msms_infos[pasef_info.frame_id].append(pasef_info)

        self._precursors: dict[int, Precursor] = {}
        self._frame_to_precursors: dict[int, list[Precursor]] = {}
        for _, row in self._precursor_df.iterrows():
            precursor_id = int(row["Id"])
            frame_id = int(row["Parent"])
            precursor = Precursor(
                _timsdata=self._timsdata,
                precurosr_id=precursor_id,
                largest_peak_mz=float(row["LargestPeakMz"]),
                average_mz=float(row["AverageMz"]),
                monoisotopic_mz=float(row["MonoisotopicMz"]) if not pd.isna(row["MonoisotopicMz"]) else None,
                charge=int(row["Charge"]) if not pd.isna(row["Charge"]) else None,
                scan_number=int(row["ScanNumber"]),
                intensity=float(row["Intensity"]),
                parent_frame=int(row["Parent"]),
                pasef_frame_msms_infos=tuple(self._pasef_msms_infos.get(frame_id, [])),
            )
            self._precursors[precursor_id] = precursor
            if frame_id not in self._frame_to_precursors:
                self._frame_to_precursors[frame_id] = []
            self._frame_to_precursors[frame_id].append(precursor)

        self._frames: dict[int, Frame] = {}
        self._ms1_frames: dict[int, DDAMs1Frame] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MS1_MSMSTYPE:
                precursors_for_frame: list[Precursor] = self._frame_to_precursors.get(
                    frame_id, []
                )
                frame = DDAMs1Frame(
                    _timsdata=self._timsdata,
                    frame_id=frame_id,
                    time=float(row["Time"]),
                    polarity=str(row["Polarity"]),
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
            elif msms_type == DDA_MS2_MSMSTYPE:
                pass
            else:
                raise ValueError(f"Unknown MsMsType {msms_type} for frame {frame_id}")

        # remove uneeded dataframes to save memory
        del self._precursor_df
        del self._frames_df
        del self._pasef_frame_msms_info_df

    @property
    def ms1_frames(self) -> Generator[DDAMs1Frame, None, None]:
        self._check_open()
        for frame in self._ms1_frames.values():
            yield frame

    @property
    def precursors(self) -> Generator[Precursor, None, None]:
        self._check_open()
        for precursor in self._precursors.values():
            yield precursor


class DIA(_DFolder):
    # doesnt have precursors frame

    def __init__(self, analysis_dir: str):
        super().__init__(analysis_dir)

        # frames
        self._frames_df = PandasTdf(str(self.analysis_tdf_path)).frames

        # window groups
        self._dia_frame_msms_windows_df = PandasTdf(
            str(self.analysis_tdf_path)
        ).dia_frame_msms_windows

        # frame to window groups
        self._dia_frame_msms_info = PandasTdf(
            str(self.analysis_tdf_path)
        ).dia_frame_msms_info

        self._dia_window_groups: dict[int, list[_DiaWindowGroup]] = {}
        for key, row in self._dia_frame_msms_windows_df.iterrows():
            window_id = int(key)  # type: ignore
            window = _DiaWindowGroup(
                window_id=window_id,
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
        for _, row in self._dia_frame_msms_info.iterrows():
            frame_id = int(row["Frame"])
            window_group_id = int(row["WindowGroup"])
            # each frame can have multiple window groups
            window_groups = self._dia_window_groups.get(window_group_id, [])
            if frame_id not in self._dia_windows:
                self._dia_windows[frame_id] = []
            for window_group in window_groups:
                dia_window = DiaWindow(
                    _timsdata=self._timsdata,
                    frame_id=frame_id,
                    window_id=window_group.window_id,
                    window_group=window_group.window_group,
                    scan_num_begin=window_group.scan_num_begin,
                    scan_num_end=window_group.scan_num_end,
                    isolation_mz=window_group.isolation_mz,
                    isolation_width=window_group.isolation_width,
                    collision_energy=window_group.collision_energy,
                )
                self._dia_windows[frame_id].append(dia_window)

        self._frames: dict[int, Frame] = {}
        self._ms1_frames: dict[int, DIAMs1Frame] = {}
        for _, row in self._frames_df.iterrows():
            frame_id = int(row["Id"])
            msms_type = int(row["MsMsType"])
            if msms_type == MS1_MSMSTYPE:
                frame = DIAMs1Frame(
                    _timsdata=self._timsdata,
                    frame_id=frame_id,
                    time=float(row["Time"]),
                    polarity=str(row["Polarity"]),
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
                    property_group=int(row["PropertyGroup"]) if not pd.isna(row["PropertyGroup"]) else None,
                    accumulation_time=float(row["AccumulationTime"]),
                    ramp_time=float(row["RampTime"]),
                    dia_windows=tuple(self._dia_windows.get(frame_id, [])),
                )
                self._ms1_frames[frame_id] = frame
                self._frames[frame_id] = frame
            elif msms_type == DIA_MS2_MSMSTYPE:
                pass
            else:
                raise ValueError(f"Unknown MsMsType {msms_type} for frame {frame_id}")

        # remove uneeded dataframes to save memory
        del self._frames_df
        del self._dia_frame_msms_windows_df
        del self._dia_frame_msms_info

    @property
    def ms1_frames(self) -> Generator[DIAMs1Frame, None, None]:
        self._check_open()
        for frame in self._ms1_frames.values():
            yield frame

    @property
    def windows(self) -> Generator[DiaWindow, None, None]:
        self._check_open()
        for window_list in self._dia_windows.values():
            for window in window_list:
                yield window


class PRM(_DFolder):
    # not implemented yet
    pass
