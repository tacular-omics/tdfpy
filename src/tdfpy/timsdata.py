import logging
import os
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
    cdll,
    create_string_buffer,
)
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Get current platform
platform = sys.platform
logger.debug(f"sys.platform: {platform}")

# Dictionary to map platforms to their respective libraries
platform_lib_map = {
    "win32": "timsdata.dll",
    "cygwin": "timsdata.dll",
    "linux": "libtimsdata.so",
}


# Function to get library name based on the platform
def get_lib_name(platform: str) -> str:
    for key, value in platform_lib_map.items():
        if platform.startswith(key):
            return value
    raise Exception("Unsupported platform.")


# Get library name based on the platform
libname = get_lib_name(platform)
logger.debug(f"platform: {platform} selected, libname: {libname}")

# Get data directory
data_dir = os.path.dirname(sys.modules["tdfpy"].__file__)  # type: ignore[type-var]

# Construct data path
data_path = os.path.join(data_dir, libname)  # type: ignore[arg-type]

logger.debug(f"data_path: {data_path}")

# Try to load library
try:
    if os.path.exists(data_path):
        dll = cdll.LoadLibrary(data_path)
    else:
        logger.debug(f"{data_path} does not exist, trying {libname}")
        dll = cdll.LoadLibrary(libname)
except Exception as e:
    logger.error(f"Error loading library: {e}")
    raise

dll.tims_open_v2.argtypes = [c_char_p, c_uint32, c_uint32]
dll.tims_open_v2.restype = c_uint64
dll.tims_close.argtypes = [c_uint64]
dll.tims_close.restype = None
dll.tims_get_last_error_string.argtypes = [c_char_p, c_uint32]
dll.tims_get_last_error_string.restype = c_uint32
dll.tims_has_recalibrated_state.argtypes = [c_uint64]
dll.tims_has_recalibrated_state.restype = c_uint32
dll.tims_read_scans_v2.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    c_void_p,
    c_uint32,
]
dll.tims_read_scans_v2.restype = c_uint32
MSMS_SPECTRUM_FUNCTOR = CFUNCTYPE(
    None, c_int64, c_uint32, POINTER(c_double), POINTER(c_float)
)
dll.tims_read_pasef_msms.argtypes = [
    c_uint64,
    POINTER(c_int64),
    c_uint32,
    MSMS_SPECTRUM_FUNCTOR,
]
dll.tims_read_pasef_msms.restype = c_uint32
dll.tims_read_pasef_msms_for_frame.argtypes = [c_uint64, c_int64, MSMS_SPECTRUM_FUNCTOR]
dll.tims_read_pasef_msms_for_frame.restype = c_uint32
MSMS_PROFILE_SPECTRUM_FUNCTOR = CFUNCTYPE(None, c_int64, c_uint32, POINTER(c_int32))
dll.tims_read_pasef_profile_msms.argtypes = [
    c_uint64,
    POINTER(c_int64),
    c_uint32,
    MSMS_PROFILE_SPECTRUM_FUNCTOR,
]
dll.tims_read_pasef_profile_msms.restype = c_uint32
dll.tims_read_pasef_profile_msms_for_frame.argtypes = [
    c_uint64,
    c_int64,
    MSMS_PROFILE_SPECTRUM_FUNCTOR,
]
dll.tims_read_pasef_profile_msms_for_frame.restype = c_uint32

dll.tims_extract_centroided_spectrum_for_frame_v2.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    MSMS_SPECTRUM_FUNCTOR,
    c_void_p,
]
dll.tims_extract_centroided_spectrum_for_frame_v2.restype = c_uint32
dll.tims_extract_centroided_spectrum_for_frame_ext.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    c_double,
    MSMS_SPECTRUM_FUNCTOR,
    c_void_p,
]
dll.tims_extract_centroided_spectrum_for_frame_ext.restype = c_uint32
dll.tims_extract_profile_for_frame.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    MSMS_PROFILE_SPECTRUM_FUNCTOR,
    c_void_p,
]
dll.tims_extract_profile_for_frame.restype = c_uint32


class ChromatogramJob(Structure):
    _fields_ = [
        ("id", c_int64),
        ("time_begin", c_double),
        ("time_end", c_double),
        ("mz_min", c_double),
        ("mz_max", c_double),
        ("ook0_min", c_double),
        ("ook0_max", c_double),
    ]


CHROMATOGRAM_JOB_GENERATOR = CFUNCTYPE(c_uint32, POINTER(ChromatogramJob), c_void_p)
CHROMATOGRAM_TRACE_SINK = CFUNCTYPE(
    c_uint32, c_int64, c_uint32, POINTER(c_int64), POINTER(c_uint64), c_void_p
)
dll.tims_extract_chromatograms.argtypes = [
    c_uint64,
    CHROMATOGRAM_JOB_GENERATOR,
    CHROMATOGRAM_TRACE_SINK,
    c_void_p,
]
dll.tims_extract_chromatograms.restype = c_uint32

convfunc_argtypes: list[Any] = [
    c_uint64,
    c_int64,
    POINTER(c_double),
    POINTER(c_double),
    c_uint32,
]

dll.tims_index_to_mz.argtypes = convfunc_argtypes
dll.tims_index_to_mz.restype = c_uint32
dll.tims_mz_to_index.argtypes = convfunc_argtypes
dll.tims_mz_to_index.restype = c_uint32

dll.tims_scannum_to_oneoverk0.argtypes = convfunc_argtypes
dll.tims_scannum_to_oneoverk0.restype = c_uint32
dll.tims_oneoverk0_to_scannum.argtypes = convfunc_argtypes
dll.tims_oneoverk0_to_scannum.restype = c_uint32

dll.tims_scannum_to_voltage.argtypes = convfunc_argtypes
dll.tims_scannum_to_voltage.restype = c_uint32
dll.tims_voltage_to_scannum.argtypes = convfunc_argtypes
dll.tims_voltage_to_scannum.restype = c_uint32

dll.tims_oneoverk0_to_ccs_for_mz.argtypes = [c_double, c_int32, c_double]
dll.tims_oneoverk0_to_ccs_for_mz.restype = c_double

dll.tims_ccs_to_oneoverk0_for_mz.argtypes = [c_double, c_int32, c_double]
dll.tims_ccs_to_oneoverk0_for_mz.restype = c_double


@contextmanager
def timsdata_connect(analysis_dir: str) -> Iterator["TimsData"]:
    td: TimsData | None = None
    try:
        td = TimsData(analysis_dir)
        yield td
    finally:
        if td:
            td.close()


def _throwLastTimsDataError(dll_handle: CDLL) -> None:
    """Throw last TimsData error string as an exception."""

    err_len = dll_handle.tims_get_last_error_string(None, 0)
    buf = create_string_buffer(err_len)
    dll_handle.tims_get_last_error_string(buf, err_len)
    raise RuntimeError(buf.value)


# Convert 1/K0 to CCS for a given charge and mz
def oneOverK0ToCCSforMz(ook0: float, charge: int, mz: float) -> float:
    return float(dll.tims_oneoverk0_to_ccs_for_mz(ook0, charge, mz))


# Convert CCS to 1/K0 for a given charge and mz
def ccsToOneOverK0ToCCSforMz(ccs: float, charge: int, mz: float) -> float:
    return float(dll.tims_ccs_to_oneoverk0_for_mz(ccs, charge, mz))


class PressureCompensationStrategy(Enum):
    NoPressureCompensation = 0
    AnalysisGlobalPressureCompensation = 1
    PerFramePressureCompensation = 2


class TimsData:
    def __init__(
        self,
        analysis_directory: str,
        use_recalibrated_state: bool = False,
        pressure_compensation_strategy: PressureCompensationStrategy = PressureCompensationStrategy.NoPressureCompensation,
    ) -> None:
        if not isinstance(analysis_directory, str):  # type: ignore[type-var]
            raise ValueError("analysis_directory must be a string.")

        self.analysis_directory = analysis_directory
        self.dll: CDLL = dll
        self.handle: int | None
        self.conn: sqlite3.Connection | None
        self.initial_frame_buffer_size: float

        self.handle = self.dll.tims_open_v2(
            analysis_directory.encode("utf-8"),
            1 if use_recalibrated_state else 0,
            pressure_compensation_strategy.value,
        )
        if self.handle == 0:
            _throwLastTimsDataError(self.dll)

        self.conn = sqlite3.connect(os.path.join(analysis_directory, "analysis.tdf"))

        self.initial_frame_buffer_size = 128  # may grow in readScans()

    def __enter__(self) -> "TimsData":
        return self

    def __exit__(self, exit_type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "handle") and self.handle is not None:
            self.dll.tims_close(self.handle)
            self.handle = None
        if hasattr(self, "conn") and self.conn is not None:
            self.conn.close()
            self.conn = None

    def __callConversionFunc(
        self,
        frame_id: int,
        input_data: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
        func: Callable[..., int],
    ) -> npt.NDArray[np.float64]:
        if type(input_data) is np.ndarray and input_data.dtype == np.float64:
            # already "native" format understood by DLL -> avoid extra copy
            in_array = input_data
        else:
            # convert data to format understood by DLL:
            in_array = np.array(input_data, dtype=np.float64)

        cnt = len(in_array)
        out = np.empty(shape=cnt, dtype=np.float64)
        success = func(
            self.handle,
            frame_id,
            in_array.ctypes.data_as(POINTER(c_double)),
            out.ctypes.data_as(POINTER(c_double)),
            cnt,
        )

        if success == 0:
            _throwLastTimsDataError(self.dll)

        return out

    def indexToMz(
        self,
        frame_id: int,
        indices: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
    ) -> npt.NDArray[np.float64]:
        return self.__callConversionFunc(frame_id, indices, self.dll.tims_index_to_mz)

    def mzToIndex(
        self,
        frame_id: int,
        mzs: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
    ) -> npt.NDArray[np.float64]:
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_mz_to_index)

    def scanNumToOneOverK0(
        self,
        frame_id: int,
        scan_nums: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
    ) -> npt.NDArray[np.float64]:
        return self.__callConversionFunc(
            frame_id, scan_nums, self.dll.tims_scannum_to_oneoverk0
        )

    def oneOverK0ToScanNum(
        self,
        frame_id: int,
        mobilities: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
    ) -> npt.NDArray[np.float64]:
        return self.__callConversionFunc(
            frame_id, mobilities, self.dll.tims_oneoverk0_to_scannum
        )

    def scanNumToVoltage(
        self,
        frame_id: int,
        scan_nums: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
    ) -> npt.NDArray[np.float64]:
        return self.__callConversionFunc(
            frame_id, scan_nums, self.dll.tims_scannum_to_voltage
        )

    def voltageToScanNum(
        self,
        frame_id: int,
        voltages: npt.NDArray[np.float64] | npt.NDArray[np.uint32] | list[float],
    ) -> npt.NDArray[np.float64]:
        return self.__callConversionFunc(
            frame_id, voltages, self.dll.tims_voltage_to_scannum
        )

    def readScansDllBuffer(
        self, frame_id: int, scan_begin: int, scan_end: int
    ) -> npt.NDArray[np.uint32]:
        """Read a range of scans from a frame, returning the data in the low-level buffer format defined for
        the 'tims_read_scans_v2' DLL function (see documentation in 'timsdata.h').

        """

        # buffer-growing loop
        while True:
            cnt = int(
                self.initial_frame_buffer_size
            )  # necessary cast to run with python 3.5
            buf = np.empty(shape=cnt, dtype=np.uint32)
            buf_len = 4 * cnt

            required_len = self.dll.tims_read_scans_v2(
                self.handle,
                frame_id,
                scan_begin,
                scan_end,
                buf.ctypes.data_as(POINTER(c_uint32)),
                buf_len,
            )
            if required_len == 0:
                _throwLastTimsDataError(self.dll)

            if required_len > buf_len:
                if required_len > 16777216:
                    # arbitrary limit for now...
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.initial_frame_buffer_size = required_len / 4 + 1  # grow buffer
            else:
                break

        return buf

    def readScans(
        self, frame_id: int, scan_begin: int, scan_end: int
    ) -> list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]]:
        """Read a range of scans from a frame, returning a list of scans, each scan being represented as a
        tuple (index_array, intensity_array).

        """

        buf = self.readScansDllBuffer(frame_id, scan_begin, scan_end)

        result: list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = []
        d = scan_end - scan_begin
        for i in range(scan_begin, scan_end):
            npeaks = buf[i - scan_begin]
            indices = buf[d : d + npeaks]
            d += npeaks
            intensities = buf[d : d + npeaks]
            d += npeaks
            result.append((indices, intensities))

        return result

    # read some peak-picked MS/MS spectra for a given list of precursors; returns a dict mapping
    # 'precursor_id' to a pair of arrays (mz_values, area_values).
    def readPasefMsMs(self, precursor_list: list[int]) -> dict[int, tuple[Any, Any]]:
        precursors_for_dll = np.array(precursor_list, dtype=np.int64)

        result: dict[int, tuple[Any, Any]] = {}

        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(
            precursor_id: int, num_peaks: int, mz_values: Any, area_values: Any
        ) -> None:
            result[precursor_id] = (mz_values[0:num_peaks], area_values[0:num_peaks])

        rc = self.dll.tims_read_pasef_msms(
            self.handle,
            precursors_for_dll.ctypes.data_as(POINTER(c_int64)),
            len(precursor_list),
            callback_for_dll,
        )

        if rc == 0:
            _throwLastTimsDataError(self.dll)

        return result

    # read peak-picked MS/MS spectra for a given frame; returns a dict mapping
    # 'precursor_id' to a pair of arrays (mz_values, area_values).
    def readPasefMsMsForFrame(self, frame_id: int) -> dict[int, tuple[Any, Any]]:
        result: dict[int, tuple[Any, Any]] = {}

        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(
            precursor_id: int, num_peaks: int, mz_values: Any, area_values: Any
        ) -> None:
            result[precursor_id] = (mz_values[0:num_peaks], area_values[0:num_peaks])

        rc = self.dll.tims_read_pasef_msms_for_frame(
            self.handle, frame_id, callback_for_dll
        )

        if rc == 0:
            _throwLastTimsDataError(self.dll)

        return result

    # read some "quasi profile" MS/MS spectra for a given list of precursors; returns a dict mapping
    # 'precursor_id' to the profile arrays (intensity_values).
    def readPasefProfileMsMs(self, precursor_list: list[int]) -> dict[int, Any]:
        precursors_for_dll = np.array(precursor_list, dtype=np.int64)

        result: dict[int, Any] = {}

        @MSMS_PROFILE_SPECTRUM_FUNCTOR
        def callback_for_dll(
            precursor_id: int, num_points: int, intensity_values: Any
        ) -> None:
            result[precursor_id] = intensity_values[0:num_points]

        rc = self.dll.tims_read_pasef_profile_msms(
            self.handle,
            precursors_for_dll.ctypes.data_as(POINTER(c_int64)),
            len(precursor_list),
            callback_for_dll,
        )

        if rc == 0:
            _throwLastTimsDataError(self.dll)

        return result

    # read "quasi profile" MS/MS spectra for a given frame; returns a dict mapping
    # 'precursor_id' to the profile arrays (intensity_values).
    def readPasefProfileMsMsForFrame(self, frame_id: int) -> dict[int, Any]:
        result: dict[int, Any] = {}

        @MSMS_PROFILE_SPECTRUM_FUNCTOR
        def callback_for_dll(
            precursor_id: int, num_points: int, intensity_values: Any
        ) -> None:
            result[precursor_id] = intensity_values[0:num_points]

        rc = self.dll.tims_read_pasef_profile_msms_for_frame(
            self.handle, frame_id, callback_for_dll
        )

        if rc == 0:
            _throwLastTimsDataError(self.dll)

        return result

    # read peak-picked spectra for a tims frame;
    # returns a pair of arrays (mz_values, area_values).
    def extractCentroidedSpectrumForFrame(
        self,
        frame_id: int,
        scan_begin: int,
        scan_end: int,
        peak_picker_resolution: float | None = None,
    ) -> tuple[Any, Any] | None:
        result: tuple[Any, Any] | None = None

        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(
            precursor_id: int, num_peaks: int, mz_values: Any, area_values: Any
        ) -> None:
            nonlocal result
            result = (mz_values[0:num_peaks], area_values[0:num_peaks])

        if peak_picker_resolution is None:
            rc = self.dll.tims_extract_centroided_spectrum_for_frame_v2(
                self.handle, frame_id, scan_begin, scan_end, callback_for_dll, None
            )  # python dos not need the additional context, we have nonlocal
        else:
            rc = self.dll.tims_extract_centroided_spectrum_for_frame_ext(
                self.handle,
                frame_id,
                scan_begin,
                scan_end,
                peak_picker_resolution,
                callback_for_dll,
                None,
            )  # python dos not need the additional context, we have nonlocal

        if rc == 0:
            _throwLastTimsDataError(self.dll)

        return result

    # read "quasi profile" spectra for a tims frame;
    # returns the profile array (intensity_values).
    def extractProfileForFrame(
        self, frame_id: int, scan_begin: int, scan_end: int
    ) -> Any | None:
        result: Any | None = None

        @MSMS_PROFILE_SPECTRUM_FUNCTOR
        def callback_for_dll(
            precursor_id: int, num_points: int, intensity_values: Any
        ) -> None:
            nonlocal result
            result = intensity_values[0:num_points]

        rc = self.dll.tims_extract_profile_for_frame(
            self.handle, frame_id, scan_begin, scan_end, callback_for_dll, None
        )  # python dos not need the additional context, we have nonlocal

        if rc == 0:
            _throwLastTimsDataError(self.dll)

        return result

    def extractChromatograms(
        self,
        jobs: Iterator[ChromatogramJob],
        trace_sink: Callable[
            [int, npt.NDArray[np.int64], npt.NDArray[np.uint64]], None
        ],
    ) -> None:
        """Efficiently extract several MS1-only extracted-ion chromatograms.

        The argument 'jobs' defines which chromatograms are to be extracted; it must be an iterator
        (generator) object producing a stream of ChromatogramJob objects. The jobs must be produced
        in the order of ascending 'time_begin'.

        The function 'trace_sink' is called for each extracted trace with three arguments: job ID,
        numpy array of frame IDs ("x-axis"), numpy array of chromatogram values ("y-axis").

        For more information, see the documentation of the C-language API of the timsdata DLL.

        """

        @CHROMATOGRAM_JOB_GENERATOR
        def wrap_gen(job: Any, user_data: Any) -> int:
            try:
                job[0] = next(jobs)
                return 1
            except StopIteration:
                return 2
            except Exception as e:
                logger.error("extractChromatograms: generator produced exception ", e)
                return 0

        @CHROMATOGRAM_TRACE_SINK
        def wrap_sink(
            job_id: int, num_points: int, frame_ids: Any, values: Any, user_data: Any
        ) -> int:
            try:
                trace_sink(
                    job_id,
                    np.array(frame_ids[0:num_points], dtype=np.int64),
                    np.array(values[0:num_points], dtype=np.uint64),
                )
                return 1
            except Exception as e:
                logger.error("extractChromatograms: sink produced exception ", e)
                return 0

        unused_user_data = 0
        rc = self.dll.tims_extract_chromatograms(
            self.handle, wrap_gen, wrap_sink, unused_user_data
        )

        if rc == 0:
            _throwLastTimsDataError(self.dll)
