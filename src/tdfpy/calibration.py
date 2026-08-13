"""Bruker timsTOF calibration models.

Pure functions over the ``MzCalibration`` and ``TimsCalibration`` rows of
``analysis.tdf``. No file or database access happens here, which keeps the models
unit-testable in isolation from any ``.d`` folder.

These reproduce Bruker's native ``tims_index_to_mz``, ``tims_mz_to_index``,
``tims_scannum_to_oneoverk0``, ``tims_oneoverk0_to_scannum`` and
``tims_scannum_to_voltage`` to within floating-point noise (~1e-10 relative for
m/z, ~1e-15 for mobility) on the bundled fixtures. See
``tests/test_calibration_golden.py``.

Only the model types observed on real data are implemented. Anything else raises
:class:`UnsupportedCalibrationError` rather than silently returning plausible but
wrong numbers — a wrong calibration is far more damaging than a hard failure,
because nothing downstream can detect it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "MZ_MODEL_TYPE",
    "TIMS_MODEL_TYPE",
    "MzCalibration",
    "TimsCalibration",
    "UnsupportedCalibrationError",
    "ccs_to_one_over_k0",
    "one_over_k0_to_ccs",
]

#: The only ``MzCalibration.ModelType`` this module implements.
MZ_MODEL_TYPE = 1
#: The only ``TimsCalibration.ModelType`` this module implements.
TIMS_MODEL_TYPE = 2

# Mason-Schamp constants, matching Bruker's tims_oneoverk0_to_ccs_for_mz.
#
# ``_CCS_K`` folds the 3/16 * sqrt(2*pi/k_B) * e/N0 prefactor and the unit
# conversions into a single number. The published value sits ~7 ppm from Bruker's
# internal constant; that is far below the precision at which CCS is physically
# meaningful (~0.1%), so the published, traceable value is preferred over one
# fitted to match the native library exactly.
_CCS_K = 18509.8632163405
_CCS_MASS_GAS = 28.013  # N2, Da
_CCS_TEMPERATURE = 31.85 + 273.15  # K

# Anything ``np.asarray`` accepts; conversions coerce to float64 on entry.
ArrayLike = npt.ArrayLike


class UnsupportedCalibrationError(NotImplementedError):
    """Raised for a calibration model type that has not been validated.

    Bruker ships several model types; only those seen on real data are
    implemented here. Rather than approximate an unknown model, callers are told
    to fall back to Bruker's native library.
    """


def _as_float_array(values: ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class MzCalibration:
    """TOF-index <-> m/z model (``MzCalibration.ModelType == 1``).

    Flight time is affine in the TOF sample index, and m/z is quadratic in flight
    time::

        t    = DigitizerDelay + tof_index * DigitizerTimebase
        t    = C0 + 1e6 * sqrt(mz / C1_eff) + C2 * mz

    ``C2`` is zero on many instruments, in which case the relation collapses to
    the familiar ``sqrt(mz) proportional to t``. ``C1`` is corrected for the drift
    between the temperatures recorded at calibration time (``T1``/``T2``) and
    those recorded on the frame being converted, scaled by ``dC1``/``dC2`` in
    parts per million.
    """

    model_type: int
    digitizer_timebase: float
    digitizer_delay: float
    t1: float
    t2: float
    dc1: float
    dc2: float
    c0: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        if self.model_type != MZ_MODEL_TYPE:
            raise UnsupportedCalibrationError(
                f"MzCalibration.ModelType {self.model_type} is not supported "
                f"(only {MZ_MODEL_TYPE} has been validated against Bruker's "
                "library). This file needs the native timsdata library."
            )

    @classmethod
    def from_row(cls, row: dict) -> "MzCalibration":
        """Build from a ``MzCalibration`` row (``sqlite3.Row`` or mapping)."""
        return cls(
            model_type=int(row["ModelType"]),
            digitizer_timebase=float(row["DigitizerTimebase"]),
            digitizer_delay=float(row["DigitizerDelay"]),
            t1=float(row["T1"]),
            t2=float(row["T2"]),
            dc1=float(row["dC1"]),
            dc2=float(row["dC2"]),
            c0=float(row["C0"]),
            c1=float(row["C1"]),
            c2=float(row["C2"]),
        )

    @property
    def min_tof_index(self) -> float:
        """Lowest TOF index this model can convert, i.e. the one where m/z is 0.

        Below it the modelled flight time is shorter than ``C0``, the root of the
        quadratic goes negative, and squaring it yields a positive but meaningless
        m/z. On the bundled fixtures this sits far below zero (about -1.2e5 to
        -2.9e5) because ``DigitizerDelay`` is much larger than ``C0``, so no real
        TOF index comes near it.
        """
        return (self.c0 - self.digitizer_delay) / self.digitizer_timebase

    def _c1_at(self, frame_t1: float, frame_t2: float) -> float:
        """``C1`` corrected for this frame's temperatures, in ppm of ``C1``."""
        drift = self.dc1 * (frame_t1 - self.t1) + self.dc2 * (frame_t2 - self.t2)
        return self.c1 * (1.0 - 1e-6 * drift)

    def index_to_mz(
        self, tof_index: ArrayLike, frame_t1: float, frame_t2: float
    ) -> npt.NDArray[np.float64]:
        """Convert TOF sample indices to m/z for a frame at the given temperatures.

        Raises:
            ValueError: If any index falls below :attr:`min_tof_index`, where the
                model has no non-negative m/z.
        """
        t = self.digitizer_delay + _as_float_array(tof_index) * self.digitizer_timebase
        k = 1e6 / np.sqrt(self._c1_at(frame_t1, frame_t2))
        d = t - self.c0
        # Below t == C0 the root is negative, and squaring it hands back a
        # perfectly plausible positive m/z that is simply wrong. Nothing
        # downstream can detect that, so refuse rather than approximate.
        below = d < 0.0
        if bool(np.any(below)):
            raise ValueError(
                f"index_to_mz: {int(np.count_nonzero(below))} of {below.size} TOF "
                f"index(es) fall below {self.min_tof_index:.6g}, where the modelled "
                f"flight time is shorter than C0 ({self.c0:.6g}) and the model has "
                "no non-negative m/z. Squaring the negative root would return a "
                "plausible but wrong m/z."
            )
        # Stable root of C2*x^2 + k*x - d = 0 for x = sqrt(mz). Written as
        # 2d / (k + sqrt(...)) rather than (-k + sqrt(...)) / 2*C2 so that it
        # stays accurate as C2 approaches zero and needs no separate branch.
        root = 2.0 * d / (k + np.sqrt(k * k + 4.0 * self.c2 * d))
        return root * root

    def mz_to_index(
        self, mz: ArrayLike, frame_t1: float, frame_t2: float
    ) -> npt.NDArray[np.float64]:
        """Convert m/z to (fractional) TOF sample indices."""
        mz_arr = _as_float_array(mz)
        t = (
            self.c0
            + 1e6 * np.sqrt(mz_arr / self._c1_at(frame_t1, frame_t2))
            + self.c2 * mz_arr
        )
        return (t - self.digitizer_delay) / self.digitizer_timebase


@dataclass(frozen=True)
class TimsCalibration:
    """Scan-number <-> voltage <-> 1/K0 model (``TimsCalibration.ModelType == 2``).

    The TIMS ramp voltage is affine in scan number, and inverse reduced mobility
    is a rational function of that voltage::

        V    = C2 + (C3 - C2) / C1 * (scan - C0 - C4)
        1/K0 = V / (C7 + C6 * V)

    Voltage falls with scan number, so 1/K0 decreases monotonically across the
    ramp. ``C5``, ``C8`` and ``C9`` are unused by this model; ``C8``/``C9`` are
    believed to drive pressure compensation, which is not implemented.
    """

    model_type: int
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c6: float
    c7: float

    def __post_init__(self) -> None:
        if self.model_type != TIMS_MODEL_TYPE:
            raise UnsupportedCalibrationError(
                f"TimsCalibration.ModelType {self.model_type} is not supported "
                f"(only {TIMS_MODEL_TYPE} has been validated against Bruker's "
                "library). This file needs the native timsdata library."
            )

    @classmethod
    def from_row(cls, row: dict) -> "TimsCalibration":
        """Build from a ``TimsCalibration`` row (``sqlite3.Row`` or mapping)."""
        return cls(
            model_type=int(row["ModelType"]),
            c0=float(row["C0"]),
            c1=float(row["C1"]),
            c2=float(row["C2"]),
            c3=float(row["C3"]),
            c4=float(row["C4"]),
            c6=float(row["C6"]),
            c7=float(row["C7"]),
        )

    def scan_to_voltage(self, scan: ArrayLike) -> npt.NDArray[np.float64]:
        """Convert scan numbers to TIMS ramp voltage."""
        return self.c2 + (self.c3 - self.c2) / self.c1 * (
            _as_float_array(scan) - self.c0 - self.c4
        )

    def voltage_to_scan(self, voltage: ArrayLike) -> npt.NDArray[np.float64]:
        """Convert TIMS ramp voltage to (fractional) scan numbers."""
        return (
            (_as_float_array(voltage) - self.c2) * self.c1 / (self.c3 - self.c2)
            + self.c0
            + self.c4
        )

    def scan_to_one_over_k0(self, scan: ArrayLike) -> npt.NDArray[np.float64]:
        """Convert scan numbers to inverse reduced mobility (1/K0)."""
        v = self.scan_to_voltage(scan)
        return v / (self.c7 + self.c6 * v)

    def one_over_k0_to_scan(self, one_over_k0: ArrayLike) -> npt.NDArray[np.float64]:
        """Convert inverse reduced mobility (1/K0) to (fractional) scan numbers."""
        y = _as_float_array(one_over_k0)
        return self.voltage_to_scan(self.c7 * y / (1.0 - self.c6 * y))


def one_over_k0_to_ccs(one_over_k0: float, charge: int, mz: float) -> float:
    """Convert 1/K0 to a collision cross section via the Mason-Schamp equation.

    Follows Bruker's convention of treating ``mz * charge`` as the ion mass
    without subtracting proton masses; changing that would diverge from the
    native library by ~600 ppm.
    """
    mass = mz * charge
    reduced_mass = (mass * _CCS_MASS_GAS) / (mass + _CCS_MASS_GAS)
    return float(
        _CCS_K * charge / np.sqrt(reduced_mass * _CCS_TEMPERATURE) * one_over_k0
    )


def ccs_to_one_over_k0(ccs: float, charge: int, mz: float) -> float:
    """Inverse of :func:`one_over_k0_to_ccs`."""
    mass = mz * charge
    reduced_mass = (mass * _CCS_MASS_GAS) / (mass + _CCS_MASS_GAS)
    return float(np.sqrt(reduced_mass * _CCS_TEMPERATURE) * ccs / (_CCS_K * charge))
