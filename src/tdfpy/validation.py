"""Read-only acquisition checks with structured, frame-specific issues."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Literal

import numpy as np

from .calibration import UnsupportedCalibrationError
from .timsdata import TimsData, UnsupportedTdfError


@dataclass(frozen=True)
class ValidationIssue:
    """An acquisition-level or frame-specific validation failure."""

    frame_id: int | None
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Validation outcome. Metadata mode does not verify compressed payloads."""

    analysis_directory: str
    mode: Literal["metadata", "full"]
    frames_checked: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_acquisition(
    analysis_dir: str | Path, *, full: bool = False
) -> ValidationReport:
    """Check supported metadata, optionally decoding every frame.

    Uses the same strict format and calibration guards as extraction. Collects
    per-frame failures and continues through the remaining frames. This is
    structural validation, not a comparison with independent vendor spectra.
    """
    issues = []
    checked = 0
    failures = (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        sqlite3.Error,
        UnsupportedTdfError,
        UnsupportedCalibrationError,
    )
    try:
        with TimsData(analysis_dir) as td:
            for fid in td.frame_ids:
                checked += 1
                try:
                    frame = td.frame_metadata(fid)
                    if (
                        frame.num_scans < 0
                        or frame.num_peaks < 0
                        or not np.isfinite(frame.time)
                    ):
                        raise ValueError("Invalid frame counts or retention time.")
                    td.calibration_key(fid)
                    # Exercise calibration references and finite conversion values.
                    mz = td.indexToMz(fid, [0])
                    mobility = td.scanNumToOneOverK0(fid, [0])
                    if not np.all(np.isfinite(mz)) or not np.all(np.isfinite(mobility)):
                        raise ValueError("Calibration produces non-finite coordinates.")
                    if full:
                        td.read_frame_arrays(fid)
                except failures as exc:
                    issues.append(ValidationIssue(fid, str(exc)))
    except failures as exc:
        issues.append(ValidationIssue(None, str(exc)))
    return ValidationReport(
        str(analysis_dir), "full" if full else "metadata", checked, tuple(issues)
    )
