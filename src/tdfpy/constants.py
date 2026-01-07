"""
This module defines constants and an enumeration related to the 'tdfpy' package.
"""

from enum import StrEnum

PROTON_MASS = 1.007276466


class TableNames(StrEnum):
    """
    Enum to store all possible analysis.tdf table names
    """

    CALIBRATION_INFO = "CalibrationInfo"
    DIA_FRAME_MSMS_INFO = "DiaFrameMsMsInfo"
    DIA_FRAME_MSMS_WINDOW_GROUPS = "DiaFrameMsMsWindowGroups"
    DIA_FRAME_MSMS_WINDOWS = "DiaFrameMsMsWindows"
    ERROR_LOG = "ErrorLog"
    FRAME_MSMS_WINDOW = "FrameMsMsInfo"
    FRAME_PROPERTIES = "FrameProperties"
    FRAMES = "Frames"
    GLOBAL_METADATA = "GlobalMetadata"
    GROUP_PROPERTIES = "GroupProperties"
    MZ_CALIBRATION = "MzCalibration"
    PASEF_FRAME_MSMS_INFO = "PasefFrameMsMsInfo"
    PRECURSORS = "Precursors"
    PROPERTIES = "Properties"
    PROPERTY_DEFINITIONS = "PropertyDefinitions"
    PROPERTY_GROUPS = "PropertyGroups"
    SEGMENTS = "Segments"
    TIMS_CALIBRATION = "TimsCalibration"
    PRM_FRAME_MEASUREMENT_MODE = "PrmFrameMeasurementMode"
    PRM_FRAME_MSMS_INFO = "PrmFrameMsMsInfo"
    PRM_TARGETS = "PrmTargets"
