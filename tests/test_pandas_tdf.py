import unittest
import pandas as pd

from tdfpy import PandasTdf

TDF_PATH = r"tests/data/200ngHeLaPASEF_1min.d/analysis.tdf"


class TestPandasTDF(unittest.TestCase):
    pd_tdf = PandasTdf(TDF_PATH)

    def test_calibration_info(self):
        self.assertTrue(isinstance(self.pd_tdf.calibration_info, pd.DataFrame))

    def test_dia_frame_msms_info(self):
        self.assertTrue(isinstance(self.pd_tdf.dia_frame_msms_info, pd.DataFrame))

    def test_dia_frame_msms_window_groups(self):
        self.assertTrue(
            isinstance(self.pd_tdf.dia_frame_msms_window_groups, pd.DataFrame)
        )

    def test_dia_frame_msms_windows(self):
        self.assertTrue(isinstance(self.pd_tdf.dia_frame_msms_windows, pd.DataFrame))

    def test_error_log(self):
        self.assertTrue(isinstance(self.pd_tdf.error_log, pd.DataFrame))

    def test_frame_msms_info(self):
        self.assertTrue(isinstance(self.pd_tdf.frame_msms_info, pd.DataFrame))

    def test_frame_properties(self):
        self.assertTrue(isinstance(self.pd_tdf.frame_properties, pd.DataFrame))

    def test_frames(self):
        self.assertTrue(isinstance(self.pd_tdf.frames, pd.DataFrame))

    def test_global_metadata(self):
        self.assertTrue(isinstance(self.pd_tdf.global_metadata, pd.DataFrame))

    def test_group_properties(self):
        self.assertTrue(isinstance(self.pd_tdf.group_properties, pd.DataFrame))

    def test_mz_calibration(self):
        self.assertTrue(isinstance(self.pd_tdf.mz_calibration, pd.DataFrame))

    def test_pasef_frame_msms_info(self):
        self.assertTrue(isinstance(self.pd_tdf.pasef_frame_msms_info, pd.DataFrame))

    def test_precursors(self):
        self.assertTrue(isinstance(self.pd_tdf.precursors, pd.DataFrame))

    def test_properties(self):
        self.assertTrue(isinstance(self.pd_tdf.properties, pd.DataFrame))

    def test_property_definitions(self):
        self.assertTrue(isinstance(self.pd_tdf.property_definitions, pd.DataFrame))

    def test_property_groups(self):
        self.assertTrue(isinstance(self.pd_tdf.property_groups, pd.DataFrame))

    def test_segments(self):
        self.assertTrue(isinstance(self.pd_tdf.segments, pd.DataFrame))

    def test_tims_calibration(self):
        self.assertTrue(isinstance(self.pd_tdf.tims_calibration, pd.DataFrame))


if __name__ == "__main__":
    unittest.main()
