import pytest

from tdfpy.constants import PROTON_MASS, TableNames


class TestProtonMass:
    def test_value(self):
        assert PROTON_MASS == pytest.approx(1.007276466, rel=1e-6)

    def test_type(self):
        assert isinstance(PROTON_MASS, float)


class TestTableNames:
    def test_is_str_enum(self):
        assert TableNames.FRAMES == "Frames"
        assert TableNames.PRECURSORS == "Precursors"
        assert TableNames.GLOBAL_METADATA == "GlobalMetadata"
        assert TableNames.CALIBRATION_INFO == "CalibrationInfo"

    def test_all_members_are_strings(self):
        for member in TableNames:
            assert isinstance(member.value, str)

    def test_str_coercible(self):
        # StrEnum members should compare equal to their string values
        assert str(TableNames.FRAMES) == "Frames"
        assert str(TableNames.DIA_FRAME_MSMS_INFO) == "DiaFrameMsMsInfo"
