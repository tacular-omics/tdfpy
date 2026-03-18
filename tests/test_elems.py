import warnings

import pytest

from tdfpy.elems import DiaWindowGroup, MsMsType, Polarity


class TestMsMsType:
    def test_values(self):
        assert MsMsType.MS1.value == 0
        assert MsMsType.DDA_MS2.value == 8
        assert MsMsType.DIA_MS2.value == 9


class TestPolarity:
    @pytest.mark.parametrize("s,expected", [
        ("positive", Polarity.POSITIVE),
        ("POSITIVE", Polarity.POSITIVE),
        ("+", Polarity.POSITIVE),
        ("negative", Polarity.NEGATIVE),
        ("NEGATIVE", Polarity.NEGATIVE),
        ("-", Polarity.NEGATIVE),
        ("unknown", Polarity.UNKNOWN),
        ("?", Polarity.UNKNOWN),
        ("mixed", Polarity.MIXED),
        ("mix", Polarity.MIXED),
    ])
    def test_from_str(self, s, expected):
        assert Polarity.from_str(s) == expected

    def test_from_str_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown polarity string"):
            Polarity.from_str("bogus")

    def test_str_enum_values(self):
        assert str(Polarity.POSITIVE) == "positive"
        assert str(Polarity.NEGATIVE) == "negative"
        assert str(Polarity.UNKNOWN) == "unknown"
        assert str(Polarity.MIXED) == "mixed"


class TestDiaWindowGroup:
    @pytest.fixture
    def window(self):
        return DiaWindowGroup(
            window_index=0,
            window_group=1,
            scan_num_begin=10,
            scan_num_end=50,
            isolation_mz=500.0,
            isolation_width=20.0,
            collision_energy=25.0,
        )

    def test_scan_num_range(self, window):
        assert window.scan_num_range == (10, 50)

    def test_mz_begin(self, window):
        assert window.mz_begin == pytest.approx(490.0)

    def test_mz_end(self, window):
        assert window.mz_end == pytest.approx(510.0)

    def test_mz_range(self, window):
        lo, hi = window.mz_range
        assert lo == pytest.approx(490.0)
        assert hi == pytest.approx(510.0)
        assert lo < hi
