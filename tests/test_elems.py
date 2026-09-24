from typing import get_args

import pytest

from tdfpy.elems import DiaWindowGroup, MsMsType, Polarity, _parse_polarity


class TestMsMsType:
    def test_values(self):
        assert MsMsType.MS1.value == 0
        assert MsMsType.DDA_MS2.value == 8
        assert MsMsType.DIA_MS2.value == 9


class TestPolarity:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("+", "positive"),
            ("-", "negative"),
            ("positive", "positive"),
            ("NEGATIVE", "negative"),
            (" + ", "positive"),
            ("?", None),
            ("mixed", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parse(self, s, expected):
        assert _parse_polarity(s) == expected

    def test_polarity_is_a_literal_not_an_enum(self):
        assert get_args(Polarity) == ("positive", "negative")


class TestDiaWindowGroup:
    @pytest.fixture
    def window(self):
        return DiaWindowGroup(
            window_index=0,
            window_group_id=1,
            scan_num_begin=10,
            scan_num_end=50,
            isolation_mz=500.0,
            isolation_width=20.0,
            collision_energy=25.0,
        )

    def test_scan_num_range(self, window):
        assert window.scan_num_range == (10, 50)

    def test_isolation_mz_range(self, window):
        lo, hi = window.isolation_mz_range
        assert lo == pytest.approx(490.0)
        assert hi == pytest.approx(510.0)
        assert lo < hi

    def test_old_isolation_names_are_gone(self, window):
        for name in ("mz_begin", "mz_end", "mz_range"):
            assert not hasattr(window, name)
