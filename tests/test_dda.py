
import pytest
from tdfpy import DDA

D_PATH = "tests/data/200ngHeLaPASEF_1min.d"

def test_dda_precursors():
    with DDA(D_PATH) as dda:
        precursors = list(dda.precursors)
        assert len(precursors) == 2519

        # Precursor 1
        # Row 0: 1, 1293.138494, 1293.371888, 1292.637062, 2.0, 162.940341, 3603.0, 1
        p1 = next(p for p in precursors if p.precurosr_id == 1)
        assert p1.largest_peak_mz == pytest.approx(1293.138494)
        assert p1.average_mz == pytest.approx(1293.371888)
        assert p1.monoisotopic_mz == pytest.approx(1292.637062)
        assert p1.charge == 2
        # Note: ScanNumber is truncated to int in reader.py
        assert p1.scan_number == 162 
        assert p1.intensity == pytest.approx(3603.0)
        assert p1.parent_frame == 1
        
        print(p1)

        # Check PASEF info for Precursor 1
        # From output: Frame 2, ScanNumBegin 150, ScanNumEnd 175, IsolationMz 1293.371888, Precursor 1
        assert len(p1.pasef_frame_msms_infos) == 6
        pasef = next((info for info in p1.pasef_frame_msms_infos if info.frame_id == 2), None)
        assert pasef is not None
        assert pasef.scan_num_begin == 150
        assert pasef.scan_num_end == 175
        assert pasef.isolation_mz == pytest.approx(1293.371888)

        # Precursor 2519 (Last one in snippet)
        # Row 2518: 2519, 635.842079, 636.152187, 635.842079, 2.0, 375.340166, 11102.0, 700
        p_last = next(p for p in precursors if p.precurosr_id == 2519)
        assert p_last.largest_peak_mz == pytest.approx(635.842079)
        assert p_last.average_mz == pytest.approx(636.152187)
        assert p_last.charge == 2
        assert p_last.scan_number == 375
        assert p_last.parent_frame == 700

def test_dda_frames():
    with DDA(D_PATH) as dda:
        # Note: DDA.ms1_frames only yields frames with MsMsType == 0
        ms1_frames = list(dda.ms1_frames)
        
        # Frame 1
        # Row 0: 1, 2400.831487, +, 8, 0, 0, 35579, 31546080, 671, 337047, ...
        f1 = next(f for f in ms1_frames if f.frame_id == 1)
        assert f1.time == pytest.approx(2400.831487)
        assert f1.polarity == "+"
        assert f1.scan_mode == 8
        assert f1.msms_type == 0
        assert f1.max_intensity == 35579
        assert f1.num_scans == 671
        assert f1.num_peaks == 337047
        assert f1.accumulation_time == pytest.approx(73.03)
        assert f1.ramp_time == pytest.approx(73.03)

        # Verify Frame 2 (MsMsType 8) is NOT in ms1_frames
        f2 = next((f for f in ms1_frames if f.frame_id == 2), None)
        assert f2 is None


if __name__ == "__main__":
    # run file
    pytest.main([__file__])