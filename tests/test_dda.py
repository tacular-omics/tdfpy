import pathlib

import numpy as np
import pytest

from tdfpy import DDA, get_acquisition_type

D_PATH = "tests/data/example_dda.d"
SKIP_NO_DATA = pytest.mark.skipif(
    not pathlib.Path(D_PATH).exists(), reason="Test data not available"
)


def test_get_acquisition_type():
    """Test that get_acquisition_type correctly identifies DDA data."""
    acq_type = get_acquisition_type(D_PATH)
    assert acq_type == "DDA"


def test_dda_precursors():
    with DDA(D_PATH) as dda:
        precursors = list(dda.precursors)
        assert len(precursors) == 2519

        # Precursor 1
        # Row 0: 1, 1293.138494, 1293.371888, 1292.637062, 2.0, 162.940341, 3603.0, 1
        p1 = next(p for p in precursors if p.precursor_id == 1)
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
        pasef = next(
            (info for info in p1.pasef_frame_msms_infos if info.frame_id == 2), None
        )
        assert pasef is not None
        assert pasef.scan_num_begin == 150
        assert pasef.scan_num_end == 175
        assert pasef.isolation_mz == pytest.approx(1293.371888)

        # Precursor 2519 (Last one in snippet)
        # Row 2518: 2519, 635.842079, 636.152187, 635.842079, 2.0, 375.340166, 11102.0, 700
        p_last = next(p for p in precursors if p.precursor_id == 2519)
        assert p_last.largest_peak_mz == pytest.approx(635.842079)
        assert p_last.average_mz == pytest.approx(636.152187)
        assert p_last.charge == 2
        assert p_last.scan_number == 375
        assert p_last.parent_frame == 700


def test_dda_frames():
    with DDA(D_PATH) as dda:
        # Note: DDA.ms1_frames only yields frames with MsMsType == 0
        ms1_frames = list(dda.ms1)

        # Frame 1
        # Row 0: 1, 2400.831487, +, 8, 0, 0, 35579, 31546080, 671, 337047, ...
        f1 = next(f for f in ms1_frames if f.frame_id == 1)
        assert f1.time == pytest.approx(2400.831487)
        assert f1.polarity == "positive"
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


def test_dda_lookup_features():
    with DDA(D_PATH) as dda:
        # Test Precursor Lookup by ID
        p1 = dda.precursors[1]
        assert p1.precursor_id == 1
        assert p1.charge == 2

        # Test MS1 Frame Lookup by ID
        f1 = dda.ms1[1]
        assert f1.frame_id == 1
        assert f1.msms_type == 0

        # Test Precursor Query by m/z
        # Precursor 1 has monoisotopic_mz 1292.637062
        mz_target = 1292.637062
        results = list(
            dda.precursors.query(
                mz=mz_target, mz_tolerance=0.01, mz_tolerance_type="da"
            )
        )
        found_ids = [p.precursor_id for p in results]
        assert 1 in found_ids

        # Test Precursor Query by RT
        # Frame 1 time is ~2400.83
        rt_target = 2400.83
        results_rt = list(dda.precursors.query(rt=rt_target, rt_tolerance=1.0))
        found_ids_rt = [p.precursor_id for p in results_rt]
        assert 1 in found_ids_rt

        # Test Invalid Lookups — messages should name the ID and the valid range.
        with pytest.raises(KeyError, match=r"Precursor ID 99999 not found"):
            _ = dda.precursors[99999]

        with pytest.raises(KeyError, match=r"MS1 frame ID 2 not found"):
            # Frame 2 is not MS1
            _ = dda.ms1[2]


@SKIP_NO_DATA
def test_dda_precursor_peaks():
    """Precursor.peaks returns a native-centroided (N, 2) m/z-sorted array."""
    with DDA(D_PATH) as dda:
        p1 = dda.precursors[1]
        peaks = p1.peaks

        assert isinstance(peaks, np.ndarray)
        assert peaks.ndim == 2
        assert peaks.shape[1] == 2
        assert peaks.shape[0] > 0
        assert np.issubdtype(peaks.dtype, np.floating)

        mz, intensity = peaks[:, 0], peaks[:, 1]
        assert np.all(mz > 0)
        assert np.all(np.isfinite(mz))
        assert np.all(intensity > 0)


@SKIP_NO_DATA
def test_dda_precursor_pasef_peaks():
    """pasef_peaks yields one (N, 2) array per PASEF MS/MS window."""
    with DDA(D_PATH) as dda:
        p1 = dda.precursors[1]
        pasef_peaks = p1.pasef_peaks

        assert isinstance(pasef_peaks, list)
        assert len(pasef_peaks) == len(p1.pasef_frame_msms_infos)
        assert len(pasef_peaks) > 0

        for arr in pasef_peaks:
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 2
            assert arr.shape[1] == 2
            if arr.shape[0] > 0:
                assert np.all(arr[:, 0] > 0)
                assert np.all(np.isfinite(arr[:, 0]))


@SKIP_NO_DATA
def test_dda_frame_raw_peaks():
    """Frame.raw_peaks() returns (N, 3) [mz, intensity, 1/K0]."""
    with DDA(D_PATH) as dda:
        f1 = dda.ms1[1]
        raw = f1.raw_peaks()

        assert isinstance(raw, np.ndarray)
        assert raw.ndim == 2
        assert raw.shape[1] == 3
        assert raw.shape[0] > 0
        assert np.issubdtype(raw.dtype, np.floating)
        assert np.all(raw[:, 0] > 0)  # m/z
        assert np.all(raw[:, 1] > 0)  # intensity
        assert np.all(raw[:, 2] > 0)  # 1/K0

        # Raw peaks are a superset of the centroided ones.
        assert raw.shape[0] >= f1.centroid().shape[0]


@SKIP_NO_DATA
def test_dda_precursor_mobility_properties():
    """Precursor exposes 1/K0 and CCS derived from its mobility scan."""
    with DDA(D_PATH) as dda:
        p1 = dda.precursors[1]

        ook0 = p1.ook0
        assert isinstance(ook0, float)
        assert 0.0 < ook0 < 3.0

        ccs = p1.ccs
        assert isinstance(ccs, float)
        assert ccs > 0.0


@SKIP_NO_DATA
def test_dda_access_after_close():
    """Spectral access after the `with` block must raise RuntimeError."""
    with DDA(D_PATH) as dda:
        frame = dda.ms1[1]
        precursor = dda.precursors[1]
        # Warm up: these all work while the reader is open.
        assert len(frame.peaks) > 0
        assert precursor.peaks.shape[1] == 2

    with pytest.raises(RuntimeError, match="closed"):
        _ = frame.peaks

    with pytest.raises(RuntimeError, match="closed"):
        frame.centroid()

    with pytest.raises(RuntimeError, match="closed"):
        _ = precursor.peaks

    with pytest.raises(RuntimeError, match="closed"):
        _ = dda.ms1

    with pytest.raises(RuntimeError, match="closed"):
        _ = dda.precursors

    # Current behaviour: metadata is read from the SQLite file on demand and
    # does not depend on the TimsData handle, so it stays available.
    assert isinstance(dda.metadata.instrument_name, str)


if __name__ == "__main__":
    # run file
    pytest.main([__file__])
