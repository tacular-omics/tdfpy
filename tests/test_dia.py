import pytest

from tdfpy import DIA, DiaWindow, DiaWindowGroup, DIAMs1Frame, get_acquisition_type

D_PATH = "tests/data/example_dia.d"


def test_get_acquisition_type_dia():
    """Test that get_acquisition_type correctly identifies DIA data."""
    acq_type = get_acquisition_type(D_PATH)
    assert acq_type == "DIA"


def test_dia_context_manager():
    """Test that DIA opens and closes correctly as a context manager."""
    with DIA(D_PATH) as dia:
        assert dia.timsdata is not None


def test_dia_ms1_frames():
    """Test that MS1 frames are loaded correctly."""
    with DIA(D_PATH) as dia:
        ms1_frames = list(dia.ms1)
        assert len(ms1_frames) == 30

        # First MS1 frame
        f1 = dia.ms1[1]
        assert isinstance(f1, DIAMs1Frame)
        assert f1.frame_id == 1
        assert f1.time == pytest.approx(0.765083)
        assert f1.polarity == "positive"
        assert f1.scan_mode == 9
        assert f1.msms_type == 0
        assert f1.max_intensity == 1325
        assert f1.summed_intensities == 1925567
        assert f1.num_scans == 918
        assert f1.num_peaks == 57972
        assert f1.accumulation_time == pytest.approx(99.953)
        assert f1.ramp_time == pytest.approx(99.953)


def test_dia_windows():
    """Test that DIA windows are loaded correctly."""
    with DIA(D_PATH) as dia:
        all_windows = list(dia.windows)
        # 470 DIA MS2 frames * 2 sub-windows per group = 940 windows
        assert len(all_windows) == 940

        # First window: WindowGroup 1, IsolationMz 813.0, Frame 2
        w0 = all_windows[0]
        assert isinstance(w0, DiaWindow)
        assert w0.window_group == 1
        assert w0.isolation_mz == pytest.approx(813.0)
        assert w0.isolation_width == pytest.approx(26.0)
        assert w0.collision_energy == pytest.approx(37.191436)
        assert w0.scan_num_begin == 393
        assert w0.scan_num_end == 657
        assert w0.frame_id == 2
        assert w0.rt == pytest.approx(0.871160)
        assert w0.polarity == "positive"


def test_dia_window_properties():
    """Test computed properties on DIA windows."""
    with DIA(D_PATH) as dia:
        w0 = next(iter(dia.windows))
        assert w0.scan_num_range == (393, 657)
        # isolation_mz=813.0, isolation_width=26.0 -> mz_begin=800.0, mz_end=826.0
        assert w0.mz_begin == pytest.approx(800.0)
        assert w0.mz_end == pytest.approx(826.0)
        assert w0.mz_range == (w0.mz_begin, w0.mz_end)


def test_dia_window_groups():
    """Test that DiaWindowGroup objects are exposed via window_groups."""
    with DIA(D_PATH) as dia:
        groups = list(dia.window_groups)
        # 16 window groups, each with 2 sub-windows = 32 entries
        assert len(groups) == 32
        for g in groups:
            assert isinstance(g, DiaWindowGroup)

        # First entry is the first sub-window of group 1
        g0 = groups[0]
        assert g0.window_group == 1
        assert g0.isolation_mz == pytest.approx(813.0)
        assert g0.isolation_width == pytest.approx(26.0)
        assert g0.scan_num_begin == 393
        assert g0.scan_num_end == 657


def test_dia_window_lookup_by_group():
    """Test DiaWindowLookup indexing by window_group ID."""
    with DIA(D_PATH) as dia:
        # Window group 1 occurs in multiple frames; each occurrence contributes
        # 2 sub-windows. There are 30 frames assigned to group 1 -> 60 entries.
        group1_windows = dia.windows[1]
        assert len(group1_windows) == 60
        for w in group1_windows:
            assert w.window_group == 1

        # Invalid group raises KeyError
        with pytest.raises(KeyError):
            _ = dia.windows[99999]

        # Group 0 doesn't exist (groups start at 1)
        with pytest.raises(KeyError):
            _ = dia.windows[0]

        # .get() returns default for unknown groups
        assert dia.windows.get(99999) is None
        assert dia.windows.get(1) is group1_windows


def test_dia_window_query_by_rt():
    """Test querying DIA windows by retention time."""
    with DIA(D_PATH) as dia:
        # First DIA MS2 frame is at rt ~0.87s; query a small window around it
        results = list(dia.windows.query(rt=1.0, rt_tolerance=1.0))
        assert len(results) > 0
        for w in results:
            assert 0.0 <= w.rt <= 2.0


def test_dia_window_query_by_window_group():
    """Test querying DIA windows by window group."""
    with DIA(D_PATH) as dia:
        results = list(dia.windows.query(window_group_index=1))
        assert len(results) > 0
        # window_group_index here matches against the per-window window_index,
        # which corresponds to the row index in DiaFrameMsMsWindows.
        for w in results:
            assert w.window_index == 1


def test_dia_ms1_frame_lookup():
    """Test MS1 frame lookup features."""
    with DIA(D_PATH) as dia:
        f1 = dia.ms1[1]
        assert f1.frame_id == 1

        # Frame 2 is a DIA MS2 frame, not MS1
        with pytest.raises(KeyError):
            _ = dia.ms1[2]

        assert dia.ms1.get(99999) is None
        assert dia.ms1.get(1) is f1


def test_dia_ms1_frame_dia_windows_attr():
    """Test that DIAMs1Frame exposes a dia_windows tuple."""
    with DIA(D_PATH) as dia:
        f1 = dia.ms1[1]
        # dia_windows holds windows whose Frame ID equals the MS1 frame ID.
        # In this dataset, DIA windows live on the adjacent MS2 frames, so
        # the MS1 frame itself has no associated dia_windows.
        assert isinstance(f1.dia_windows, tuple)
        assert len(f1.dia_windows) == 0


def test_dia_ms1_frame_centroid():
    """Test that an MS1 frame can be centroided."""
    with DIA(D_PATH) as dia:
        f1 = dia.ms1[1]
        peaks = f1.centroid()
        # shape (N, 3): [m/z, intensity, 1/K0]
        assert peaks.ndim == 2
        assert peaks.shape[1] == 3
        assert peaks.shape[0] > 0


def test_dia_window_centroid():
    """Test that a DIA window can be centroided."""
    with DIA(D_PATH) as dia:
        w0 = next(iter(dia.windows))
        peaks = w0.centroid()
        assert peaks.ndim == 2
        assert peaks.shape[1] == 3


def test_dia_metadata():
    """Test that metadata is accessible via inherited properties."""
    with DIA(D_PATH) as dia:
        assert isinstance(dia.metadata.schema_type, str)
        assert isinstance(dia.metadata.instrument_name, str)


if __name__ == "__main__":
    pytest.main([__file__])
