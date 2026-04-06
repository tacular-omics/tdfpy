import pytest

from tdfpy import PRM, PrmTarget, PrmTransition, get_acquisition_type

D_PATH = "tests/data/20260328_IRT_DDA_30spd_MS_long_Method_v06_S1-E5_1_2345.d"


def test_get_acquisition_type_prm():
    """Test that get_acquisition_type correctly identifies PRM data."""
    acq_type = get_acquisition_type(D_PATH)
    assert acq_type == "PRM"


def test_prm_context_manager():
    """Test that PRM opens and closes correctly as a context manager."""
    with PRM(D_PATH) as prm:
        assert prm.timsdata is not None


def test_prm_targets():
    """Test that PRM targets are loaded correctly."""
    with PRM(D_PATH) as prm:
        targets = list(prm.targets)
        assert len(targets) == 11

        # Target 1
        t1 = prm.targets[1]
        assert isinstance(t1, PrmTarget)
        assert t1.target_id == 1
        assert t1.external_id is None
        assert t1.time == pytest.approx(354.0)
        assert t1.one_over_k0 == pytest.approx(0.81)
        assert t1.monoisotopic_mz == pytest.approx(487.26)
        assert t1.charge == 2
        assert t1.description == ""

        # Target 2
        t2 = prm.targets[2]
        assert t2.monoisotopic_mz == pytest.approx(644.82)
        assert t2.time == pytest.approx(864.0)
        assert t2.one_over_k0 == pytest.approx(0.945, abs=0.001)
        assert t2.charge == 2


def test_prm_target_lookup_by_id():
    """Test PrmTargetLookup indexing and error handling."""
    with PRM(D_PATH) as prm:
        t1 = prm.targets[1]
        assert t1.target_id == 1

        with pytest.raises(KeyError):
            _ = prm.targets[99999]

        assert prm.targets.get(99999) is None
        assert prm.targets.get(1) is t1


def test_prm_target_query_by_mz():
    """Test querying PRM targets by m/z."""
    with PRM(D_PATH) as prm:
        results = list(
            prm.targets.query(mz=487.26, mz_tolerance=0.01, mz_tolerance_type="da")
        )
        assert len(results) >= 1
        assert any(t.target_id == 1 for t in results)


def test_prm_target_query_by_rt():
    """Test querying PRM targets by retention time."""
    with PRM(D_PATH) as prm:
        results = list(prm.targets.query(rt=354.0, rt_tolerance=1.0))
        assert len(results) >= 1
        assert any(t.target_id == 1 for t in results)


def test_prm_target_query_by_ook0():
    """Test querying PRM targets by 1/K0."""
    with PRM(D_PATH) as prm:
        results = list(prm.targets.query(ook0=0.81, ook0_tolerance=0.01))
        assert len(results) >= 1
        assert any(t.target_id == 1 for t in results)


def test_prm_transitions():
    """Test that PRM transitions are loaded correctly."""
    with PRM(D_PATH) as prm:
        transitions = list(prm.transitions)
        assert len(transitions) == 3308


def test_prm_target_transitions():
    """Test that PrmTarget.transitions holds all transitions for that target."""
    with PRM(D_PATH) as prm:
        t1 = prm.targets[1]
        assert len(t1.transitions) > 0
        assert len(t1.transitions) == len(prm.transitions[1])
        for tr in t1.transitions:
            assert isinstance(tr, PrmTransition)
            assert tr.target is t1


def test_prm_transition_lookup_by_target():
    """Test PrmTransitionLookup indexing by target ID."""
    with PRM(D_PATH) as prm:
        t1_transitions = prm.transitions[1]
        assert len(t1_transitions) > 0
        for t in t1_transitions:
            assert isinstance(t, PrmTransition)
            assert t.target.target_id == 1

        with pytest.raises(KeyError):
            _ = prm.transitions[99999]


def test_prm_transition_fields():
    """Test PRM transition field values from the first row."""
    with PRM(D_PATH) as prm:
        # Frame 275, Target 1
        t1_transitions = prm.transitions[1]
        tr = next(t for t in t1_transitions if t.frame_id == 275)

        assert tr.scan_num_begin == 1492
        assert tr.scan_num_end == 1565
        assert tr.isolation_mz == pytest.approx(487.26)
        assert tr.isolation_width == pytest.approx(1.0)
        assert tr.collision_energy == pytest.approx(20.0)
        assert tr.polarity == "positive"


def test_prm_transition_target_reference():
    """Test that transitions correctly reference their PRM target."""
    with PRM(D_PATH) as prm:
        t1_transitions = prm.transitions[1]
        tr = t1_transitions[0]
        assert tr.target is prm.targets[1]
        assert tr.target.monoisotopic_mz == pytest.approx(487.26)


def test_prm_transition_properties():
    """Test computed properties on PRM transitions."""
    with PRM(D_PATH) as prm:
        t1_transitions = prm.transitions[1]
        tr = t1_transitions[0]

        assert tr.scan_num_range == (1492, 1565)
        assert tr.mz_begin == pytest.approx(486.76)
        assert tr.mz_end == pytest.approx(487.76)
        assert tr.mz_range == (tr.mz_begin, tr.mz_end)


def test_prm_transition_query_by_rt():
    """Test querying transitions by retention time."""
    with PRM(D_PATH) as prm:
        # Frame 275 has RT close to the start
        first_tr = prm.transitions[1][0]
        results = list(prm.transitions.query(target=1, rt=first_tr.rt, rt_tolerance=5.0))
        assert len(results) >= 1
        assert all(t.target.target_id == 1 for t in results)


def test_prm_ms1_frames():
    """Test that MS1 frames are loaded correctly."""
    with PRM(D_PATH) as prm:
        ms1_frames = list(prm.ms1)
        assert len(ms1_frames) == 1714

        # First MS1 frame
        f1 = prm.ms1[1]
        assert f1.frame_id == 1
        assert f1.time == pytest.approx(1.723616)
        assert f1.polarity == "positive"
        assert f1.msms_type == 0


def test_prm_ms1_frame_lookup():
    """Test MS1 frame lookup features."""
    with PRM(D_PATH) as prm:
        f1 = prm.ms1[1]
        assert f1.frame_id == 1

        with pytest.raises(KeyError):
            # Frame 275 is PRM MS2, not in ms1 lookup
            _ = prm.ms1[275]


def test_prm_metadata():
    """Test that metadata is accessible via inherited properties."""
    with PRM(D_PATH) as prm:
        assert isinstance(prm.metadata.schema_type, str)
        assert isinstance(prm.metadata.instrument_name, str)


if __name__ == "__main__":
    pytest.main([__file__])
