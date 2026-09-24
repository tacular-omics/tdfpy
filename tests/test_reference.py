"""tdfpy checked against sources that do not share its code.

* CCS: the Mason-Schamp equation evaluated here from CODATA 2018 constants, and
  published drift-tube CCS values for the Agilent ESI-L tune mix.
* Metadata: the DDA/DIA/PRM readers compared field by field with direct sqlite3
  queries on the bundled fixtures.
* Centroiding: merge_peaks on hand-built clusters against hand-computed
  intensity-weighted means.
"""

import math
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tdfpy import DDA, DIA, PRM
from tdfpy.calibration import _CCS_K, ccs_to_one_over_k0, one_over_k0_to_ccs
from tdfpy.centroiding import _HAS_NUMBA, merge_peaks
from tdfpy.elems import Polarity

DATA = Path("tests/data")

# CODATA 2018 exact / recommended values.
E_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23  # J/K
DALTON = 1.66053906660e-27  # kg
# Loschmidt number: gas number density at 273.15 K, 101.325 kPa (the N0 of the
# reduced mobility K0, whose unit is cm^2/(V s)).
N0 = 101325.0 / (K_B * 273.15)  # m^-3

N2_MASS = 28.013  # Da, the drift gas Bruker uses
TEMPERATURE = 305.0  # K, Bruker's convention


def _mason_schamp_ccs(one_over_k0: float, charge: int, mz: float) -> float:
    """CCS in Å^2 from 1/K0 in V s/cm^2, written out from the Mason-Schamp equation.

    CCS = 3 z e / (16 N0) * sqrt(2 pi / (mu k_B T)) * (1 / K0)
    """
    mass = mz * charge  # Bruker's convention: no proton subtraction
    mu = mass * N2_MASS / (mass + N2_MASS) * DALTON
    k0 = 1.0 / one_over_k0 * 1e-4  # cm^2/(V s) -> m^2/(V s)
    ccs_m2 = 3.0 * charge * E_CHARGE / (16.0 * N0) * math.sqrt(2.0 * math.pi / (mu * K_B * TEMPERATURE)) / k0
    return ccs_m2 * 1e20


def test_ccs_constant_matches_codata():
    """_CCS_K folds the Mason-Schamp prefactor; recompute it from CODATA 2018 constants."""
    expected = 3.0 / 16.0 * E_CHARGE / N0 * math.sqrt(2.0 * math.pi / (K_B * DALTON)) * 1e4 * 1e20
    # Bruker's constant predates CODATA 2018; the residual is 0.17 ppm.
    assert _CCS_K == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("charge", [1, 2, 3, 4])
@pytest.mark.parametrize("mz", [100.0, 445.12, 1000.0, 1800.0])
@pytest.mark.parametrize("one_over_k0", [0.6, 0.95, 1.3, 1.7])
def test_ccs_matches_mason_schamp(one_over_k0, charge, mz):
    ccs = one_over_k0_to_ccs(one_over_k0, charge, mz)
    assert ccs == pytest.approx(_mason_schamp_ccs(one_over_k0, charge, mz), rel=1e-6)
    assert ccs_to_one_over_k0(ccs, charge, mz) == pytest.approx(one_over_k0, rel=1e-12)


# Agilent ESI-L tune mix, [M+H]+ / singly charged ions. 1/K0 from Bruker's
# timsTOF tune-mix calibration table; DTCCS_N2 from Stow et al., Anal. Chem.
# 2017, 89, 9048-9055 (interlaboratory drift-tube study).
TUNE_MIX = [
    (622.0290, 0.9915, 202.96),
    (922.0098, 1.1986, 243.64),
    (1221.9906, 1.3934, 282.20),
]


@pytest.mark.parametrize(("mz", "one_over_k0", "published_ccs"), TUNE_MIX)
def test_ccs_matches_published_tune_mix(mz, one_over_k0, published_ccs):
    assert one_over_k0_to_ccs(one_over_k0, 1, mz) == pytest.approx(published_ccs, rel=2e-4)


def _rows(d: Path, sql: str) -> list[sqlite3.Row]:
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql).fetchall()


def _need(d: Path) -> None:
    if not d.is_dir():
        pytest.skip("Test data not available")


def _check_frame(frame, row) -> None:
    assert frame.time == row["Time"]
    assert frame.polarity is Polarity.from_str(row["Polarity"])
    assert frame.scan_mode == row["ScanMode"]
    assert frame.msms_type == row["MsMsType"]
    assert frame.tims_id == row["TimsId"]
    assert frame.max_intensity == row["MaxIntensity"]
    assert frame.summed_intensities == row["SummedIntensities"]
    assert frame.num_scans == row["NumScans"]
    assert frame.num_peaks == row["NumPeaks"]
    assert frame.mz_calibration == row["MzCalibration"]
    assert frame.t1 == row["T1"]
    assert frame.t2 == row["T2"]
    assert frame.tims_calibration == row["TimsCalibration"]
    assert frame.property_group == row["PropertyGroup"]
    assert frame.accumulation_time == row["AccumulationTime"]
    assert frame.ramp_time == row["RampTime"]


@pytest.mark.parametrize(("reader", "name"), [(DDA, "example_dda.d"), (DIA, "example_dia.d"), (PRM, "example_prm.d")])
def test_ms1_frames_match_sqlite(reader, name):
    d = DATA / name
    _need(d)
    rows = {r["Id"]: r for r in _rows(d, "SELECT * FROM Frames WHERE MsMsType = 0")}
    with reader(d) as folder:
        frames = {f.frame_id: f for f in folder.ms1}
        assert frames.keys() == rows.keys()
        for frame_id, frame in frames.items():
            _check_frame(frame, rows[frame_id])


def test_dda_precursors_and_pasef_windows_match_sqlite():
    d = DATA / "example_dda.d"
    _need(d)
    times = {r["Id"]: r["Time"] for r in _rows(d, "SELECT Id, Time FROM Frames")}
    precursors = {r["Id"]: r for r in _rows(d, "SELECT * FROM Precursors")}
    windows: dict[int, list[sqlite3.Row]] = {}
    for r in _rows(d, "SELECT * FROM PasefFrameMsMsInfo ORDER BY Frame, ScanNumBegin"):
        windows.setdefault(r["Precursor"], []).append(r)

    with DDA(d) as dda:
        got = {p.precursor_id: p for p in dda.precursors}
        assert got.keys() == precursors.keys()
        for pid, p in got.items():
            row = precursors[pid]
            assert p.largest_peak_mz == row["LargestPeakMz"]
            assert p.average_mz == row["AverageMz"]
            assert p.monoisotopic_mz == row["MonoisotopicMz"]
            assert p.charge == row["Charge"]
            assert p.scan_number == row["ScanNumber"]
            assert p.intensity == row["Intensity"]
            assert p.parent_frame == row["Parent"]
            assert p.rt == times[row["Parent"]]
            got_windows = sorted(p.pasef_frame_msms_infos, key=lambda w: (w.frame_id, w.scan_num_begin))
            expected = windows.get(pid, [])
            assert len(got_windows) == len(expected)
            for w, wr in zip(got_windows, expected, strict=True):
                # PasefFrameMsMsInfo.Frame is the MS/MS frame the window was acquired in.
                assert w.frame_id == wr["Frame"]
                assert w.rt == times[wr["Frame"]]
                assert (w.scan_num_begin, w.scan_num_end) == (wr["ScanNumBegin"], wr["ScanNumEnd"])
                assert w.isolation_mz == wr["IsolationMz"]
                assert w.isolation_width == wr["IsolationWidth"]
                assert w.collision_energy == wr["CollisionEnergy"]
                assert w.precursor == pid
        # Each MS1 frame lists exactly the precursors whose Parent it is.
        for frame in dda.ms1:
            assert sorted(p.precursor_id for p in frame.precursors) == sorted(i for i, r in precursors.items() if r["Parent"] == frame.frame_id)


def test_dia_windows_match_sqlite():
    d = DATA / "example_dia.d"
    _need(d)
    times = {r["Id"]: r["Time"] for r in _rows(d, "SELECT Id, Time FROM Frames")}
    groups = {r["Frame"]: r["WindowGroup"] for r in _rows(d, "SELECT * FROM DiaFrameMsMsInfo")}
    defs: dict[int, list[tuple]] = {}
    for r in _rows(d, "SELECT * FROM DiaFrameMsMsWindows"):
        defs.setdefault(r["WindowGroup"], []).append((r["ScanNumBegin"], r["ScanNumEnd"], r["IsolationMz"], r["IsolationWidth"], r["CollisionEnergy"]))

    with DIA(d) as dia:
        by_frame: dict[int, list[tuple]] = {}
        for w in dia.windows:
            assert w.window_group == groups[w.frame_id]
            assert w.rt == times[w.frame_id]
            by_frame.setdefault(w.frame_id, []).append((w.scan_num_begin, w.scan_num_end, w.isolation_mz, w.isolation_width, w.collision_energy))
        assert by_frame.keys() == groups.keys()
        for frame_id, got in by_frame.items():
            assert sorted(got) == sorted(defs[groups[frame_id]])


KERNELS = [pytest.param(False, id="python"), pytest.param(True, id="numba", marks=pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed"))]


@pytest.mark.parametrize("use_numba", KERNELS)
def test_merge_peaks_weighted_mean_by_hand(use_numba):
    # Two clusters 1 Da apart; weights chosen so the means are easy to check by hand.
    mz = np.array([500.000, 500.002, 500.004, 501.000, 501.001])
    intensity = np.array([100.0, 300.0, 100.0, 50.0, 150.0])
    im = np.array([1.00, 1.01, 1.02, 0.80, 0.80])
    out = merge_peaks(mz, intensity, im, mz_tolerance=10.0, im_tolerance=0.05, min_peaks=1, use_numba=use_numba)
    # Seeds are taken brightest first: the 300-count point, then the 150-count point.
    np.testing.assert_allclose(out[:, 0], [500.002, 501.00075], rtol=0, atol=1e-9)
    np.testing.assert_allclose(out[:, 1], [500.0, 200.0])
    np.testing.assert_allclose(out[:, 2], [1.01, 0.80], rtol=0, atol=1e-12)
