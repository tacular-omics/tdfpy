"""Regression tests pinning the calibration conversions to reference values.

The reference values in ``tests/data/calibration_golden.json`` were captured from
Bruker's native ``timsdata`` library (see
``scripts/generate_calibration_golden.py``). They exist so the conversions stay
correct after the native library is gone: a pure-Python reimplementation is only
trustworthy if it reproduces these numbers.

Tolerances are deliberately tight — far tighter than any plausible "close enough"
error. The whole point is to fail on a wrong *model*, not merely on garbage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tdfpy.timsdata import TimsData, oneOverK0ToCCSforMz

DATA_DIR = Path("tests/data")
GOLDEN_PATH = DATA_DIR / "calibration_golden.json"

# m/z is compared relatively; 1e-8 is 0.01 ppm. The recovered model agrees with
# Bruker to ~1e-10 relative, so this leaves ~100x headroom for platform float
# differences while still catching an error 100000x smaller than one that would
# matter analytically.
MZ_RTOL = 1e-8
# Index/scan positions are compared in index units, mobility in 1/K0 units.
INDEX_ATOL = 1e-4
OOK0_ATOL = 1e-9
SCAN_ATOL = 1e-6
VOLTAGE_ATOL = 1e-9
# CCS uses the published Mason-Schamp constants, which sit ~7 ppm from Bruker's
# internal ones. That offset is analytically irrelevant (CCS is meaningful to
# ~0.1%) but it is real, so the tolerance accommodates it deliberately.
CCS_RTOL = 2e-5


def _golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden file not found: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text())


GOLDEN = _golden() if GOLDEN_PATH.exists() else {"fixtures": {}, "ccs": []}
FIXTURES = sorted(GOLDEN.get("fixtures", {}))


@pytest.mark.parametrize("fixture", FIXTURES)
def test_calibration_matches_golden(fixture: str) -> None:
    """Every captured conversion must reproduce its reference value."""
    d = DATA_DIR / fixture
    if not d.is_dir():
        pytest.skip(f"test data not found: {d}")

    entries = GOLDEN["fixtures"][fixture]["frames"]
    assert entries, "golden file has no frames for this fixture"

    with TimsData(str(d)) as td:
        for entry in entries:
            fid = entry["frame_id"]
            tof = np.asarray(entry["tof_indices"], dtype=np.float64)

            got_mz = np.asarray(td.indexToMz(fid, tof), dtype=np.float64)
            np.testing.assert_allclose(
                got_mz,
                entry["mz"],
                rtol=MZ_RTOL,
                atol=0,
                err_msg=f"{fixture} frame {fid}: index->mz drifted",
            )

            np.testing.assert_allclose(
                np.asarray(td.mzToIndex(fid, np.asarray(entry["mz"]))),
                entry["mz_to_index"],
                rtol=0,
                atol=INDEX_ATOL,
                err_msg=f"{fixture} frame {fid}: mz->index drifted",
            )

            scans = np.asarray(entry["scans"], dtype=np.float64)
            np.testing.assert_allclose(
                np.asarray(td.scanNumToOneOverK0(fid, scans)),
                entry["one_over_k0"],
                rtol=0,
                atol=OOK0_ATOL,
                err_msg=f"{fixture} frame {fid}: scan->1/K0 drifted",
            )

            np.testing.assert_allclose(
                np.asarray(
                    td.oneOverK0ToScanNum(fid, np.asarray(entry["one_over_k0"]))
                ),
                entry["one_over_k0_to_scan"],
                rtol=0,
                atol=SCAN_ATOL,
                err_msg=f"{fixture} frame {fid}: 1/K0->scan drifted",
            )

            np.testing.assert_allclose(
                np.asarray(td.scanNumToVoltage(fid, scans)),
                entry["voltage"],
                rtol=0,
                atol=VOLTAGE_ATOL,
                err_msg=f"{fixture} frame {fid}: scan->voltage drifted",
            )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_index_mz_roundtrip(fixture: str) -> None:
    """``mzToIndex`` must invert ``indexToMz`` on the captured grid."""
    d = DATA_DIR / fixture
    if not d.is_dir():
        pytest.skip(f"test data not found: {d}")

    with TimsData(str(d)) as td:
        for entry in GOLDEN["fixtures"][fixture]["frames"]:
            fid = entry["frame_id"]
            tof = np.asarray(entry["tof_indices"], dtype=np.float64)
            back = np.asarray(td.mzToIndex(fid, td.indexToMz(fid, tof)))
            np.testing.assert_allclose(back, tof, rtol=0, atol=INDEX_ATOL)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_ook0_scan_roundtrip(fixture: str) -> None:
    """``oneOverK0ToScanNum`` must invert ``scanNumToOneOverK0``."""
    d = DATA_DIR / fixture
    if not d.is_dir():
        pytest.skip(f"test data not found: {d}")

    with TimsData(str(d)) as td:
        for entry in GOLDEN["fixtures"][fixture]["frames"]:
            fid = entry["frame_id"]
            scans = np.asarray(entry["scans"], dtype=np.float64)
            back = np.asarray(
                td.oneOverK0ToScanNum(fid, td.scanNumToOneOverK0(fid, scans))
            )
            np.testing.assert_allclose(back, scans, rtol=0, atol=SCAN_ATOL)


def test_ccs_matches_golden() -> None:
    """CCS conversion must stay within the documented offset from Bruker."""
    probes = GOLDEN.get("ccs")
    if not probes:
        pytest.skip("golden file has no CCS probes")

    got = [oneOverK0ToCCSforMz(p["one_over_k0"], p["charge"], p["mz"]) for p in probes]
    np.testing.assert_allclose(got, [p["ccs"] for p in probes], rtol=CCS_RTOL, atol=0)


def test_mobility_is_monotonic_in_scan() -> None:
    """1/K0 must decrease monotonically with scan number.

    A sign flip or a swapped ramp endpoint is the most likely way to get mobility
    badly wrong while still producing plausible-looking numbers, and a
    value-by-value golden comparison alone would not name that failure clearly.
    """
    for fixture in FIXTURES:
        d = DATA_DIR / fixture
        if not d.is_dir():
            continue
        for entry in GOLDEN["fixtures"][fixture]["frames"]:
            ook0 = np.asarray(entry["one_over_k0"])
            assert np.all(np.diff(ook0) < 0), (
                f"{fixture} frame {entry['frame_id']}: 1/K0 is not decreasing in scan"
            )
