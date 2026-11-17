"""``Ms1FrameLookup`` / ``PrecursorLookup`` range queries answer from a sorted index.

Every test compares against a brute-force scan written from the old definition: keep an element
if ``_in`` accepts each bounded value, in lookup order. The index must return the same objects
in the same order.
"""

import inspect
import math
import random
from types import SimpleNamespace

import pytest

from tdfpy import DDA
from tdfpy.lookup import Ms1FrameLookup, PrecursorLookup

DDA_PATH = "tests/data/example_dda.d"
NAN = float("nan")


def _in(value, bounds):
    return bounds is None or bounds[0] <= value <= bounds[1]


def _scan_precursors(items, mz_range, rt_range):
    return [p for p in items if _in(p.precursor_mz, mz_range) and _in(p.rt, rt_range)]


def _scan_frames(items, rt_range):
    return [f for f in items if _in(f.rt, rt_range)]


def _random_range(rng, values):
    kind = rng.choice(["none", "empty", "all", "stored", "random", "point", "reversed", "nan"])
    finite = [v for v in values if math.isfinite(v)]
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    if kind == "none":
        return None
    if kind == "empty":
        return (hi + span, hi + 2 * span)
    if kind == "all":
        return (lo - span, hi + span)
    if kind == "stored":
        a, b = sorted(rng.sample(finite, 2))
        return (a, b)
    if kind == "point":
        v = rng.choice(finite)
        return (v, v)
    if kind == "reversed":
        return (hi, lo)
    if kind == "nan":
        return rng.choice([(NAN, hi), (lo, NAN), (NAN, NAN)])
    a = rng.uniform(lo - span / 10, hi + span / 10)
    return (a, a + rng.uniform(0, span / 3))


def _synthetic_precursors(rng, n=120):
    # Few distinct values so ties and duplicated bounds occur; some NaN m/z and RT.
    mzs = [round(rng.uniform(300, 320), 1) for _ in range(n)]
    rts = [round(rng.uniform(0, 30), 0) for _ in range(n)]
    mzs[3] = mzs[50] = NAN
    rts[7] = rts[80] = NAN
    shuffled = list(range(n))
    rng.shuffle(shuffled)  # lookup order is not sorted order
    return {i: SimpleNamespace(precursor_mz=mzs[i], rt=rts[i], tag=i) for i in shuffled}


def test_precursor_query_range_matches_scan_on_random_ranges():
    rng = random.Random(77)
    mapping = _synthetic_precursors(rng)
    lookup = PrecursorLookup(mapping)
    items = list(mapping.values())
    mzs = [p.precursor_mz for p in items]
    rts = [p.rt for p in items]
    for _ in range(500):
        mz_range, rt_range = _random_range(rng, mzs), _random_range(rng, rts)
        got = list(lookup.query_range(precursor_mz_range=mz_range, rt_range=rt_range))
        want = _scan_precursors(items, mz_range, rt_range)
        assert [p.tag for p in got] == [p.tag for p in want], (mz_range, rt_range)
        assert all(a is b for a, b in zip(got, want, strict=True))


def test_ms1_query_range_matches_scan_on_random_ranges():
    rng = random.Random(78)
    rts = [round(rng.uniform(0, 100), 0) for _ in range(150)]
    rts[10] = NAN
    order = list(range(len(rts)))
    rng.shuffle(order)
    mapping = {i: SimpleNamespace(rt=rts[i], tag=i) for i in order}
    lookup = Ms1FrameLookup(mapping)
    items = list(mapping.values())
    for _ in range(500):
        rt_range = _random_range(rng, rts)
        got = list(lookup.query_range(rt_range=rt_range))
        want = _scan_frames(items, rt_range)
        assert [f.tag for f in got] == [f.tag for f in want], rt_range


def test_bounds_on_stored_values_are_inclusive():
    mapping = {i: SimpleNamespace(precursor_mz=mz, rt=float(i), tag=i) for i, mz in enumerate([5.0, 3.0, 3.0, 4.0, 1.0])}
    lookup = PrecursorLookup(mapping)
    assert [p.tag for p in lookup.query_range(precursor_mz_range=(3.0, 4.0))] == [1, 2, 3]
    assert [p.tag for p in lookup.query_range(precursor_mz_range=(3.0, 3.0))] == [1, 2]
    assert [p.tag for p in lookup.query_range(rt_range=(1.0, 3.0))] == [1, 2, 3]
    assert [p.tag for p in lookup.query_range(precursor_mz_range=(1.0, 5.0), rt_range=(4.0, 4.0))] == [4]


def test_none_ranges_yield_everything_in_lookup_order_and_nan_never_matches_a_bounded_range():
    mapping = {1: SimpleNamespace(precursor_mz=NAN, rt=2.0), 2: SimpleNamespace(precursor_mz=1.0, rt=NAN), 3: SimpleNamespace(precursor_mz=2.0, rt=1.0)}
    lookup = PrecursorLookup(mapping)
    assert list(lookup.query_range()) == list(mapping.values())
    assert list(lookup.query_range(precursor_mz_range=(-math.inf, math.inf))) == [mapping[2], mapping[3]]
    assert list(lookup.query_range(rt_range=(-math.inf, math.inf))) == [mapping[1], mapping[3]]


def test_unorderable_values_keep_the_scan_behaviour():
    # The types say float, but ``_in`` on None raised on a bounded range and passed an unbounded one.
    mapping = {1: SimpleNamespace(precursor_mz=None, rt=1.0), 2: SimpleNamespace(precursor_mz=2.0, rt=2.0)}
    lookup = PrecursorLookup(mapping)
    assert list(lookup.query_range()) == list(mapping.values())
    with pytest.raises(TypeError):
        list(lookup.query_range(precursor_mz_range=(0.0, 3.0)))


def test_query_range_is_a_keyword_only_generator():
    lookup = PrecursorLookup({1: SimpleNamespace(precursor_mz=1.0, rt=1.0)})
    assert inspect.isgenerator(lookup.query_range(precursor_mz_range=(0.0, 2.0)))
    assert inspect.isgenerator(Ms1FrameLookup({1: SimpleNamespace(rt=1.0)}).query_range())
    with pytest.raises(TypeError):
        lookup.query_range((0.0, 2.0))  # type: ignore[misc]


def test_real_dda_lookups_match_scan():
    rng = random.Random(79)
    with DDA(DDA_PATH) as reader:
        items = list(reader.precursors)
        mzs = [p.precursor_mz for p in items]
        rts = [p.rt for p in items]
        for _ in range(200):
            mz_range, rt_range = _random_range(rng, mzs), _random_range(rng, rts)
            got = list(reader.precursors.query_range(precursor_mz_range=mz_range, rt_range=rt_range))
            want = _scan_precursors(items, mz_range, rt_range)
            assert [p.precursor_id for p in got] == [p.precursor_id for p in want]
        # ``query`` turns a centre and tolerance into the same range.
        for mz in mzs[:50]:
            width = mz * 10 / 1e6
            got = list(reader.precursors.query(precursor_mz=mz, mz_tolerance=10, mz_tolerance_unit="ppm", rt_tolerance=1e9))
            assert [p.precursor_id for p in got] == [p.precursor_id for p in _scan_precursors(items, (mz - width, mz + width), None)]

        frames = list(reader.ms1)
        frts = [f.rt for f in frames]
        for _ in range(200):
            rt_range = _random_range(rng, frts)
            got = list(reader.ms1.query_range(rt_range=rt_range))
            assert [f.frame_id for f in got] == [f.frame_id for f in _scan_frames(frames, rt_range)]


class _Counted:
    """Counts reads of ``precursor_mz`` and ``rt`` so a test can tell a scan from an index lookup."""

    reads = 0

    def __init__(self, mz, rt):
        self._mz, self._rt = mz, rt

    @property
    def precursor_mz(self):
        type(self).reads += 1
        return self._mz

    @property
    def rt(self):
        type(self).reads += 1
        return self._rt


def test_a_narrow_query_does_not_read_every_element():
    n = 2000
    mapping = {i: _Counted(100.0 + i * 0.01, float(i)) for i in range(n)}
    precursors = PrecursorLookup(mapping)
    frames = Ms1FrameLookup(mapping)
    _Counted.reads = 0
    hits = list(precursors.query_range(precursor_mz_range=(110.0, 110.05)))
    assert len(hits) == 6
    assert list(frames.query_range(rt_range=(5.0, 7.0))) == [mapping[5], mapping[6], mapping[7]]
    assert list(precursors.query_range(rt_range=(5.0, 7.0))) == [mapping[5], mapping[6], mapping[7]]
    assert _Counted.reads < 100, f"{_Counted.reads} attribute reads for three narrow queries over {n} elements: this is a scan"
