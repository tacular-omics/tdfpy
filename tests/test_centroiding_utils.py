import numpy as np
import pytest

from tdfpy.centroiding import _sum_by_tof_index, batch_iterator, calculate_nmass

PROTON = 1.007276466812


class TestCalculateNmass:
    def test_charge_1(self):
        mz = 500.0
        charge = 1
        expected = mz * 1 - 1 * PROTON
        assert calculate_nmass(mz, charge) == pytest.approx(expected)

    def test_charge_2(self):
        mz = 250.5
        charge = 2
        expected = mz * 2 - 2 * PROTON
        assert calculate_nmass(mz, charge) == pytest.approx(expected)

    def test_charge_3(self):
        mz = 167.0
        charge = 3
        expected = mz * 3 - 3 * PROTON
        assert calculate_nmass(mz, charge) == pytest.approx(expected)

    def test_neutral_mass_positive(self):
        # Neutral mass should always be positive for typical peptide mz/charge
        assert calculate_nmass(500.0, 1) > 0
        assert calculate_nmass(250.0, 2) > 0

    def test_roundtrip(self):
        # If neutral_mass = mz * z - z * proton, then mz = (neutral_mass + z * proton) / z
        neutral = 1000.0
        for charge in (1, 2, 3):
            mz = (neutral + charge * PROTON) / charge
            assert calculate_nmass(mz, charge) == pytest.approx(neutral)


class TestBatchIterator:
    def test_even_split(self):
        items = list(range(6))
        batches = list(batch_iterator(items, 2))
        assert batches == [[0, 1], [2, 3], [4, 5]]

    def test_uneven_split(self):
        items = list(range(5))
        batches = list(batch_iterator(items, 2))
        assert batches == [[0, 1], [2, 3], [4]]

    def test_batch_larger_than_list(self):
        items = [1, 2, 3]
        batches = list(batch_iterator(items, 10))
        assert batches == [[1, 2, 3]]

    def test_empty_list(self):
        batches = list(batch_iterator([], 5))
        assert batches == []

    def test_batch_size_one(self):
        items = [10, 20, 30]
        batches = list(batch_iterator(items, 1))
        assert batches == [[10], [20], [30]]


class TestSumByTofIndex:
    """``_sum_by_tof_index`` picks between bincount and unique on data density.

    The choice is a pure performance decision, so the two branches must be
    indistinguishable in their output. The way to force each is the TOF grid
    width: a compact index range takes the bincount branch, a sparse one
    (same peaks, one index moved far out) takes the unique branch.
    """

    @staticmethod
    def _widen(tof: np.ndarray, grid_width: int) -> np.ndarray:
        """Same peaks, but with the grid stretched to ``grid_width``."""
        out = tof.copy()
        out[-1] = grid_width - 1
        return out

    def test_branches_agree_including_zero_intensities(self):
        base = np.array([5, 5, 9, 9, 12, 30], dtype=np.int64)
        intensity = np.array([0.0, 0.0, 3.0, 4.0, 0.0, 11.0])

        dense_keys, dense_sums = _sum_by_tof_index(base, intensity)
        # 6 peaks * 16 < 100_000, so this takes the unique branch.
        sparse = self._widen(base, 100_000)
        sparse_keys, sparse_sums = _sum_by_tof_index(sparse, intensity)

        # The moved peak aside, both branches must report the same bins.
        np.testing.assert_array_equal(dense_keys[:-1], sparse_keys[:-1])
        np.testing.assert_allclose(dense_sums, sparse_sums)
        # Bins totalling zero are dropped by both: 5 (0+0) and 12 (0) are gone.
        assert 5 not in dense_keys and 5 not in sparse_keys
        assert 12 not in dense_keys and 12 not in sparse_keys
        np.testing.assert_array_equal(dense_keys[:-1], [9])
        np.testing.assert_allclose(dense_sums, [7.0, 11.0])

    def test_all_zero_intensities_yield_nothing_on_either_branch(self):
        tof = np.array([1, 1, 4, 7], dtype=np.int64)
        zeros = np.zeros(4)
        for indices in (tof, self._widen(tof, 100_000)):
            keys, sums = _sum_by_tof_index(indices, zeros)
            assert keys.size == 0 and sums.size == 0

    def test_branches_agree_on_random_data(self):
        rng = np.random.default_rng(5)
        tof = rng.integers(0, 200, 300).astype(np.int64)
        intensity = rng.integers(0, 4, 300).astype(np.float64)  # plenty of zeros
        dense_keys, dense_sums = _sum_by_tof_index(tof, intensity)

        # Widen the grid past the 16x threshold to force the unique branch,
        # using an index that is already present so the bins do not change.
        sparse = tof.copy()
        moved = int(tof.max())
        sparse[tof == moved] = 100_000
        sparse_keys, sparse_sums = _sum_by_tof_index(sparse, intensity)

        remap = np.where(sparse_keys == 100_000, moved, sparse_keys)
        order = np.argsort(remap)
        np.testing.assert_array_equal(dense_keys, remap[order])
        np.testing.assert_allclose(dense_sums, sparse_sums[order])
        assert np.all(dense_sums > 0)

    def test_sorted_ascending_by_tof_index(self):
        tof = np.array([9, 2, 40, 2], dtype=np.int64)
        keys, sums = _sum_by_tof_index(tof, np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(keys, [2, 9, 40])
        np.testing.assert_allclose(sums, [6.0, 1.0, 3.0])
