import pytest

from tdfpy.centroiding import batch_iterator, calculate_nmass


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
