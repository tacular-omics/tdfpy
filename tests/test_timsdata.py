import unittest

from tdfpy import timsdata

TDF_PATH = r"tests/data/example_dda.d"


class TestTimsData(unittest.TestCase):
    def test_timsdata(self):
        with timsdata.timsdata_connect(TDF_PATH) as td:
            self.assertTrue(td.conn is not None)


if __name__ == "__main__":
    unittest.main()
