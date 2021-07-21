import unittest
import pandas as pd

from src.concrete_strength_dataset_read import ReadDataset


class CSVReadDataTest(unittest.TestCase):

    def test_read(self):
        rd = ReadDataset(r'datasets/Concrete_Data_Yeh.csv')
        # given
        EXPECTED = True
        # when
        ACTUAL = isinstance(rd.dataset_read(), pd.DataFrame)
        # then
        self.assertEqual(EXPECTED, ACTUAL)
