import unittest
import pandas as pd

from src.strength_prediction_with_lr import ReadDataset


class CSVReadDataTest(unittest.TestCase):

    def test_read(self):
        rd = ReadDataset(r'datasets/Concrete_Data_Yeh.csv')
        # given
        EXPECTED = True
        # when
        ACTUAL = isinstance(rd.concrete_strength_dataset_read(), pd.DataFrame)
        # then
        self.assertEqual(EXPECTED, ACTUAL)
