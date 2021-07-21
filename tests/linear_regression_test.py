"""
    Linear regression test.
"""

import unittest
from src.linear_regression import LRExample
from src.concrete_strength_dataset_read import ReadDataset
from sklearn.linear_model import LinearRegression


class LinearRegressionTest(unittest.TestCase):
    """
        Linear regression test method.
    """

    def test_read(self):
        """
            Linear regression testing.
        """
        cs = ReadDataset(r'../datasets/Concrete_Data_Yeh.csv')
        lre = LRExample(r'../datasets/Concrete_Data_Yeh.csv')
        x_train, x_test, y_train, y_test = lre.split_data(cs)
        param_grid = [
            {'fit_intercept': [True, False],
             'normalize': [True, False]}
        ]
        # given
        EXPECTED = True
        # when
        ACTUAL = isinstance(lre.grid_search(param_grid, x_train, y_train), LinearRegression)
        # then
        self.assertEqual(EXPECTED, ACTUAL)

    def test_score(self):
        cs = ReadDataset(r'../datasets/Concrete_Data_Yeh.csv')
        lre = LRExample(r'../datasets/Concrete_Data_Yeh.csv')
        x_train, x_test, y_train, y_test = lre.split_data(cs)
        param_grid = [
            {'fit_intercept': [True, False],
             'normalize': [True, False]}
        ]
        # given
        EXPECTED = 0.5
        # when
        lre.grid_search(param_grid, x_train, y_train)
        ACTUAL = lre.search.best_score_
        # then
        self.assertGreater(ACTUAL, EXPECTED)
