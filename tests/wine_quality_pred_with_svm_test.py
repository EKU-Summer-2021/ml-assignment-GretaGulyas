"""
    Support vector machine test.
"""

import unittest
import os
import pandas as pd
from src.wine_quality_pred_with_svm import SVMExample


class SVMTest(unittest.TestCase):
    """
        Support vector machine test class.
    """

    def setUp(self):
        """
            Setting up the dataset.
        """

        self.svm = SVMExample()
        self.wine_quality_dataset = pd.read_csv(
            'https://raw.githubusercontent.com'
            '/EKU-Summer-2021/ml-assignment-GretaGulyas/master/datasets/winequality-red.csv')
        self.x_train, self.x_test, self.y_train, self.y_test = self.svm.split_data(self.wine_quality_dataset)
        self.svm.train(self.x_train, self.y_train)
        param_grid = [
            {'kernel': ['rbf', 'sigmoid', 'linear'],
             'C': [0.5, 0.1, 1],
             'gamma': ['auto']}
        ]
        self.svm.grid_search(self.x_train, self.y_train, param_grid)

    def test_dataset_number_of_columns(self):
        """
            Testing the CSV file.
        """

        expected = 12
        actual = len(self.wine_quality_dataset.columns)
        self.assertEqual(expected, actual)

    def test_result_dir_exists(self):
        """
            Results dir exists or not.
        """

        expected = True
        actual = os.path.exists(os.path.join(os.getcwd(), 'Results'))
        self.assertEqual(expected, actual)

    def test_results_dir_not_empty(self):
        """
            Results dir empty or not.
        """

        expected = False
        actual = not os.listdir((os.path.join(os.getcwd(), 'Results')))
        self.assertEqual(expected, actual)
