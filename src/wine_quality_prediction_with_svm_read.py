"""
    Reads the dataset for the wine quality prediction with SVM.
"""

# pylint: disable=too-few-public-methods

import pandas as pd


class ReadData:
    """
        Class with method that reads the dataset.
    """

    def __init__(self, dataset_location):
        self.dataset = dataset_location
        self.svm_dataset = self.dataset_read()

    def dataset_read(self):
        """
            Function for reading the CSV file.
        """

        dataset = pd.read_csv(self.dataset)
        return dataset
